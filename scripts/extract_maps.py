import itertools
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append("./carla-0.9.6-py3.5-linux-x86_64.egg")  # isort:skip
import carla
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np


class WaypointNode:
    def __init__(
        self,
        bin: Tuple[int, int],
        waypoint: carla.Waypoint,
        children: Optional[List[Tuple[int, int]]] = None,
    ):
        self.bin = bin
        self.waypoint = waypoint
        if children is None:
            children = []
        self.children = children

    def __eq__(self, other):
        return other.bin == self.bin

    def __hash__(self):
        return hash(self.bin)

    @property
    def has_children(self):
        return len(self.children) > 0

    def add_child(self, waypoint_bin: Tuple[int, int]):
        if waypoint_bin in self.children:
            raise RuntimeError(
                f"Waypoint {self.waypoint} has multiple edges to same bin {waypoint_bin}"
            )

        self.children.append(waypoint_bin)

    def remove_child(self, waypoint_bin: Tuple[int, int]):
        if waypoint_bin not in self.children:
            raise ValueError(
                f'Cannot remove child with bin "{waypoint_bin}" as it is not in child list'
            )
        self.children.remove(waypoint_bin)

    def parent_of(self, other):
        return self.parent_of_bin(other.bin)

    def parent_of_bin(self, waypoint_bin: Tuple[int, int]):
        return waypoint_bin in self.children

    def distance(self, other):
        return math.sqrt(
            (other.waypoint.transform.location.x - self.waypoint.transform.location.x)
            ** 2
            + (other.waypoint.transform.location.y - self.waypoint.transform.location.y)
            ** 2
        )


class WaypointGraph:
    _nodes: Dict[Tuple[int, int], WaypointNode]
    x_bins: np.ndarray
    y_bins: np.ndarray

    def __init__(self, x_bins, y_bins):
        self._nodes = {}
        self.x_bins = x_bins
        self.y_bins = y_bins

    def town02_5m_patch(self) -> None:
        """Manually fix asthetically strange graph topo in Town02"""
        node1 = self.get_node_by_bin((27737, 23804))
        node1.children.append((28244, 24339))
        node2 = self.get_node_by_bin((28986, 23805))
        try:
            idx = node2.children.index((28244, 24339))
        except ValueError as err:
            print(
                "Error: could not find erroneous edge in graph.  Is this the right map parameterization?"
            )
            print(err)
            return
        del node2.children[idx]
        return

    @property
    def waypoint_locations(self) -> List[Tuple[float]]:
        return [
            (node.waypoint.transform.location.x, node.waypoint.transform.location.y)
            for node in self.get_all_nodes()
        ]

    def generate_matrix(self) -> np.ndarray:
        max_children = max([len(node.children) for node in self.get_all_nodes()])
        matrix = np.zeros((len(self.get_all_nodes()), max_children))

        ordered_graph = {
            data[0]: (idx, data[1]) for idx, data in enumerate(self._nodes.items())
        }

        for node_bin, data in ordered_graph.items():
            node_idx, node = data
            matrix[node_idx] = node_idx
            for action_idx, child in enumerate(node.children):
                child_idx, _ = ordered_graph[child]
                matrix[node_idx, action_idx] = child_idx
        return matrix

    def get_node_locations(self) -> np.ndarray:
        return np.array(
            [
                (node.waypoint.transform.location.x, node.waypoint.transform.location.y)
                for node in self.get_all_nodes()
            ]
        )

    def merge_nodes(self, node1: WaypointNode, node2: WaypointNode):
        if node1.parent_of(node2) and node2.parent_of(node1):
            print("Warning: node loop detected on merge")
            # TODO: refactor to handle case explicitly instead of falling through
            node1.remove_child(node2.bin)

        if node1.parent_of(node2):
            parent = node1
            child = node2
        elif node2.parent_of(node1):
            parent = node2
            child = node1
        else:
            raise ValueError("Cannot merge nodes with no direct relation")
        parent.remove_child(child.bin)
        self.replace_node(parent, child)

    def remove_node(self, target: WaypointNode):
        if target not in self.get_all_nodes():
            raise ValueError(f"Cannot remove node {target}, doesn't exist in graph")
        if target.has_children:
            raise ValueError(f"Cannot remove node {target}, has children")
        if len(self.get_waypoint_parents(target)) > 0:
            raise ValueError(f"Cannot remove node {target}, has parents")
        self._nodes.pop(target.bin)

    def get_waypoint_parents(self, target: WaypointNode):
        return [node for node in self.get_all_nodes() if node.parent_of(target)]

    def replace_node(self, target: WaypointNode, replacement: WaypointNode):
        if target == replacement:
            raise ValueError("Cannot replace node same node")
        if target.parent_of(replacement) or replacement.parent_of(target):
            raise ValueError(
                "Cannot replace node with direct connection, use merge_nodes instead"
            )

        for child_bin in list(target.children):
            target.remove_child(child_bin)
            if child_bin not in replacement.children:
                replacement.add_child(child_bin)
        for node in self.get_all_nodes():
            if target.bin in node.children:
                node.remove_child(target.bin)
                if replacement.bin not in node.children:
                    node.add_child(replacement.bin)
        self.remove_node(target)

    # TODO: replace with handling of loopbacks at init
    def remove_loopback_children(self):
        for node in self.get_all_nodes():
            try:
                node.remove_child(node.bin)
            except ValueError:
                pass

    def remove_orphans(self):
        while self.has_orphans:
            for node in self.get_all_nodes():
                if len(self.get_waypoint_parents(node)) == 0:
                    for child in node.children:
                        node.remove_child(child)
                    self.remove_node(node)

    def remove_childless(self):
        while self.has_childless:
            for node in self.get_all_nodes():
                if not node.has_children:
                    self.remove_node(node)

    # TODO: refactor out, should not be necessary
    def remove_twins(self):
        for node in self.get_all_nodes():
            node.children = list(set(node.children))

    def merge_close_neighbors(self, threshold: float):
        all_nodes = self.get_all_nodes()
        for node in all_nodes:
            if node not in self.get_all_nodes():
                continue
            new_children = []
            for child_bin in node.children:
                if child_bin not in [n.bin for n in self.get_all_nodes()]:
                    continue
                child = self.get_node_by_bin(child_bin)
                if node.distance(child) < threshold:
                    self.merge_nodes(node, child)

    def insert_node(self, parent: WaypointNode, child: WaypointNode, new: WaypointNode):
        if not parent.parent_of(child):
            raise ValueError(
                f"Cannot insert node, parent node {parent} has no child {child}"
            )
        if parent == new or child == new:
            raise ValueError(f"Cannot insert node, new node is parent or child node")

        # TODO: refactor for edge case where interpolated node is already in map
        parent.remove_child(child.bin)
        parent.add_child(new.bin)
        new.add_child(child.bin)
        self._nodes[new.bin] = new

    def insert_chain(
        self, parent: WaypointNode, child: WaypointNode, chain: List[WaypointNode]
    ):
        if not parent.parent_of(child):
            raise ValueError(
                f"Cannot insert chain, parent node {parent} has no child {child}"
            )
        if parent in chain or child in chain:
            raise ValueError(f"Cannot insert chain, new node is parent or child node")

        if len(chain) == 0:
            raise ValueError(f"Cannot insert chain, chain contains no nodes")

        parent.remove_child(child.bin)
        # TODO: refactor to avoid direct catching of non-parentage
        if not parent.parent_of(chain[0]):
            parent.add_child(chain[0].bin)
        chain[-1].add_child(child.bin)
        for node in chain:
            # TODO: implement wrapper function to avoid direct dict assignment
            self._nodes[node.bin] = node

    # FIXME: handle edge case where nodes have multiple next waypoints
    def get_waypoint_next(self, node: WaypointNode, distance: float) -> WaypointNode:
        waypoint_list = node.waypoint.next(distance)
        if len(waypoint_list) == 0:
            raise RuntimeError(
                f"Failure in node interpolation, node {node} has no next waypoint"
            )
        if len(waypoint_list) > 1:
            raise RuntimeError(
                f"Failure in node interpolation, node {node} has multiple next waypoints"
            )
        return WaypointNode(self.waypoint_to_bin(waypoint_list[0]), waypoint_list[0])

    def _interpolate_nodes_coarse(
        self, node_start: WaypointNode, node_end: WaypointNode, distance: float
    ) -> List[WaypointNode]:
        inter_nodes = [node_start]
        while inter_nodes[-1].distance(node_end) > 2 * distance:
            inter_nodes.append(self.get_waypoint_next(inter_nodes[-1], distance))
        if inter_nodes[-1].distance(node_end) >= (4.0 / 3.0) * distance:
            inter_nodes.append(
                self.get_waypoint_next(
                    inter_nodes[-1], inter_nodes[-1].distance(node_end) / 2
                )
            )

        return inter_nodes[1:]

    def _interpolate_nodes_fine(
        self,
        node_start: WaypointNode,
        node_end: WaypointNode,
        distance: float,
        epsilon: float = 0.01,
    ) -> List[WaypointNode]:
        inter_nodes = [node_start]
        node_cur = node_start
        while node_cur.distance(node_end) > distance:
            if inter_nodes[-1].distance(node_cur) >= distance:
                inter_nodes.append(node_cur)
            node_cur = self.get_waypoint_next(node_cur, epsilon)
        if inter_nodes[-1].distance(node_end) >= (4.0 / 3.0) * distance:
            inter_nodes.append(
                self.get_waypoint_next(
                    inter_nodes[-1], inter_nodes[-1].distance(node_end) / 2
                )
            )
        return inter_nodes[1:]

    @staticmethod
    def connect_chain(chain: List[WaypointNode]):
        for idx, node in enumerate(chain[:-1]):
            node.add_child(chain[idx + 1].bin)

    def interpolate_nodes(self, distance: float):
        self.interpolated_nodes = []
        for node in self.get_all_nodes():
            for child_bin in node.children:
                child = self.get_node_by_bin(child_bin)
                new_nodes = self._interpolate_nodes_coarse(node, child, distance)

                if len(new_nodes) == 0:
                    continue

                self.connect_chain(new_nodes)
                self.insert_chain(node, child, new_nodes)

                self.interpolated_nodes += new_nodes

    def allow_lane_changes(self):
        for new_node in self.interpolated_nodes:
            left_lane = new_node.waypoint.get_left_lane()
            right_lane = new_node.waypoint.get_right_lane()
            for lane_waypoint in (left_lane, right_lane):
                if lane_waypoint is not None and self.has_waypoint(lane_waypoint):
                    lane_node = self.get_node_by_waypoint(lane_waypoint)
                    if len(lane_node.children) >= 1:
                        self.add_waypoint_child(
                            new_node.waypoint,
                            self.get_node_by_bin(lane_node.children[0]).waypoint,
                        )

    def has_waypoint(self, waypoint: carla.Waypoint):
        return self.waypoint_to_bin(waypoint) in self.get_all_waypoint_bins()

    @property
    def has_orphans(self):
        return any(
            [len(self.get_waypoint_parents(node)) == 0 for node in self.get_all_nodes()]
        )

    @property
    def has_childless(self):
        return not all([node.has_children for node in self.get_all_nodes()])

    def add_waypoint(self, waypoint: carla.Waypoint):
        if self.has_waypoint(waypoint):
            raise ValueError("Cannot add duplicate waypoint to graph")
        waypoint_bin = self.waypoint_to_bin(waypoint)
        self._nodes[waypoint_bin] = WaypointNode(bin=waypoint_bin, waypoint=waypoint)

    def add_waypoint_child(self, parent: carla.Waypoint, child: carla.Waypoint):
        node = self.get_node_by_waypoint(parent)
        if self.waypoint_to_bin(child) in node.children:
            print("Warning: attempted duplicate add of child node")
            return
        node.add_child(self.waypoint_to_bin(child))

    def add_waypoint_children(
        self, parent: carla.Waypoint, children: List[carla.Waypoint]
    ):
        node = self.get_node_by_waypoint(parent)
        for child in children:
            self.add_waypoint_child(node, child)

    def get_node_by_bin(self, waypoint_bin: Tuple[int, int]):
        if waypoint_bin not in self._nodes.keys():
            raise ValueError(
                f'Waypoint bin "{waypoint_bin}" not found within current waypoints.'
            )
        return self._nodes[waypoint_bin]

    def get_node_by_waypoint(self, waypoint: Tuple[int, int]) -> WaypointNode:
        waypoint_bin = self.waypoint_to_bin(waypoint)
        return self.get_node_by_bin(waypoint_bin)

    def get_node_by_index(self, index: int) -> WaypointNode:
        if index >= len(self.get_all_nodes()):
            raise IndexError(f'Index "{index}" out of range for graph.')
        return list(self._nodes.values())[index]

    def get_all_nodes(self) -> List[WaypointNode]:
        return tuple(self._nodes.values())

    def get_all_waypoint_bins(self) -> List[Tuple[int, int]]:
        return [node.bin for node in self.get_all_nodes()]

    def waypoint_to_bin(self, waypoint: carla.Waypoint) -> Tuple[int, int]:
        return self._waypoint_to_bin(waypoint, self.x_bins, self.y_bins)

    @staticmethod
    def _waypoint_to_bin(
        waypoint: carla.Waypoint, x_bins: np.ndarray, y_bins: np.ndarray
    ) -> Tuple[int, int]:
        x_arr = np.array(waypoint.transform.location.x)
        y_arr = np.array(waypoint.transform.location.y)

        x_indices = np.digitize(x_arr, x_bins)
        y_indices = np.digitize(y_arr, y_bins)

        return (x_indices, y_indices)


def get_max_indices(
    topology: List[Tuple[carla.Waypoint, carla.Waypoint]], buffer: int = 200
) -> Tuple[float, float]:
    waypoints = list(itertools.chain.from_iterable(topology))
    return (
        max([w.transform.location.x for w in waypoints]) + buffer,
        max([w.transform.location.y for w in waypoints]) + buffer,
    )


def get_min_indices(
    topology: List[Tuple[carla.Waypoint, carla.Waypoint]], buffer: int = 200
) -> Tuple[float, float]:
    waypoints = list(itertools.chain.from_iterable(topology))
    return (
        min([w.transform.location.x for w in waypoints]) - buffer,
        min([w.transform.location.y for w in waypoints]) - buffer,
    )


def init_map_bins(
    min_pos: Tuple[float, float], max_pos: Tuple[float, float], num_bins: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    x_bins = np.linspace(min_pos[0], max_pos[0], num=num_bins)
    y_bins = np.linspace(min_pos[1], max_pos[1], num=num_bins)

    return (x_bins, y_bins)


def topology_to_graph(
    topology: List[Tuple[carla.Waypoint, carla.Waypoint]],
    x_bins: np.ndarray,
    y_bins: np.ndarray,
):
    graph = WaypointGraph(x_bins, y_bins)
    for w_start, w_end in topology:
        if not graph.has_waypoint(w_start):
            graph.add_waypoint(w_start)
        if not graph.get_node_by_waypoint(w_start).parent_of_bin(
            graph.waypoint_to_bin(w_end)
        ):
            graph.add_waypoint_child(w_start, w_end)

    return graph


def get_sim_obs(
    x_bins: np.ndarray, y_bins: np.ndarray, z_bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Lidar point cloud generation."""
    point_cloud = []
    # Map point cloud data to np array.
    for point in lidar_data:
        point_cloud.append((point.x, point.y, point.z))
    point_cloud = np.array(point_cloud)

    # print('Point cloud size: {}, {}'.format(np.shape(point_cloud), np.max(point_cloud)))
    # Aggregate points into bins.
    lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
    lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0)
    lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0)

    return (np.array(lidar.flatten()), point_cloud.flatten())

if __name__ == "__main__":

    # Connect to Carla client
    client = carla.Client("localhost", 2000)
    client.set_timeout(7)
    carla_server_ver = client.get_server_version()

    # Initialize topology dict

    data_dir = Path("../carla_data")
    target_maps = client.get_available_maps()
    
    for map_id in target_maps:

        map_name = Path(map_id).stem
        map_dir = data_dir / (map_name + "_data")
        client.load_world(map_name)
        world = client.get_world()
        cur_map = world.get_map()
        spawns = cur_map.get_spawn_points()

        # Set simulation to run with a fixed time-step (if non-eval environment).
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.1
        settings.synchronous_mode = True
        world.apply_settings(settings)

        topo = cur_map.get_topology()

        min_ind = get_min_indices(topo)
        max_ind = get_max_indices(topo)

        x_bins, y_bins = init_map_bins(min_ind, max_ind, num_bins=50000)
        graph = topology_to_graph(topo, x_bins, y_bins)
        graph.remove_loopback_children()
        graph.remove_orphans()
        graph.remove_childless()

        graph.remove_twins()

        merge_threshold = 5
        interpolate_dist = 20

        graph.merge_close_neighbors(threshold=merge_threshold)
        graph.interpolate_nodes(distance=interpolate_dist)

        # Do not add lane changes in maps that lack them
        if map_name not in ["Town01", "Town02"]:
            graph.allow_lane_changes()

        # Hack to fix intersectoin at nodes 0, 113, 26 creating visual error
        if map_name == "Town02" and merge_threshold == 5 and interpolate_dist == 20:
            graph.town02_5m_patch()

        px = 1 / plt.rcParams["figure.dpi"]
        fig_len = 10 * 670 * px
        fig, ax = plt.subplots(figsize=(fig_len, fig_len))

        points = np.array(graph.waypoint_locations)

        ax.set_title(map_name)
        ax.scatter(points[:, 0], points[:, 1], color="g", s=5)

        graph_list = list(enumerate(graph.get_all_nodes()))

        for node_idx, node in graph_list:
            for bin_ in node.children:
                if bin_ not in [n.bin for n in graph.get_all_nodes()]:
                    continue

                child = graph.get_node_by_bin(bin_)
                # TODO: use node waypoint dict for children so arrows line up
                ax.arrow(
                    node.waypoint.transform.location.x,
                    node.waypoint.transform.location.y,
                    child.waypoint.transform.location.x
                    - node.waypoint.transform.location.x,
                    child.waypoint.transform.location.y
                    - node.waypoint.transform.location.y,
                    color="grey",
                    head_width=1,
                    length_includes_head=True,
                    alpha=0.2,
                )

        map_dir.mkdir(mode=0o755, exist_ok=True, parents=True)
        fig.savefig(map_dir / "graph.png")
        for node_idx, node in graph_list:
            ax.text(
                node.waypoint.transform.location.x,
                node.waypoint.transform.location.y,
                f"{node_idx}",
                ha="center",
                va="center",
                color="k",
            )
        fig.savefig(map_dir / "graph_labels.png")
        plt.close(fig)

        fig_disp, ax_disp = plt.subplots(figsize=(fig_len, fig_len))
        ax_disp.set_title(map_name, fontsize=200)
        ax_disp.set_yticks([])
        ax_disp.set_xticks([])
        ax_disp.scatter(points[:, 0], points[:, 1], color="g", s=500, zorder=2)

        for node_idx, node in graph_list:
            for bin_ in node.children:
                if bin_ not in [n.bin for n in graph.get_all_nodes()]:
                    continue

                child = graph.get_node_by_bin(bin_)
                # TODO: use node waypoint dict for children so arrows line up
                ax_disp.arrow(
                    node.waypoint.transform.location.x,
                    node.waypoint.transform.location.y,
                    child.waypoint.transform.location.x
                    - node.waypoint.transform.location.x,
                    child.waypoint.transform.location.y
                    - node.waypoint.transform.location.y,
                    color="black",
                    head_width=2,
                    width=0.5,
                    length_includes_head=True,
                    zorder=1,
                )
        fig_disp.savefig(map_dir / "graph_display.png")

        node_locations = graph.get_node_locations()
        np.savetxt(map_dir / "waypoint_locations.csv", node_locations, delimiter=",")

        transition_matrix = graph.generate_matrix()
        np.savetxt(
            map_dir / "transition_matrix.csv",
            transition_matrix,
            fmt="%i",
            delimiter=",",
        )

        seed = 0
        state = 0
        rand_reset = np.random.RandomState(seed)

        # Randomly select vehicle type from list of available vehicles
        vehicles = world.get_blueprint_library().filter("vehicle.*")

        # Spawn agent vehicle at first map spawn point
        player = world.try_spawn_actor(vehicles[0], spawns[0])
        player.set_simulate_physics(False)
        player.set_transform(graph.get_all_nodes()[state].waypoint.transform)

        # Add lidar sensor
        obs_range = 50  # observation range (meter)
        lidar_bin = 0.125  # lidar bin size (meter)
        num_lidar_bins = 256
        lidar_height = 2.1
        obs_size = int(obs_range / lidar_bin)

        x_bins = np.linspace(
            -0.5 * obs_range,
            0.5 * obs_range + lidar_bin,
            num_lidar_bins,
        )
        y_bins = np.linspace(
            -0.5 * obs_range,
            0.5 * obs_range + lidar_bin,
            num_lidar_bins,
        )
        z_bins = np.linspace(-lidar_height - 2, lidar_height + 2, 3)

        lidar_trans = carla.Transform(carla.Location(x=0.0, z=lidar_height))
        lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "32")
        lidar_bp.set_attribute("range", str(obs_range * 100))  # (cm)

        lidar_sensor = world.spawn_actor(lidar_bp, lidar_trans, attach_to=player)

        # FIXME: this is a hack to plcae lidar_data in global scope, wrap into class instead
        def get_lidar_data(data):
            global lidar_data
            lidar_data = data

        lidar_sensor.listen(lambda data: get_lidar_data(data))

        world.tick()

        observations = []
        point_clouds = []

        for node in graph.get_all_nodes():
            player.set_transform(node.waypoint.transform)
            world.tick()
            obs, cloud = get_sim_obs(x_bins, y_bins, z_bins)
            observations.append(obs)
            # Flatten point cloud structure before appending to point_clouds list
            point_clouds.append(cloud)

        np.savetxt(map_dir / "observations.csv", np.array(observations), delimiter=",")
        np.savez(
            map_dir / "point_clouds.npz", point_clouds=np.array(point_clouds, dtype=object)
        )
