from pathlib import Path
from typing import Sequence, Tuple, Union
import numpy as np

import gym

class CarleEnv(gym.Env):

    def __init__(
        self,
        is_eval_env: bool,
        seed: int,
        dataset_dir: Union[str, Path],
        goal_states: Sequence[int],
        reset_state: int,
        discount: float,
        crosswalk_states: Sequence[int],
        agent: str
    ):
        # PRNG for random rewards.
        self._rand = np.random.RandomState(seed)

        # Count timestep and record stochastic reward
        self.count = 0

        # Record path if the environment is for evaluation 
        self.record = True if is_eval_env else False
        self.curr_path = []
        self.all_paths = []

        # Record quantiles of all state action pair (for QR-DQN agent)
        self.agent = agent
        self.quantiles = []

        # Set state information for goals and resets
        self.goal_states = goal_states
        self.reset_state = reset_state
        self.state = self.reset_state
        self.prev_state = None # used for checking self-transition
        self.crosswalk_states = crosswalk_states

        # Initialize environment parameters and data
        dataset_path = Path(dataset_dir)
        self.waypoint_locations = np.loadtxt(
            fname=dataset_path / "waypoint_locations.csv", delimiter=","
        )
        self.transition_matrix = np.loadtxt(
            fname=dataset_path / "transition_matrix.csv", dtype=int, delimiter=","
        )
        self.observations = np.loadtxt(
            fname=dataset_path / "observations.csv", delimiter=","
        )

        # Define action space and observation space
        self.action_space = gym.spaces.Discrete(np.shape(self.transition_matrix)[1])
        self.observation_shape = (1,255,255)
        self.observation_space = gym.spaces.Box(np.zeros(self.observation_shape),np.ones(self.observation_shape),dtype=np.float32)

        # Initialize transition counter
        self.transition_counts = np.zeros_like(self.transition_matrix)

        # Set discount factor
        self.discount = discount

        # Set rewards values
        self.rewards = -3 * np.ones(len(self.transition_matrix))
        self.rewards[np.array(goal_states)] = 0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Apply the given action and transition to the next state."""
        self.prev_state = self.state
        self.transition_counts[self.state, action] += 1
        self.state = self.transition_matrix[self.state, action]
        self.count += 1

        if self.record:
            self.curr_path.append(self.state)

        return self.get_obs(), self.get_reward(), self.get_done(), self.get_state()

    def reset(self) -> np.ndarray:
        """Reset the environment and save path if the environment is for evaluation"""
        self.transition_counts = np.zeros_like(self.transition_matrix)
        self.state = self.reset_state
        self.prev_state = None
        self.count = 0

        if self.record:
            if np.shape(self.curr_path)[0] != 0:
                self.all_paths.append(self.curr_path)
                self.curr_path = []

        return self.get_obs()

    def get_obs_at_state(self, state:int) -> np.ndarray:
        """Returns the observation image (ground) for a given state."""
        scans = np.reshape(self.observations[state],(255,255,2))
        ground = scans[:,:,1]
        #return np.array(ground.flatten())
        return np.array([ground])

    def get_obs(self) -> np.ndarray:
        """Returns the observation image (ground) for the current state."""
        #scans = np.reshape(self.observations[self.state],(255,255,2))
        #ground = scans[:,:,1]
        #return np.array(ground.flatten())
        return self.get_obs_at_state(self.state)

    def save_quantiles(self, quantiles:np.ndarray) -> np.ndarray:
        """Save quantiles of all state action pair (for QR-DQN agent)"""
        assert self.agent == "QRDQN", "save_quantiles is only avaible to the QR-DQN agent"
        self.quantiles.append(quantiles)

    def get_quantiles(self) -> np.ndarray:
        """Get quantiles of all state action pair (for QR-DQN agent)"""
        assert self.agent == "QRDQN", "get_quantiles is only avaible to the QR-DQN agent"
        return np.array(self.quantiles)

    def get_reward(self, ssd_thres:int=15) -> float:
        """Returns the reward for reaching the current state."""
        # Penalize the self-transition action
        if self.prev_state == self.state:
            return self.rewards[self.state] - ssd_thres - 3

        # Add noise at the simulated cross walks.
        if self.state in self.crosswalk_states:
            # Use vonmises distribution as stand-in for wrapped gaussian
            #  - Interval is bounded from -2 to 0 with below parameters
            #  - kappa parameter is inversely proportional to variance
            #  - see:
            #     https://numpy.org/devdocs/reference/random/generated/numpy.random.vonmises.html
            return 3*(self._rand.vonmises(mu=0, kappa=1) / np.pi) - 3

        # Deterministic traffic penalty otherwise.
        return self.rewards[self.state]

    def get_done(self) -> bool:
        """Returns a done flag if the goal is reached."""
        return self.state in self.goal_states

    def get_state(self) -> dict:
        """Return current state id"""
        info = {"state_id":self.state}
        return info

    def get_count(self) -> list:
        """Return count since last reset"""
        return self.count

    def get_all_paths(self) -> list:
        """Return all paths in evaluation"""
        self.reset()
        return self.all_paths

    def get_num_of_states(self) -> int:
        return np.shape(self.transition_matrix)[0]
