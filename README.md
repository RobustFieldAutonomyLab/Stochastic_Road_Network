## System Requirements
-  Ubuntu 20.04

## Installation
1. Install [NVIDIA Container Runtime](https://nvidia.github.io/nvidia-container-runtime/)
```
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt install nvidia-container-runtime

sudo systemctl daemon-reload
sudo systemctl restart docker
```

2. Clone this git repo and enter the directory.
```
git clone git@github.com:RobustFieldAutonomyLab/Stochastic_Road_Network.git
cd Stochastic_Road_Network
```

3. Install relevant system dependencies for CARLA Python Library:
```
sudo apt install libpng16-16 libjpeg8 libtiff5
```

## Usage
1. Run the Docker script (initializes headless CARLA server under Docker)
```
sudo ./carla_docker.sh -oe
```

2. Run the data generation script:
```
$ python scripts/extract_maps.py
```

3. Run the main experiment script:
```
$ python run_stable_baselines3.py -C [experiment config file (required)] -P [number of processes (optional)] -D [cuda device (optional)]
```

## Experiment Parameterization
Example configuration files are provided in the **config** directory, and see [parameters.md](parameters.md) for detailed explanations of common parameters.

## Third Party Libraries
This project uses implementations of A2C, PPO, DQN and QRDQN agents from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and [stable-baselines3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib), and makes some modifications to apply to the proposed environment. There are some agent specific parameters in the provided configuration files, please refer to [on_policy_algorithm.py](https://github.com/RobustFieldAutonomyLab/Stochastic_Road_Network/blob/main/thirdparty/stable_baselines3/common/on_policy_algorithm.py) ((A2C and PPO)) and [off_policy_algorithm.py](https://github.com/RobustFieldAutonomyLab/Stochastic_Road_Network/blob/main/thirdparty/stable_baselines3/common/off_policy_algorithm.py) (DQN and QRDQN) for further information.
