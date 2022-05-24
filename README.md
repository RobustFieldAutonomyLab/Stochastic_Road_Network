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

3. Enter the repo root folder and install the packages:
```
$ pip install -r requirements.txt
$ pip install -e .
```

4. Install relevant system dependencies for CARLA Python Library:
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
$ python run_stable_baselines3.py
```

## Experiment Parameterization
Example configuration files are provided in the **config** directory, and see [parameters.md](parameters.md) for detailed explanations.
