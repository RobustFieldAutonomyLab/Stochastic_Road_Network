import gym
import sys
sys.path.insert(0,"./thirdparty")
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from sb3_contrib.qrdqn import QRDQN
import numpy as np
import os
import argparse
import json
import itertools
from multiprocessing import Pool
import csv

parser = argparse.ArgumentParser(description="Run baseline experiments")
parser.add_argument(
    "-C",
    "--config-file",
    dest="config_file",
    type=open,
    required=True,
    help="configuration file for experiment parameters",
)
parser.add_argument(
    "-P",
    "--num-procs",
    dest="num_procs",
    type=int,
    default=1,
    help="number of subprocess workers to use for trial parallelization",
)
parser.add_argument(
    "-D",
    "--device",
    dest="device",
    type=str,
    default="auto",
    help="device to run all subprocesses, could only specify 1 device in each run"
)


def product(*args, repeat=1):
    # This function is a modified version of 
    # https://docs.python.org/3/library/itertools.html#itertools.product
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def trial_params(params):
    if isinstance(params,(str,int,float)):
        return [params]
    elif isinstance(params,list):
        return params
    elif isinstance(params, dict):
        keys, vals = zip(*params.items())
        mix_vals = []
        for val in vals:
            val = trial_params(val)
            mix_vals.append(val)
        return [dict(zip(keys, mix_val)) for mix_val in itertools.product(*mix_vals)]
    else:
        raise TypeError("Parameter type is incorrect.")

def params_dashboard(params):
    print("\n====== Experiment Setup ======\n")
    print("seed: ",params["base"]["seed"])
    print("num_timesteps: ",params["base"]["num_timesteps"])
    print("agent: ",params["agent"]["name"])
    print("network: ",params["policy"])
    print("discount: ",params["agent"]["discount"])
    print("learning rate: ",params["agent"]["alpha"])
    print("map: ",params["environment"]["map_name"])
    print("start_state: ",params["environment"]["start_state"])
    print("goal_states: ",params["environment"]["goal_states"])
    print("crosswalk_states: ",params["environment"]["crosswalk_states"])
    if params["agent"]["name"] == "QRDQN":
        print("eval policy: ",params["agent"]["eval_policy"])
        if params["agent"]["eval_policy"] == "Thresholded_SSD":
            print("ssd thres: ",params["agent"]["ssd_thres"])
    print("\n")

def run_trial(params,device):

    lr = params["agent"]["alpha"]
    sd = params["base"]["seed"]
    cw = params["environment"]["crosswalk_states"]
    stoc = "no_stoc" if (np.shape(cw)[0]==0) else "stoc" 

    print("creating behaviour env")
    behave_env = gym.make("carle_gym:carle-v0",
                    seed = sd,
                    is_eval_env=False,
                    dataset_dir=params["environment"]["data_dir"]+"/"+params["environment"]["map_name"]+"_data",
                    goal_states=tuple(params["environment"]["goal_states"]),
                    reset_state=params["environment"]["start_state"],
                    discount=params["agent"]["discount"],
                    crosswalk_states=cw,
                    agent=params["agent"]["name"],
                    network=params["policy"],
                    r_base=params["environment"]["r_base"],
                    r_loopback=params["environment"]["r_loopback"])

    print("creating evaluation env")
    evaluate_env = gym.make("carle_gym:carle-v0",
                    seed = sd,
                    is_eval_env=True,
                    dataset_dir=params["environment"]["data_dir"]+"/"+params["environment"]["map_name"]+"_data",
                    goal_states=tuple(params["environment"]["goal_states"]),
                    reset_state=params["environment"]["start_state"],
                    discount=params["agent"]["discount"],
                    crosswalk_states=cw,
                    agent=params["agent"]["name"],
                    network=params["policy"],
                    r_base=params["environment"]["r_base"],
                    r_loopback=params["environment"]["r_loopback"])

    behave_env.reset()
    evaluate_env.reset()
    
    if params["agent"]["name"] == "QRDQN":
        save_dir = os.path.join(params["save_dir"],params["agent"]["name"],params["environment"]["map_name"],params["policy"],params["agent"]["eval_policy"],stoc,"buffer_"+str(params["agent"]["buffer_size"]),"n_quantile_"+str(params["agent"]["n_quantiles"]),"lr_"+str(lr),"seed_"+str(sd))
    else:
        save_dir = os.path.join(params["save_dir"],params["agent"]["name"],params["environment"]["map_name"],params["policy"],stoc,"buffer_"+str(params["agent"]["buffer_size"]),"lr_"+str(lr),"seed_"+str(sd)) 

    os.makedirs(save_dir)
    param_file = os.path.join(save_dir,"trial_config.json")
    with open(param_file, 'w+') as outfile:
        json.dump(params, outfile)
    
    if params["policy"] == "MlpPolicy":
        policy_args = {}
    elif params["policy"] == "CnnPolicy":
        policy_args = {"normalize_images":False}
    else:
        raise RuntimeError("The network strucutre is not available")
    
    eval_args = {}
    if params["agent"]["name"] == "PPO":
        model = PPO(params["policy"], 
                    behave_env, 
                    verbose=1,
                    policy_kwargs=policy_args,
                    learning_rate=lr, 
                    seed=sd, 
                    n_steps=params["agent"]["buffer_size"], 
                    batch_size=params["agent"]["batch_size"], 
                    n_epochs=params["agent"]["n_epochs"],
                    gamma=params["agent"]["discount"],
                    device=device)
    elif params["agent"]["name"] == "A2C":
        model = A2C(params["policy"], 
                    behave_env, 
                    verbose=1, 
                    policy_kwargs=policy_args,
                    learning_rate=lr, 
                    seed=sd, 
                    n_steps=params["agent"]["buffer_size"], 
                    gamma=params["agent"]["discount"],
                    device=device)
    elif params["agent"]["name"] == "DQN":
        model = DQN(params["policy"], 
                    behave_env, 
                    verbose=1,
                    policy_kwargs=policy_args,
                    learning_rate=lr, 
                    exploration_fraction=params["agent"]["eps_fraction"],
                    exploration_final_eps=params["agent"]["epsilon"],
                    seed=sd, 
                    buffer_size=params["agent"]["buffer_size"], 
                    learning_starts=params["agent"]["buffer_size"],
                    batch_size=params["agent"]["batch_size"], 
                    gamma=params["agent"]["discount"],
                    device=device)
    elif params["agent"]["name"] == "QRDQN":
        policy_args["n_quantiles"] = params["agent"]["n_quantiles"]
        eval_args["eval_policy"] = params["agent"]["eval_policy"]
        if params["agent"]["eval_policy"] == "Thresholded_SSD":
            eval_args["ssd_thres"] = params["agent"]["ssd_thres"]
        model = QRDQN(params["policy"],
                      behave_env, 
                      verbose=1,
                      policy_kwargs=policy_args,
                      learning_rate=lr,
                      exploration_fraction=params["agent"]["eps_fraction"],
                      exploration_final_eps=params["agent"]["epsilon"],
                      seed=sd, 
                      buffer_size=params["agent"]["buffer_size"], 
                      learning_starts=params["agent"]["buffer_size"],
                      batch_size=params["agent"]["batch_size"], 
                      gamma=params["agent"]["discount"],
                      device=device)
    else:
        raise RuntimeError("The agent is not available.")

    model.learn(total_timesteps=params["base"]["num_timesteps"], eval_env=evaluate_env, eval_freq=params["base"]["eval_freq"], n_eval_episodes=1, eval_log_path=save_dir, **eval_args)

    # save all paths in evaluations
    all_eval_paths = evaluate_env.get_all_paths()
    paths_file = os.path.join(save_dir,"eval_paths.csv")
    with open(paths_file, "w", newline="") as f:
        write = csv.writer(f)
        write.writerows(all_eval_paths)

    # save all quantiles in evalutions (for QR-DQN agent)
    if params["agent"]["name"] == "QRDQN":
        all_eval_q = evaluate_env.get_quantiles()
        np.save(os.path.join(save_dir,"eval_quantiles.npy"),all_eval_q)

    behave_env.close()
    evaluate_env.close()

if __name__ == "__main__":
    args = parser.parse_args()
    params = json.load(args.config_file)
    
    params_dashboard(params)
    goal_states = params["environment"].pop("goal_states",None)
    assert goal_states is not None, "goal states not exist"
    crosswalk_states = params["environment"].pop("crosswalk_states",None)
    assert crosswalk_states is not None, "crosswalk states should be [] if not exist"

    trial_param_list = trial_params(params)

    if args.num_procs == 1:
        for param in trial_param_list:
            param["environment"]["goal_states"] = goal_states
            param["environment"]["crosswalk_states"] = crosswalk_states
            run_trial(param,args.device)
    else:
        with Pool(processes=args.num_procs) as pool:
            for param in trial_param_list:
                param["environment"]["goal_states"] = goal_states
                param["environment"]["crosswalk_states"] = crosswalk_states
                pool.apply_async(run_trial,(param,args.device))
            
            pool.close()
            pool.join()

