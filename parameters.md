# Experiment Config File parameters

- **`base`**: Trial parameters.
  - **`seed`**: RNG Seeds.
  - **`eval_freq`**: Number of training timesteps between two evaluations (per trial).
  - **`num_steps`**: Number of total training steps (per trial).
- **`agent`**: Agent parameters.
  - **`name`**: Method (A2C, PPO, DQN, or QRDQN).
  - **`discount`**: Discount factor (gamma).
  - **`num_quantiles`**: Num quantiles. 
  - **`alpha`**: Learning step size (alpha). 
- **`environment`**: Environment Parameters.
  - **`data_dir`**: Directory where environment data is stored.
  - **`map_name`**: Name of the target Map.
  - **`start_state`**: Start state.
  - **`goal_states`**: List of goal states.
  - **`crosswalk_states`**: List of stochastic crosswalk states. 
- **`policy`**: Agent network structure (currently only "CnnPolicy" is available).
- **`save_dir`**: Directory where experiment data is stored.
