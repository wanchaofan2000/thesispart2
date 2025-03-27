target_leader = (-0, -0, 0, 0)
num_drones = 32
time_step = 0.3
num_of_episodes = 100
timesteps_per_episode = 20000
learning_rate_actor = 1E-6
learning_rate_critic = 1E-4
# state_dim = 8
state_dim = 4
action_dim = 2

critic_dim = state_dim + action_dim

communication_range = 30