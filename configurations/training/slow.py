#%% Training information
batch_size = 100
max_steps = 400000

#%%
moving_average_decay = 0.9999
num_epochs_per_decay = 1000.0
learning_rate_decay_factor = 0.1
initial_learning_rate = 1e-2

#%%
keep_prob = [0.50, 0.20]
eval_interval_secs = int(60 * 1.5)
