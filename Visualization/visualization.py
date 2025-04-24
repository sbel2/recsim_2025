#%%
%cd /projectnb/vkolagrp/yiliu/hrandomw/project_2/recsim_2025

#%%
import json
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

#%%
algo_list = ["bandit","Contextual_bandit_DocOnly","Contextual_bandit_UserDoc"]
algo_list_1 =["naive_dqn","double_dqn"]  #"naive_dqn_target_network",

plt.figure(figsize = (10,6))

random_file_path = "./logs/random/train/plot_data.json"
with open(random_file_path, 'r') as f:
    random_data = json.load(f)

random_avg_rewards = [entry["avg_rew"] for entry in random_data]
mean_random_reward = sum(random_avg_rewards) / len(random_avg_rewards)
max_episode = max(entry["episode"] for entry in random_data)
# smoothed_random_rewards = moving_average(random_avg_rewards, window_size=5)
# smoothed_random_episodes = random_episodes[:len(smoothed_random_rewards)]

# plt.plot(smoothed_random_episodes, smoothed_random_rewards, label="random",
#          linestyle='--', linewidth=2.5, color='black')
plt.hlines(y=mean_random_reward, xmin=0, xmax=max_episode,
           colors='black', linestyles='--', linewidth=2.5, label='random (mean)')

for algo in algo_list_1:
    file_path = f"./logs/{algo}/train/plot_data.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
    episodes = [entry["episode"] for entry in data]
    avg_rewards = [entry["avg_rew"] for entry in data]
    
    smoothed_rewards = moving_average(avg_rewards, window_size=5)
    smoothed_episodes = episodes[:len(smoothed_rewards)]
    plt.plot(smoothed_episodes, smoothed_rewards, label=algo)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Episode vs Smoothed Average Reward for Different Algorithms")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./Visualization/results/Final_plot_2.png")
plt.show()
# %%

# %%
# Avg Reward

algo_list_2 = ["random","UCB_bandit","Contextual_bandit","Contextual_bandit_DocOnly","Contextual_bandit_UserDoc","naive_dqn","naive_dqn_target_network","double_dqn"] 
avg_reward_dict = {}
stable_avg_reward_dict = {}
for algo in algo_list_2:
    file_path = f"./logs/{algo}/train/plot_data.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
            avg_rewards = [entry["avg_rew"] for entry in data]
            mean_reward = sum(avg_rewards) / len(avg_rewards) if avg_rewards else 0
            stable_rewards = avg_rewards[50:] 
            stable_mean_reward = sum(stable_rewards) / len(stable_rewards)
            avg_reward_dict[algo] = mean_reward
            stable_avg_reward_dict[algo] = stable_mean_reward
# %%
