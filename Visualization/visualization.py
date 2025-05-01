#%%
# %cd /projectnb/ds543/ysi/recsim_2025

#%%
import json
import numpy as np
import matplotlib.pyplot as plt

#%%
def smooth_data(x, y, window=200, stride=20):
    """
    滑动窗口平滑 + 降采样
    """
    smooth_x = []
    smooth_y = []
    for i in range(0, len(y) - window + 1, stride):
        x_slice = x[i:i + window]
        y_slice = y[i:i + window]
        smooth_x.append(np.mean(x_slice))
        smooth_y.append(np.mean(y_slice))
    return smooth_x, smooth_y

def load_and_smooth_plot(path, label, color, window=200, stride=20):
    with open(path, 'r') as f:
        data = json.load(f)

    # 按 training_step 排序
    data.sort(key=lambda x: x['training_step'])
    steps = [d['training_step'] for d in data]
    rewards = [d['avg_reward'] for d in data]

    if len(rewards) < window:
        print(f"[Warning] {label} 数据不足以平滑 (长度={len(rewards)})，将跳过")
        return None

    smooth_steps, smooth_rewards = smooth_data(steps, rewards, window, stride)
    return smooth_steps, smooth_rewards, label, color

# === 配置路径、标签、颜色 ===
models = [
    {
        "path": "./logs/best_policy/train/plot_data.json",
        "label": "Best",
        "color": "purple"
    },
    {
        "path": "./logs/best_policy_quality/train/plot_data.json",
        "label": "Best Quality",
        "color": "blue"
    }
    # {
    #     "path": "./logs/random/train/plot_data.json",
    #     "label": "Random",
    #     "color": "gray"
    # },
    # {
    #     "path": "./logs/UCB_bandit_c_0.1/train/plot_data.json",
    #     "label": "Bandit 0.1",
    #     "color": "blue"
    # },
    # {
    #     "path": "./logs/UCB_bandit_c_0.5/train/plot_data.json",
    #     "label": "Bandit 0.5",
    #     "color": "red"
    # },
    # {
    #     "path": "./logs/UCB_bandit_c_1.0/train/plot_data.json",
    #     "label": "Bandit 1.0",
    #     "color": "purple"
    # },
    # {
    #     "path": "./logs/UCB_bandit_c_2.0/train/plot_data.json",
    #     "label": "Bandit 2.0",
    #     "color": "green"
    # },
    # {
    #     "path": "./logs/UCB_bandit/train/plot_data.json",
    #     "label": "Bandit",
    #     "color": "blue"
    # },
    # {
    #     "path": "./logs/Contextual_bandit_DocOnly/train/plot_data.json",
    #     "label": "Contextual Bandit (DocOnly)",
    #     "color": "green"
    # },
    # {
    #     "path": "./logs/Contextual_bandit_UserDoc/train/plot_data.json",
    #     "label": "Contextual Bandit (User+Doc)",
    #     "color": "orange"
    # }
    # {
    #     "path": "./logs/naive_dqn/20000_200_1/train/plot_data.json",
    #     "label": "Naive DQN",
    #     "color": "red"
    # },
    # {
    #     "path": "./logs/double_dqn/20000_200_1/train/plot_data.json",
    #     "label": "Double DQN",
    #     "color": "green"
    # }
]

# === 开始画图 ===
plt.figure(figsize=(10, 6))

for model in models:
    steps, rewards, label, color = load_and_smooth_plot(model["path"], model["label"], model["color"])
    plt.plot(steps, rewards, label=label, color=color)

plt.xlabel("Training Steps")
plt.ylabel("Average Reward")
plt.title("Training Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Visualization/bandit_vs_random.png")
plt.show()



# algo_list = ["random", "UCB_bandit"] #,"Contextual_bandit_DocOnly","Contextual_bandit_UserDoc"]
# algo_list_1 =["naive_dqn","double_dqn"]  #"naive_dqn_target_network",

# plt.figure(figsize = (10,6))

# random_file_path = "./logs/random/train/plot_data.json"
# with open(random_file_path, 'r') as f:
#     random_data = json.load(f)

# random_avg_rewards = [entry["avg_rew"] for entry in random_data]
# mean_random_reward = sum(random_avg_rewards) / len(random_avg_rewards)
# max_episode = max(entry["episode"] for entry in random_data)
# # smoothed_random_rewards = moving_average(random_avg_rewards, window_size=5)
# # smoothed_random_episodes = random_episodes[:len(smoothed_random_rewards)]

# # plt.plot(smoothed_random_episodes, smoothed_random_rewards, label="random",
# #          linestyle='--', linewidth=2.5, color='black')
# plt.hlines(y=mean_random_reward, xmin=0, xmax=max_episode,
#            colors='black', linestyles='--', linewidth=2.5, label='random (mean)')
# #%%
# algo_list_1 =["random", "UCB_bandit"]
# for algo in algo_list_1:
#     file_path = f"./logs/{algo}/train/plot_data.json"
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#     episodes = [entry["episode"] for entry in data]
#     avg_rewards = [entry["avg_rew"] for entry in data]
    
#     smoothed_rewards = moving_average(avg_rewards, window_size=100)
#     smoothed_episodes = episodes[:len(smoothed_rewards)]
#     plt.plot(smoothed_episodes, smoothed_rewards, label=algo)

# plt.xlabel("Episode")
# plt.ylabel("Average Reward")
# plt.title("Episode vs Smoothed Average Reward for Different Algorithms")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig("./Visualization/results/Final_plot_2.png")
# plt.show()
# # %%

# # %%
# # Avg Reward

# algo_list_2 = ["random","UCB_bandit","Contextual_bandit","Contextual_bandit_DocOnly","Contextual_bandit_UserDoc","naive_dqn","naive_dqn_target_network","double_dqn"] 
# avg_reward_dict = {}
# stable_avg_reward_dict = {}
# for algo in algo_list_2:
#     file_path = f"./logs/{algo}/train/plot_data.json"
#     if os.path.exists(file_path):
#         with open(file_path, "r") as f:
#             data = json.load(f)
#             avg_rewards = [entry["avg_rew"] for entry in data]
#             mean_reward = sum(avg_rewards) / len(avg_rewards) if avg_rewards else 0
#             stable_rewards = avg_rewards[50:] 
#             stable_mean_reward = sum(stable_rewards) / len(stable_rewards)
#             avg_reward_dict[algo] = mean_reward
#             stable_avg_reward_dict[algo] = stable_mean_reward
# # %%

# %%
