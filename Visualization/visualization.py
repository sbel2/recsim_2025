#%%
import os
import json
import numpy as np
import matplotlib.pyplot as plt

#%%
algos = [
    "random", "bandit", "Contextual_bandit_DocOnly",
    "Contextual_bandit_UserDoc", "naive_dqn", "double_dqn"
]
file_types = ['episode_watch_times.json', 'step_watch_times.json', 'eval_watch_times.json']
base_log_path = "./logs"
save_dir = "./Visualization/results"
os.makedirs(save_dir, exist_ok=True)

def smooth(y, window_size=10):
    if len(y) < window_size:
        return y
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')

def plot_all_algos_for_filetype(file_type, smooth_window=10):
    plt.figure(figsize=(10, 6))
    for algo in algos:
        path = os.path.join(base_log_path, algo, "20000_200", "train", file_type)
        if not os.path.exists(path):
            print(f"文件缺失: {path}")
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        if file_type == 'episode_watch_times.json':
            x = data['episode']
            y = data['total_watch_time']
        elif file_type == 'step_watch_times.json':
            x = data['training_step']
            y = data['avg_total_watch_time']
        elif file_type == 'eval_watch_time.json':
            x = data['episode']
            y = data['total_watch_time']
        else:
            continue

        if len(x) != len(y):
            print(f"数据长度不一致: {path}")
            continue

        y_smooth = smooth(y, smooth_window)
        x_smooth = x[:len(y_smooth)]
        plt.plot(x_smooth, y_smooth, label=algo)

    plt.xlabel("Episode" if 'episode' in file_type else "Training Step")
    plt.ylabel("Total Watch Time (Smoothed)")
    plt.title(f"{file_type.split('.')[0]} - Total Watch Time (Smoothed)")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, f"{file_type.replace('.json', '')}_watch_time_smoothed.png")
    plt.savefig(save_path)
    plt.close()
    print(f"保存图像至：{save_path}")


def plot_smoothed_recovered(file_type, smooth_window=10):
    plt.figure(figsize=(10, 6))
    for algo in algos:
        path = os.path.join(base_log_path, algo, "20000_200", "train", file_type)
        if not os.path.exists(path):
            print(f"缺失文件: {path}")
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        # 选择 x, y 数据
        if file_type == 'episode_watch_times.json':
            x = data['episode']
            y = data['recovered_time']
        elif file_type == 'step_watch_times.json':
            x = data['training_step']
            y = data['avg_recovered_time']
        elif file_type == 'eval_watch_time.json':
            x = data['episode']
            y = data['recovered_time']
        else:
            continue

        if len(x) != len(y):
            print(f"数据长度不一致: {path}")
            continue

        y_smooth = smooth(y, smooth_window)
        x_smooth = x[:len(y_smooth)]

        plt.plot(x_smooth, y_smooth, label=algo)

    plt.xlabel("Episode" if 'episode' in file_type else "Training Step")
    plt.ylabel("Recovered Time (Smoothed)")
    plt.title(f"{file_type.split('.')[0]} - Recovered Time")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, f"{file_type.replace('.json', '')}_recovered.png")
    plt.savefig(save_path)
    plt.close()
    print(f"保存图像至：{save_path}")

def plot_smoothed_combined(file_type, smooth_window=10):
    plt.figure(figsize=(12, 6))
    for algo in algos:
        path = os.path.join(base_log_path, algo, "20000_200", "train", file_type)
        if not os.path.exists(path):
            print(f"缺失文件: {path}")
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        # 获取 x 轴和 y 轴数据
        if file_type == 'episode_watch_times.json':
            x = data['episode']
            total = data['total_watch_time']
            recovered = data['recovered_time']
        elif file_type == 'step_watch_times.json':
            x = data['training_step']
            total = data['avg_total_watch_time']
            recovered = data['avg_recovered_time']
        elif file_type == 'eval_watch_time.json':
            x = data['episode']
            total = data['total_watch_time']
            recovered = data['recovered_time']
        else:
            continue

        if not (len(x) == len(total) == len(recovered)):
            print(f"数据长度不一致: {path}")
            continue

        # 平滑处理
        total_smooth = smooth(total, smooth_window)
        recovered_smooth = smooth(recovered, smooth_window)
        x_smooth = x[:len(total_smooth)]

        plt.plot(x_smooth, total_smooth, label=f"{algo} - total")
        plt.plot(x_smooth, recovered_smooth, linestyle='--', label=f"{algo} - recovered")

    plt.xlabel("Episode" if 'episode' in file_type else "Training Step")
    plt.ylabel("Watch Time (Smoothed)")
    plt.title(f"{file_type.split('.')[0]} - Total & Recovered Watch Time")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, f"{file_type.replace('.json', '')}_combined.png")
    plt.savefig(save_path)
    plt.close()
    print(f"保存图像至：{save_path}")

#%%
for file_type in file_types:
    if file_type == 'eval_watch_time.json':
        plot_all_algos_for_filetype(file_type, 2)
        plot_smoothed_recovered(file_type,2)
        plot_smoothed_combined(file_type,2)
    else:
        plot_all_algos_for_filetype(file_type, 1000)
        plot_smoothed_recovered(file_type,1000)
        plot_smoothed_combined(file_type,1000)
        
# %%
## Avg reward for the last 1000
results = []

for algo in algos:
    path = os.path.join(base_log_path, algo, "20000_200", "train", "step_watch_times.json")
    if not os.path.exists(path):
        print(f"❌ 缺失文件: {path}")
        continue

    with open(path, 'r') as f:
        data = json.load(f)

    steps = data["training_step"]
    total_watch = data["avg_total_watch_time"]
    recovered = data["avg_recovered_time"]

    if len(steps) < 10000:
        print(f"⚠️ {algo} 数据不足10000条，仅有 {len(steps)} 条，将使用全部数据")
        total_watch_last = total_watch
        recovered_last = recovered
    else:
        total_watch_last = total_watch[-10000:]
        recovered_last = recovered[-10000:]

    mean_total = np.mean(total_watch_last)
    mean_recovered = np.mean(recovered_last)
    results.append((algo, mean_total, mean_recovered))

# 打印结果
print(f"{'算法':<30} {'Avg Total Watch Time':>25} {'Avg Recovered Time':>25}")
for algo, mean_total, mean_recovered in results:
    print(f"{algo:<30} {mean_total:>25.2f} {mean_recovered:>25.2f}")