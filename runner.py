# runner.py

import os
import time
import json
import logging
import pickle
import numpy as np
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter


class Runner:
    def __init__(self,
                 base_dir,
                 create_agent_fn,
                 env,
                 episode_log_file='',
                 checkpoint_file_prefix='ckpt',
                 max_steps_per_episode=27000):

        if base_dir is None:
            raise ValueError('Missing base_dir.')

        self._base_dir = base_dir
        self._output_dir = os.path.join(base_dir, 'train')
        self._checkpoint_dir = os.path.join(self._output_dir, 'checkpoints')
        self._create_agent_fn = create_agent_fn
        self._env = env
        self._checkpoint_file_prefix = checkpoint_file_prefix
        self._max_steps_per_episode = max_steps_per_episode
        self._episode_log_file = episode_log_file
        self._episode_writer = None
        self._plot_data = []
        self._eval_plot_data = []

        os.makedirs(self._output_dir, exist_ok=True)
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        self._summary_writer = SummaryWriter(log_dir=self._output_dir)
        self._agent = self._create_agent_fn(env=self._env)

    def _checkpoint_path(self, iteration):
        return os.path.join(self._checkpoint_dir, f"{self._checkpoint_file_prefix}_{iteration}.pkl")

    def _load_latest_checkpoint(self):
        files = [f for f in os.listdir(self._checkpoint_dir) if f.endswith('.pkl')]
        if not files:
            return 0, 0

        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest = files[-1]
        path = os.path.join(self._checkpoint_dir, latest)

        with open(path, 'rb') as f:
            data = pickle.load(f)

        if self._agent.unbundle(data):
            logging.info(f"Loaded checkpoint from {latest}")
            return data['iteration'] + 1, data['total_steps'] + 1

        return 0, 0

    def _save_checkpoint(self, iteration, total_steps):
        data = self._agent.bundle()
        if data:
            data['iteration'] = iteration
            data['total_steps'] = total_steps
            path = self._checkpoint_path(iteration)
            with open(path, 'wb') as f:
                pickle.dump(data, f)

    def _run_one_episode(self):
        step_number = 0
        total_reward = 0.0
        observation = self._env.reset()
        action = self._agent.begin_episode(observation)

        while step_number < self._max_steps_per_episode:
            last_observation = observation
            observation, reward, done, info = self._env.step(action)
            total_reward += reward
            step_number += 1

            if done:
                break
            # print(f"Step: {step_number}, Reward: {reward}, Observation: {observation}")
            action = self._agent.step(reward, observation)

        self._agent.end_episode(reward, observation)
        return step_number, total_reward

    def _initialize_metrics(self):
        self._stats = {
            'episode_length': [],
            'episode_time': [],
            'episode_reward': [],
        }
        self._env.reset_metrics()

    def _write_metrics(self, step, suffix):
        num_steps = np.sum(self._stats['episode_length'])
        time_per_step = np.sum(self._stats['episode_time']) / max(1, num_steps)
        avg_len = np.mean(self._stats['episode_length'])
        avg_rew = np.mean(self._stats['episode_reward'])
        std_rew = np.std(self._stats['episode_reward'])

        # --- TensorBoard ---
        self._summary_writer.add_scalar(f'TimePerStep/{suffix}', time_per_step, step)
        self._summary_writer.add_scalar(f'AverageEpisodeLength/{suffix}', avg_len, step)
        self._summary_writer.add_scalar(f'AverageEpisodeRewards/{suffix}', avg_rew, step)
        self._summary_writer.add_scalar(f'StdEpisodeRewards/{suffix}', std_rew, step)
        self._env.write_metrics(lambda tag, value: self._summary_writer.add_scalar(f'{tag}/{suffix}', value, step))

        # --- Console Output ---
        print(f"[{suffix.upper()}][Step {step}] AvgLen: {avg_len:.2f} | AvgRew: {avg_rew:.2f} | "
            f"StdRew: {std_rew:.2f} | Time/Step: {time_per_step:.4f}")

        # --- Log File Output ---
        log_path = os.path.join(self._output_dir, f'{suffix}_metrics.txt')
        with open(log_path, 'a') as f:
            f.write(
                f"[Step {step}] "
                f"AvgLen: {avg_len:.2f}, "
                f"AvgRew: {avg_rew:.2f}, "
                f"StdRew: {std_rew:.2f}, "
                f"Time/Step: {time_per_step:.4f}\n"
            )

    # Plot training_step vs avg_reward(over iteration)
    def run_training(self, max_training_steps=250000, num_iterations=100, checkpoint_frequency=1):
        logging.info('Beginning training...')
        start_iter, total_steps = self._load_latest_checkpoint()

        if num_iterations <= start_iter:
            logging.warning('No training needed. Exiting.')
            return

        # --- 新增：专门记录所有iteration的数据 ---
        all_runs_steps_rewards = []  # 每个iteration保存一个list，里面是(step, reward)

        for iteration in tqdm(range(start_iter, num_iterations), desc="Training Episode"):
            logging.info(f'[TRAIN] Starting iteration {iteration + 1}/{num_iterations}')
            self._initialize_metrics()
            num_steps = 0
            iteration_steps_rewards = []  # 当前 iteration 的 (step, reward) 数据
            start_time = time.time()

            while num_steps < max_training_steps:
                episode_start = time.time()
                ep_len, ep_reward = self._run_one_episode()
                episode_time = time.time() - episode_start

                self._stats['episode_length'].append(ep_len)
                self._stats['episode_time'].append(episode_time)
                self._stats['episode_reward'].append(ep_reward)

                logging.info(
                    f"[TRAIN] Episode {len(self._stats['episode_length'])} | "
                    f"Reward: {ep_reward:.2f} | Length: {ep_len} | Time: {episode_time:.2f}s"
                )

                num_steps += ep_len
                total_steps += ep_len  # 注意total_steps是全局累加的

                # 记录当前total_steps vs reward
                iteration_steps_rewards.append((total_steps, ep_reward))

            all_runs_steps_rewards.append(iteration_steps_rewards)  # 保存这一条run的数据

            # 写 metrics
            self._write_metrics(total_steps, suffix='train')

            if (iteration + 1) % checkpoint_frequency == 0 or (iteration + 1) == num_iterations:
                self._save_checkpoint(iteration, total_steps)
                logging.info(f"[TRAIN] Saved checkpoint at iteration {iteration}")

        # --- 整理多个iteration后的数据，统一对齐steps，求平均 ---
        logging.info("[TRAIN] Start aggregating plot data...")

        # Step1：收集所有出现过的step
        step_set = set()
        for run in all_runs_steps_rewards:
            for step, reward in run:
                step_set.add(step)

        common_steps = sorted(list(step_set))  # 所有出现过的training_steps（去重排序）

        # Step2：对每个step，计算所有iteration上reward的平均
        step_to_rewards = {step: [] for step in common_steps}
        for run in all_runs_steps_rewards:
            for step, reward in run:
                step_to_rewards[step].append(reward)

        # Step3：把平均reward保存到_plot_data
        self._plot_data = []
        for step in common_steps:
            rewards = step_to_rewards[step]
            avg_reward = np.mean(rewards)
            self._plot_data.append({
                "training_step": step,
                "avg_reward": avg_reward
            })

        # 保存 plot_data
        plot_path = os.path.join(self._output_dir, 'plot_data.json')
        print("Training plot path: ", plot_path)
        with open(plot_path, 'w') as f:
            json.dump(self._plot_data, f)
        logging.info(f"[TRAIN] Saved plot data to {plot_path}")


    def run_evaluation(self, max_eval_episodes=100, checkpoint_dir=None):
        logging.info('Beginning evaluation...')
        checkpoint_dir = checkpoint_dir or self._checkpoint_dir

        files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        # --- Initialize dict to save step vs avg_eval_reward ---
        self._eval_plot_data = []  # ✨每次评估一条点：(checkpoint训练步数, 平均reward)

        # Case 1: No checkpoints, fall back to evaluating current agent
        if not files:
            logging.info("[EVAL] No checkpoint found. Evaluating current agent directly.")
            self._initialize_metrics()

            for ep_idx in tqdm(range(max_eval_episodes), desc="Evaluation Episode"):
                ep_len, ep_reward = self._run_one_episode()
                self._stats['episode_length'].append(ep_len)
                self._stats['episode_reward'].append(ep_reward)
                print(f"[EVAL] CurrentAgent | Episode {ep_idx+1} | Reward: {ep_reward:.2f} | Length: {ep_len}")

            self._write_metrics(0, suffix='eval')
            avg_rew = np.mean(self._stats['episode_reward'])
            self._eval_plot_data.append({"training_step": 0, "avg_reward": avg_rew})

        else:
            # Case 2: Evaluate from each checkpoint
            for file in files:
                with open(os.path.join(checkpoint_dir, file), 'rb') as f:
                    data = pickle.load(f)

                if not self._agent.unbundle(data):
                    logging.warning(f"Could not load checkpoint {file}")
                    continue

                checkpoint_steps = data.get('total_steps', 0)  # ✨从checkpoint中取出训练步数
                logging.info(f"[EVAL] Evaluating checkpoint {file} at training step {checkpoint_steps}")
                self._initialize_metrics()

                for ep_idx in tqdm(range(max_eval_episodes), desc=f"Evaluation @ Step {checkpoint_steps}"):
                    ep_len, ep_reward = self._run_one_episode()
                    self._stats['episode_length'].append(ep_len)
                    self._stats['episode_reward'].append(ep_reward)
                    print(f"[EVAL] {file} | Episode {ep_idx+1} | Reward: {ep_reward:.2f} | Length: {ep_len}")

                # 写 tensorboard
                self._write_metrics(checkpoint_steps, suffix='eval')

                # --- ✨保存这次 evaluation 的 (training_step, avg_reward) ✨---
                avg_rew = np.mean(self._stats['episode_reward'])
                self._eval_plot_data.append({
                    "training_step": checkpoint_steps,
                    "avg_reward": avg_rew
                })

        # --- Save eval plot data ---
        plot_path = os.path.join(self._output_dir, 'eval_plot_data.json')
        print("Eval plot path: ", plot_path)
        with open(plot_path, 'w') as f:
            json.dump(self._eval_plot_data, f)
        logging.info(f"[EVAL] Saved eval plot data to {plot_path}")

    
