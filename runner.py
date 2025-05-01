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
        self._summary_writer = SummaryWriter(log_dir=self._output_dir)
        self._agent = self._create_agent_fn(env=self._env)

        os.makedirs(self._output_dir, exist_ok=True)
        os.makedirs(self._checkpoint_dir, exist_ok=True)

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
            observation, reward, done, info = self._env.step(action)
            total_reward += reward
            step_number += 1

            # print(f"Step: {step_number}, Reward: {reward:.2f}, Observation: {observation}")
            if done:
                break
            action = self._agent.step(reward, observation)

        self._agent.end_episode(reward, observation)
        # user_state = self._env.user_model._user_state
        user_state = self._env._environment.user_model._user_state
        return step_number, total_reward, user_state.total_watch_time, user_state.recovered_time

    def _initialize_metrics(self):
        self._stats = {
            'episode_length': [],
            'episode_time': [],
            'episode_reward': [],
        }
        self._env.reset_metrics()

    def run_training(self, max_training_steps=250000, num_iterations=100, checkpoint_frequency=1):
        logging.info('Beginning training...')
        all_total_watch_time = []
        all_recovered_time = []

        step_level_data = {}

        start_iter, total_steps = self._load_latest_checkpoint()
        if num_iterations <= start_iter:
            logging.warning('No training needed. Exiting.')
            return

        for iteration in tqdm(range(start_iter, num_iterations), desc="Training Episode"):
            self._initialize_metrics()
            num_steps = 0
            ep_lengths, ep_rewards, ep_total_watches, ep_recovereds = [], [], [], []
            start_time = time.time()

            while num_steps < max_training_steps:
                ep_len, ep_reward, total_watch, recovered = self._run_one_episode()
                all_total_watch_time.append(total_watch)
                all_recovered_time.append(recovered)

                ep_lengths.append(ep_len)
                ep_rewards.append(ep_reward)
                ep_total_watches.append(total_watch)
                ep_recovereds.append(recovered)

                if total_steps not in step_level_data:
                    step_level_data[total_steps] = {
                        'total_watch': [],
                        'recovered': []
                    }
                step_level_data[total_steps]['total_watch'].append(total_watch)
                step_level_data[total_steps]['recovered'].append(recovered)

                num_steps += ep_len
                total_steps += ep_len

            end_time = time.time()
            avg_len = np.mean(ep_lengths)
            avg_rew = np.mean(ep_rewards)
            std_rew = np.std(ep_rewards)
            time_per_step = (end_time - start_time) / num_steps

            print(f"[TRAIN][Step {total_steps}] AvgLen: {avg_len:.2f} | AvgRew: {avg_rew:.2f} | "
                  f"StdRew: {std_rew:.2f} | Time/Step: {time_per_step:.4f}")

            if (iteration + 1) % checkpoint_frequency == 0 or (iteration + 1) == num_iterations:
                self._save_checkpoint(iteration, total_steps)

        episode_data = {
            'episode': list(range(1, len(all_total_watch_time) + 1)),
            'total_watch_time': all_total_watch_time,
            'recovered_time': all_recovered_time
        }
        with open(os.path.join(self._output_dir, 'episode_watch_times.json'), 'w') as f:
            json.dump(episode_data, f, indent=2)

        step_data = {
            'training_step': [],
            'avg_total_watch_time': [],
            'avg_recovered_time': []
        }
        for step, values in step_level_data.items():
            step_data['training_step'].append(step)
            step_data['avg_total_watch_time'].append(np.mean(values['total_watch']))
            step_data['avg_recovered_time'].append(np.mean(values['recovered']))

        with open(os.path.join(self._output_dir, 'step_watch_times.json'), 'w') as f:
            json.dump(step_data, f, indent=2)

        logging.info('Saved episode_watch_times.json and step_watch_times.json')


    def run_evaluation(self, num_eval_episodes=20):
        logging.info('Starting evaluation...')
        eval_total_watch_time = []
        eval_recovered_time = []

        self._initialize_metrics()

        for _ in tqdm(range(num_eval_episodes), desc = "Evaluation episode"):
            _, _, total_watch, recovered = self._run_one_episode()
            eval_total_watch_time.append(total_watch)
            eval_recovered_time.append(recovered)

        eval_data = {
            'episode': list(range(1, num_eval_episodes + 1)),
            'total_watch_time': eval_total_watch_time,
            'recovered_time': eval_recovered_time,
        }

        with open(os.path.join(self._output_dir, 'eval_watch_times.json'), 'w') as f:
            json.dump(eval_data, f, indent=2)

        logging.info('Saved eval_watch_times.json')

