#%%
import os
import numpy as np
import shutil
import torch
from pathlib import Path
from runner import Runner
import interest_evolution

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#%% 
seed = 0
np.random.seed(seed)

env_config = {
    'num_candidates': 10,
    'slate_size': 2,
    'resample_documents': True,
    'seed': seed,
}
MAX_TRAINING_STEPS = 20000
NUM_ITERATIONS = 200

#%%
#random agent
from random_agent import RandomAgent

tmp_base_dir = f'./logs/random/{MAX_TRAINING_STEPS}_{NUM_ITERATIONS}'

# Automatically delete train and eval directories if they exist
train_log_dir = os.path.join(tmp_base_dir, 'train')
eval_log_dir = os.path.join(tmp_base_dir, 'eval')

for log_dir in [train_log_dir, eval_log_dir]:
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

# Recreate base log directory
Path(tmp_base_dir).mkdir(parents=True, exist_ok=True)

def create_random_agent(env, **kwargs):
    return RandomAgent(action_space=env.action_space)
    
runner_random = Runner(
    base_dir=tmp_base_dir,
    create_agent_fn=create_random_agent,
    env=interest_evolution.create_environment(env_config),
)

runner_random.run_training(max_training_steps=MAX_TRAINING_STEPS, num_iterations=NUM_ITERATIONS)
runner_random.run_evaluation(num_eval_episodes=5)

#%%
#Best policy

from best_policy import BestPolicyAgent
tmp_base_dir = f'./logs/best_policy/{MAX_TRAINING_STEPS}_{NUM_ITERATIONS}'

train_log_dir = os.path.join(tmp_base_dir, 'train')
eval_log_dir = os.path.join(tmp_base_dir, 'eval')

def create_best_policy_agent(env, **kwargs):
    obs = env.reset()
    user_dim = obs['user'].shape[0]
    
    first_doc = next(iter(obs['doc'].values()))
    doc_dim = first_doc.shape[0]
    n_arms = len(obs['doc'])
    slate_size = env_config['slate_size']
    
    return BestPolicyAgent(doc_dim=doc_dim, n_arms=n_arms, slate_size=slate_size)

runner_best = Runner(
    base_dir=tmp_base_dir,
    create_agent_fn=create_best_policy_agent,
    env=interest_evolution.create_environment(env_config)
)

runner_best.run_training(max_training_steps=MAX_TRAINING_STEPS, num_iterations=NUM_ITERATIONS)
runner_best.run_evaluation(num_eval_episodes=5)

#%%
#UCB bandit

from bandit import BanditAgentWrapper

tmp_base_dir = f'./logs/bandit/{MAX_TRAINING_STEPS}_{NUM_ITERATIONS}'

def create_bandit_agent(env):
    return BanditAgentWrapper(n_arms=env.action_space.nvec[0])

train_log_dir = os.path.join(tmp_base_dir, 'train')
eval_log_dir = os.path.join(tmp_base_dir, 'eval')
for log_dir in [train_log_dir, eval_log_dir]:
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
Path(tmp_base_dir).mkdir(parents=True, exist_ok=True)

# Run Bandit agent
runner_bandit = Runner(
    base_dir=tmp_base_dir,
    create_agent_fn=create_bandit_agent,
    env=interest_evolution.create_environment(env_config),
)

runner_bandit.run_training(max_training_steps=MAX_TRAINING_STEPS, num_iterations=NUM_ITERATIONS)
runner_bandit.run_evaluation(num_eval_episodes=5)

#%%
#Contextual bandit doc-only and user-doc
from interest_evolution import create_environment
from contextual_bandit import DocOnlyContextualBanditAgent, UserDocContextualBanditAgent

def clean_log_dirs(base_dir):
    for sub in ['train', 'eval']:
        log_dir = os.path.join(base_dir, sub)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
    Path(base_dir).mkdir(parents=True, exist_ok=True)

#agent factory
def create_contextual_bandit_agent(env, use_user_context=False):
    doc_obs_space = env.observation_space['doc']
    user_obs_space = env.observation_space['user']
    first_doc_key = list(doc_obs_space.spaces.keys())[0]

    doc_dim = doc_obs_space.spaces[first_doc_key].shape[0]
    user_dim = user_obs_space.shape[0]
    n_arms = len(doc_obs_space.spaces)

    if use_user_context:
        print(f"[Bandit] Using user + doc context (dim = {user_dim + doc_dim})")
        return UserDocContextualBanditAgent(
            user_dim=user_dim,
            doc_dim=doc_dim,
            n_arms=n_arms,
            epsilon=0.1
        )
    else:
        print(f"[Bandit] Using doc-only context (dim = {doc_dim})")
        return DocOnlyContextualBanditAgent(
            doc_dim=doc_dim,
            n_arms=n_arms,
            epsilon=0.1
        )

def run_contextual_bandit(label, use_user_context):
    log_dir = f'./logs/Contextual_bandit_{label}/{MAX_TRAINING_STEPS}_{NUM_ITERATIONS}'
    clean_log_dirs(log_dir)

    env = create_environment(env_config)

    runner = Runner(
        base_dir=log_dir,
        create_agent_fn=lambda env: create_contextual_bandit_agent(env, use_user_context),
        env=env,
    )

    print(f"\n=== Training Contextual Bandit ({label}) ===")
    runner.run_training(max_training_steps=MAX_TRAINING_STEPS, num_iterations=NUM_ITERATIONS)
    runner.run_evaluation(max_eval_episodes=NUM_EVAL_EPISODES)

#run both agents
run_contextual_bandit(label="DocOnly", use_user_context=False)
run_contextual_bandit(label="UserDoc", use_user_context=True)

#%%
#Naive DQN
from dqn import DQNAgent
from gymnasium import spaces

tmp_base_dir = f'./logs/naive_dqn/{MAX_TRAINING_STEPS}_{NUM_ITERATIONS}'

print("Start DQN")

def create_dqn_agent(env, double=False, **kwargs):
    doc_obs_space = env.observation_space['doc']

    if isinstance(doc_obs_space, spaces.Dict):
        first_key = list(doc_obs_space.spaces.keys())[0]
        per_doc_dim = doc_obs_space.spaces[first_key].shape[0]
        num_docs = len(doc_obs_space.spaces)
        obs_dim = per_doc_dim * num_docs
    else:
        raise ValueError("Unsupported doc observation space:", doc_obs_space)

    if isinstance(env.action_space, spaces.Discrete):
        n_actions = env.action_space.n
    elif isinstance(env.action_space, spaces.MultiDiscrete):
        n_actions = env.action_space.nvec[0]
    else:
        raise ValueError("Unsupported action space type")

    print(f"obs_dim: {obs_dim}, n_actions: {n_actions}")
    return DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        double=double  # ✅ indicate whether it is double dqn
    )

train_log_dir = os.path.join(tmp_base_dir, 'train')
eval_log_dir = os.path.join(tmp_base_dir, 'eval')

for log_dir in [train_log_dir, eval_log_dir]:
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

Path(tmp_base_dir).mkdir(parents=True, exist_ok=True)

runner_naive_dqn = Runner(
    base_dir=tmp_base_dir,
    create_agent_fn=lambda env: create_dqn_agent(env, double=False),
    env=interest_evolution.create_environment(env_config),
)

runner_naive_dqn.run_training(max_training_steps=MAX_TRAINING_STEPS, num_iterations=NUM_ITERATIONS)
runner_naive_dqn.run_evaluation(num_eval_episodes=5)

#%%
#Double DQN
tmp_base_dir = f'./logs/double_dqn_{MAX_TRAINING_STEPS}_{NUM_ITERATIONS}'
train_log_dir = os.path.join(tmp_base_dir, 'train')
eval_log_dir = os.path.join(tmp_base_dir, 'eval')

for log_dir in [train_log_dir, eval_log_dir]:
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

Path(tmp_base_dir).mkdir(parents=True, exist_ok=True)

runner_double_dqn = Runner(
    base_dir=tmp_base_dir,
    create_agent_fn=lambda env: create_dqn_agent(env, double=True),
    env=interest_evolution.create_environment(env_config),
)

runner_double_dqn.run_training(max_training_steps=MAX_TRAINING_STEPS, num_iterations=NUM_ITERATIONS)
runner_double_dqn.run_evaluation(num_eval_episodes=5)
