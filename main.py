import os
import numpy as np
import shutil
import torch
from pathlib import Path
from runner import Runner
import interest_evolution
from bandit import BanditAgentWrapper
from random_agent import RandomAgent


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
#%%
tmp_base_dir = './logs/random'

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
    
# --- Initialize training runner ---
runner_random = Runner(
    base_dir=tmp_base_dir,
    create_agent_fn=create_random_agent,
    env=interest_evolution.create_environment(env_config),
)

# --- Run training + evaluation ---
runner_random.run_training(max_training_steps=2000, num_iterations=200)
runner_random.run_evaluation(max_eval_episodes=5)

# %%
# UCB_Bandit 
tmp_base_dir = './logs/UCB_bandit'
def create_bandit_agent(env):
    return BanditAgentWrapper(n_arms=env.action_space.nvec[0])

# Delete and recreate directories
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

runner_bandit.run_training(max_training_steps=2000, num_iterations=200)
runner_bandit.run_evaluation(max_eval_episodes=5)

#%%
# tmp_base_dir = './logs/Contextual_bandit'
from contextual_bandit import DocOnlyContextualBanditAgent, UserDocContextualBanditAgent

# def create_contextual_bandit_agent(env, **kwargs):
#     n_arms = env.action_space.nvec[0]  # Use .nvec[0] for MultiDiscrete
#     return ContextualBanditAgent(n_arms=n_arms, epsilon=0.1)

# train_log_dir = os.path.join(tmp_base_dir, 'train')
# eval_log_dir = os.path.join(tmp_base_dir, 'eval')

# for log_dir in [train_log_dir, eval_log_dir]:
#     if os.path.exists(log_dir):
#         shutil.rmtree(log_dir)

# Path(tmp_base_dir).mkdir(parents=True, exist_ok=True)

# runner_cb = Runner(
#     base_dir=tmp_base_dir,
#     create_agent_fn=create_contextual_bandit_agent,
#     env=interest_evolution.create_environment(env_config),
# )

# runner_cb.run_training(max_training_steps=2000, num_iterations=100)
# runner_cb.run_evaluation(max_eval_episodes=5)

#%%
# DocOnlyContextual 和UserDocContextual
from interest_evolution import create_environment
def create_contextual_bandit_agent(env_config, use_user_context=False):
    env = create_environment(env_config)
    doc_obs_space = env.observation_space['doc']
    user_obs_space = env.observation_space['user']
    
    # 获取文档特征维度
    first_doc_key = list(doc_obs_space.spaces.keys())[0]
    doc_dim = doc_obs_space.spaces[first_doc_key].shape[0]
    
    # 获取用户兴趣向量维度
    user_dim = user_obs_space.shape[0]
    
    # 获取文档数量（arms数量）
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

tmp_base_dir_1 = './logs/Contextual_bandit_DocOnly'
tmp_base_dir_2 = './logs/Contextual_bandit_UserDoc'

train_log_dir_1 = os.path.join(tmp_base_dir_1, 'train')
eval_log_dir_1 = os.path.join(tmp_base_dir_1, 'eval')


for log_dir in [train_log_dir_1, eval_log_dir_1]:
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

Path(tmp_base_dir_1).mkdir(parents=True, exist_ok=True)

runner_cb_doc = Runner(
    base_dir=tmp_base_dir_1,
    create_agent_fn=lambda env: create_contextual_bandit_agent(env_config, use_user_context=False),
    env=interest_evolution.create_environment(env_config),
)
#%%
runner_cb_doc.run_training(max_training_steps=2000, num_iterations=200)
runner_cb_doc.run_evaluation(max_eval_episodes=5)
#%%
# USER+DOC
train_log_dir_2 = os.path.join(tmp_base_dir_2, 'train')
eval_log_dir_2 = os.path.join(tmp_base_dir_2, 'eval')


for log_dir in [train_log_dir_2, eval_log_dir_2]:
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

Path(tmp_base_dir_2).mkdir(parents=True, exist_ok=True)

runner_cb_user_doc = Runner(
    base_dir=tmp_base_dir_2,
    create_agent_fn=lambda env: create_contextual_bandit_agent(env_config, use_user_context=True),
    env=interest_evolution.create_environment(env_config),
)
#%%
runner_cb_user_doc.run_training(max_training_steps=2000, num_iterations=200)
runner_cb_user_doc.run_evaluation(max_eval_episodes=5)

#%%
from naive_dqn import NaiveDQNAgent
from gymnasium import spaces
tmp_base_dir = './logs/naive_dqn'

def create_naive_dqn_agent(env, **kwargs):
    doc_obs_space = env.observation_space['doc']

    if isinstance(doc_obs_space, spaces.Dict):
        # 每个文档的 shape 应该一致，取第一个文档的 shape[0]
        first_key = list(doc_obs_space.spaces.keys())[0]
        per_doc_dim = doc_obs_space.spaces[first_key].shape[0]
        num_docs = len(doc_obs_space.spaces)
        obs_dim = per_doc_dim * num_docs  # e.g., 20 * 10 = 200
    else:
        raise ValueError("Unsupported doc observation space:", doc_obs_space)

    if isinstance(env.action_space, spaces.Discrete):
        n_actions = env.action_space.n
    elif isinstance(env.action_space, spaces.MultiDiscrete):
        n_actions = env.action_space.nvec[0]
    else:
        raise ValueError("Unsupported action space type")

    print(f"obs_dim: {obs_dim}, n_actions: {n_actions}")
    return NaiveDQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        epsilon=kwargs.get('epsilon', 0.1),
        gamma=kwargs.get('gamma', 0.99),
        lr=kwargs.get('lr', 1e-3)
    )

train_log_dir = os.path.join(tmp_base_dir, 'train')
eval_log_dir = os.path.join(tmp_base_dir, 'eval')

for log_dir in [train_log_dir, eval_log_dir]:
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

Path(tmp_base_dir).mkdir(parents=True, exist_ok=True)
#%%
runner_naive_dqn = Runner(
    base_dir=tmp_base_dir,
    create_agent_fn=create_naive_dqn_agent,
    env=interest_evolution.create_environment(env_config),
)
#%%
runner_naive_dqn.run_training(max_training_steps=2000, num_iterations=200)
runner_naive_dqn.run_evaluation(max_eval_episodes=5)
#%%
# Revised Naive DQN
from naive_dqn_re import NaiveDQN_target_Agent
tmp_base_dir = './logs/naive_dqn_target_network'

def create_naive_dqn_re_agent(env, **kwargs):
    doc_obs_space = env.observation_space['doc']

    if isinstance(doc_obs_space, spaces.Dict):
        # 每个文档的 shape 应该一致，取第一个文档的 shape[0]
        first_key = list(doc_obs_space.spaces.keys())[0]
        per_doc_dim = doc_obs_space.spaces[first_key].shape[0]
        num_docs = len(doc_obs_space.spaces)
        obs_dim = per_doc_dim * num_docs  # e.g., 20 * 10 = 200
    else:
        raise ValueError("Unsupported doc observation space:", doc_obs_space)

    if isinstance(env.action_space, spaces.Discrete):
        n_actions = env.action_space.n
    elif isinstance(env.action_space, spaces.MultiDiscrete):
        n_actions = env.action_space.nvec[0]
    else:
        raise ValueError("Unsupported action space type")

    print(f"obs_dim: {obs_dim}, n_actions: {n_actions}")
    return NaiveDQN_target_Agent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        epsilon=kwargs.get('epsilon', 1),
        gamma=kwargs.get('gamma', 0.99),
        lr=kwargs.get('lr', 1e-4)
    )
train_log_dir = os.path.join(tmp_base_dir, 'train')
eval_log_dir = os.path.join(tmp_base_dir, 'eval')

for log_dir in [train_log_dir, eval_log_dir]:
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

Path(tmp_base_dir).mkdir(parents=True, exist_ok=True)
#%%
runner_naive_dqn_re = Runner(
    base_dir=tmp_base_dir,
    create_agent_fn=create_naive_dqn_re_agent,
    env=interest_evolution.create_environment(env_config),
)
#%%
runner_naive_dqn_re.run_training(max_training_steps=2000, num_iterations=100)
runner_naive_dqn_re.run_evaluation(max_eval_episodes=5)

#%%
# Double DQN
from double_dqn import DoubleDQNAgent
tmp_base_dir = './logs/double_dqn'

def create_double_dqn_agent(env, **kwargs):
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

    print(f"[DoubleDQN] obs_dim: {obs_dim}, n_actions: {n_actions}")
    return DoubleDQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        epsilon=kwargs.get('epsilon', 0.1),
        gamma=kwargs.get('gamma', 0.99),
        lr=kwargs.get('lr', 1e-3),
        tau=kwargs.get('tau', 0.005)  # soft update rate
    )
train_log_dir = os.path.join(tmp_base_dir, 'train')
eval_log_dir = os.path.join(tmp_base_dir, 'eval')

for log_dir in [train_log_dir, eval_log_dir]:
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

Path(tmp_base_dir).mkdir(parents=True, exist_ok=True)
runner_double_dqn = Runner(
    base_dir=tmp_base_dir,
    create_agent_fn=create_double_dqn_agent,
    env=interest_evolution.create_environment(env_config),
)
#%%
runner_double_dqn.run_training(max_training_steps=2000, num_iterations=100)
runner_double_dqn.run_evaluation(max_eval_episodes=5)

#%%
from multistep_dqn import MultiStepDQNAgent
tmp_base_dir = './logs/multistep_dqn'

def create_multistep_dqn_agent(env, **kwargs):
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

    print(f"[MultiStepDQN] obs_dim: {obs_dim}, n_actions: {n_actions}")
    return MultiStepDQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        n_step=kwargs.get('n_step', 3),
        epsilon=kwargs.get('epsilon', 0.1),
        gamma=kwargs.get('gamma', 0.99),
        lr=kwargs.get('lr', 1e-3)
    )
train_log_dir = os.path.join(tmp_base_dir, 'train')
eval_log_dir = os.path.join(tmp_base_dir, 'eval')

for log_dir in [train_log_dir, eval_log_dir]:
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

Path(tmp_base_dir).mkdir(parents=True, exist_ok=True)
runner_multistep_dqn = Runner(
    base_dir=tmp_base_dir,
    create_agent_fn=create_multistep_dqn_agent,
    env=interest_evolution.create_environment(env_config),
)

#%%
runner_multistep_dqn.run_training(max_training_steps=10, num_iterations=1)
runner_multistep_dqn.run_evaluation(max_eval_episodes=5)

# %%
