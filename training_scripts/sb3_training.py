import gymnasium as gym  # For DummyEnv and type hints
import torch.nn as nn    # For activation callable
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import sync_envs_normalization  # Optional, but for clarity
from sai_rl import SAIClient  # Or 'arenax_sai' if aliased in your env
import numpy as np

# Competition setup
COMP_ID = "cmp_xnSCxcJXQclQ"
sai = SAIClient(comp_id=COMP_ID)

# NEW: Dummy env class for loading VecNormalize (matches obs space)
class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)  # Assuming 12 actions from prior logs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(45, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(45, dtype=np.float32), 0.0, False, False, {}

# Create multi-task environment (samples tasks randomly on reset)
def make_env():
    return sai.make_env()  # No env_id for multi-task

# Vectorize and normalize (training)
train_env = DummyVecEnv([lambda: Monitor(make_env())])  # Wrap in Monitor for logging
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# Print spaces for debugging
print(f"Training Observation space: {train_env.observation_space}")
print(f"Training Action space: {train_env.action_space}")

# PPO model with improved params (lower LR for stability, deeper net, vf clip)
model = PPO(
    "MlpPolicy",  # For Box spaces
    train_env,  # Use train_env
    learning_rate=1e-4,  # Lowered from 3e-4 to reduce KL explosion
    n_steps=2048,
    batch_size=128,  # Larger batch for stability
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,  # NEW: Enable VF clipping for value stability
    ent_coef=0.01,  # Encourage exploration
    verbose=1,
    device="auto",  # GPU if available
    policy_kwargs={
        "net_arch": [512, 256, 128],  # Deeper/wider for better representation
        "activation_fn": nn.ReLU  # Callable activation
    }
)

# Eval env: Match wrappers but disable reward norm for raw scores
eval_env_base = DummyVecEnv([lambda: Monitor(make_env())])
eval_env = VecNormalize(eval_env_base, norm_obs=True, norm_reward=False, clip_obs=10.0)  # Obs norm yes, reward no

eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path="./logs/", 
    log_path="./logs/", 
    eval_freq=10000, 
    deterministic=True, 
    render=False
)

# Train with more steps for improvement
total_timesteps = 5000000  # Increased to 5M for better convergence (from 1M; scale as needed)
model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

# Save model and normalized env stats
model.save("ppo_multi_task")  # NEW: Save model for reloading/submission
train_env.save("vec_normalize.pkl")

# UPDATED: BenchmarkPreprocessor with dummy env for load
class BenchmarkPreprocessor:
    def __init__(self, norm_path):
        # Create dummy VecEnv matching spaces
        dummy_base = DummyVecEnv([lambda: DummyEnv()])
        self.norm = VecNormalize.load(norm_path, dummy_base)  # Now provides venv
        self.norm.training = False  # Inference mode
        self.norm.norm_reward = False  # Don't normalize rewards here

    def __call__(self, obs):
        # Handle obstacle task: slice to common 45 dims (skip leading padding; adjust if needed)
        if obs.shape[-1] == 54:
            obs = obs[:, -45:]  # Last 45 dims match multi-task (test with [:45] if scores drop)
        elif obs.shape[-1] != 45:
            raise ValueError(f"Unexpected obs dim {obs.shape[-1]}; expected 45 or 54")
        # Apply normalization
        return self.norm.normalize_obs(obs)

preprocessor = BenchmarkPreprocessor("vec_normalize.pkl")

# Custom get_actions for benchmark
def get_actions(obs, info):
    obs_pre = preprocessor(obs)
    action, _ = model.predict(obs_pre, deterministic=True)
    return action

# Local evaluation (aligns with submission)
benchmark_results = sai.benchmark(
    model, 
    get_actions=get_actions,  # Use preprocessed obs
    use_custom_eval=True, 
    num_envs=50  # 50 episodes for robust avg
)
print(f"Benchmark Results: {benchmark_results}")

# Visualize
sai.watch(model, num_runs=3)  # Watch 3 episodes across tasks

# Close envs
train_env.close()
eval_env.close()
