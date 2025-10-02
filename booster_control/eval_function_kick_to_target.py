import numpy as np
_FLOAT_EPS = np.finfo(np.float64).eps

reward_config = {
    "offside": -1.0, 
    "success": 2.0, 
    "distance": 0.5,
    "steps": -0.2,
}

def evaluation_fn(env, eval_state):
    if not eval_state.get("timestep", False):
        eval_state["timestep"] = 0

    raw_reward = env.compute_reward()

    raw_reward.update({"steps": np.float64(1.0)})

    eval_state["timestep"] += 1

    if eval_state.get("terminated", False) or eval_state.get("truncated", False):
        reward = 0.0
        for key, value in raw_reward.items():
            if key in reward_config:
                # handle boolean values as 1.0 or 0.0
                val = float(value) if not isinstance(value, bool) else (1.0 if value else 0.0)
                reward += val * reward_config[key]

        return (reward, eval_state)

    return (0.0, eval_state)
