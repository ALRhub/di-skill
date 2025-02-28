import fancy_gym

from contextual_envs.contextual_async_vector_env import ContextualAsyncVectorEnv


def create_env_fn(env_id, seed, index, **kwargs):
    def env_fn():
        return fancy_gym.make(env_id, seed + index, **kwargs)

    return env_fn


def create_envs(env_id, seed, n_envs, **kwargs):
    env_fns = [create_env_fn(env_id, seed, i, **kwargs) for i in range(n_envs)]
    vec_env_fun = ContextualAsyncVectorEnv
    try:
        env = vec_env_fun(env_fns)
    except AssertionError:
        env = vec_env_fun(env_fns, daemon=False)

    return env
