import multiprocessing as mp
import sys
from copy import deepcopy
from enum import Enum

import numpy as np
from gym.error import AlreadyPendingCallError, NoAsyncCallError
from gym.spaces import Tuple
from gym.vector import AsyncVectorEnv
from gym.vector.async_vector_env import AsyncState
from gym.vector.utils import concatenate


class ContextualAsyncState(Enum):
    WAITING_SET_CONTEXT = 'set_context'
    WAITING_SAMPLE_CONTEXTS = 'sample_contexts'
    WAITING_GET_CONTEXT_PUNISHMENT = 'get_context_punishment'


class ContextualAsyncVectorEnv(AsyncVectorEnv):
    """Vectorized contextual environment that runs multiple environments in parallel. It
        uses `multiprocessing` processes, and pipes for communication.

        Parameters
        ----------
        env_fns : iterable of callable
            Functions that create the environments.

        observation_space : `gym.spaces.Space` instance, optional
            Observation space of a single environment. If `None`, then the
            observation space of the first environment is taken.

        action_space : `gym.spaces.Space` instance, optional
            Action space of a single environment. If `None`, then the action space
            of the first environment is taken.

        context_space : `gym.spaces.Space` instance, optional
            Context space of a single environment. If `None`, then the context space
            of the first environment is taken.

        shared_memory : bool (default: `True`)
            If `False`, then the observations from the worker processes are
            communicated back through shared variables. This can improve the
            efficiency if the observations are large (e.g. images).

        copy : bool (default: `True`)
            If `True`, then the `reset` and `step` methods return a copy of the
            observations.

        context : str, optional
            Context for multiprocessing. If `None`, then the default context is used.
            Only available in Python 3.

        daemon : bool (default: `True`)
            If `True`, then subprocesses have `daemon` flag turned on; that is, they
            will quit if the head process quits. However, `daemon=True` prevents
            subprocesses to spawn children, so for some environments you may want
            to have it set to `False`

        worker : function, optional
            WARNING - advanced mode option! If set, then use that worker in a subprocess
            instead of a default one. Can be useful to override some inner vector env
            logic, for instance, how resets on done are handled. Provides high
            degree of flexibility and a high chance to shoot yourself in the foot; thus,
            if you are writing your own worker, it is recommended to start from the code
            for `_worker` (or `_worker_shared_memory`) method below, and add changes
        """

    def __init__(self, env_fns, observation_space=None, action_space=None, context_space=None,
                 shared_memory=False, copy=True, context=None, daemon=True, worker=None):
        super(ContextualAsyncVectorEnv, self).__init__(env_fns=env_fns, observation_space=observation_space,
                                                       action_space=action_space, shared_memory=shared_memory,
                                                       copy=copy, context=context, daemon=daemon,
                                                       worker=worker or _contextual_worker)
        if context_space is None:
            dummy_env = env_fns[0]()
            context_space = dummy_env.context_space
            dummy_env.close()
            del dummy_env
        self.single_context_space = context_space
        self.context_space = Tuple((context_space,) * self.num_envs)

    def step_async(self, actions):
        """
        Parameters
        ----------
        actions : iterable of samples from `action_space`
            List of actions.
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError('Calling `step_async` while waiting '
                                          'for a pending call to `{0}` to complete.'.format(
                self._state.value), self._state.value)

        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(('step', action))
        self._active_pipes = actions.shape[0]
        self._state = AsyncState.WAITING_STEP

    def step_wait(self, timeout=None):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `step_wait` times out. If
            `None`, the call to `step_wait` never times out.

        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.

        rewards : `np.ndarray` instance (dtype `np.float_`)
            A vector of rewards from the vectorized environment.

        dones : `np.ndarray` instance (dtype `np.bool_`)
            A vector whose entries indicate whether the episode has ended.

        infos : list of dict
            A list of auxiliary diagnostic information.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError('Calling `step_wait` without any prior call '
                                   'to `step_async`.', AsyncState.WAITING_STEP.value)

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError('The call to `step_wait` has timed out after '
                                  '{0} second{1}.'.format(timeout, 's' if timeout > 1 else ''))

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes[:self._active_pipes]])
        self._raise_if_errors(successes)

        observations_list, rewards, dones, infos = zip(*results)
        if not self.shared_memory:
            # observations = concatenate(observations_list, self.observations[:self._active_pipes],
            #                                 self.single_observation_space)
            observations = concatenate(self.single_observation_space, observations_list,
                                       self.observations[:self._active_pipes])

        self._active_pipes = self.num_envs
        self._state = AsyncState.DEFAULT

        return (deepcopy(observations[:self._active_pipes]) if self.copy else observations[:self._active_pipes],
                np.array(rewards), np.array(dones, dtype=np.bool_), infos)

    def sample_contexts(self, n_samples: int):
        self.sample_contexts_async(n_samples)
        return self.sample_contexts_wait()

    def sample_contexts_async(self, n_samples):
        """
        Parameters
        ----------
        n_samples : integer of number of context samples for current env
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError('Calling `sample_contexts_async` while waiting '
                                          'for a pending call to `{0}` to complete.'.format(
                self._state.value), self._state.value)

        for pipe, n_sample in zip(self.parent_pipes, n_samples):
            pipe.send(('sample_contexts', n_sample))
        self._active_pipes = n_samples.shape[0]
        self._state = ContextualAsyncState.WAITING_SAMPLE_CONTEXTS

    def sample_contexts_wait(self, timeout=None):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `sample_contexts_wait` times out. If
            `None`, the call to `sample_contexts_wait` never times out.
        """
        self._assert_is_running()
        if self._state != ContextualAsyncState.WAITING_SAMPLE_CONTEXTS:
            raise NoAsyncCallError('Calling `sample_contexts_wait` without any prior '
                                   'call to `sample_contexts_wait`.', ContextualAsyncState.WAITING_SAMPLE_CONTEXTS.value)

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError('The call to `sample_contexts_wait` has timed out after '
                                  '{0} second{1}.'.format(timeout, 's' if timeout > 1 else ''))

        ctxts, successes = zip(*[pipe.recv() for pipe in self.parent_pipes[:self._active_pipes]])
        self._raise_if_errors(successes)
        self._active_pipes = self.num_envs
        self._state = AsyncState.DEFAULT
        return np.array(ctxts)

    def set_context(self, contexts):
        r"""Take a context for each sub-environments.

        Parameters
        ----------
        contexts : iterable of samples from `context_space`
            List of contexts.
        """

        self.set_context_async(contexts)
        return self.set_context_wait()

    def set_context_async(self, contexts):
        """
        Parameters
        ----------
        contexts : iterable of samples from `context_space`
            List of contexts.
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError('Calling `set_context_async` while waiting '
                                          'for a pending call to `{0}` to complete.'.format(
                self._state.value), self._state.value)

        for pipe, context in zip(self.parent_pipes, contexts):
            pipe.send(('set_context', context))

        self._active_pipes = contexts.shape[0]
        self._state = ContextualAsyncState.WAITING_SET_CONTEXT

    def set_context_wait(self, timeout=None):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `set_context_wait` times out. If
            `None`, the call to `set_context_wait` never times out.
        """
        self._assert_is_running()
        if self._state != ContextualAsyncState.WAITING_SET_CONTEXT:
            raise NoAsyncCallError('Calling `set_context_wait` without any prior '
                                   'call to `set_context_async`.', ContextualAsyncState.WAITING_SET_CONTEXT.value)

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError('The call to `set_context_wait` has timed out after '
                                  '{0} second{1}.'.format(timeout, 's' if timeout > 1 else ''))

        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes[:self._active_pipes]])
        self._raise_if_errors(successes)
        self._active_pipes = self.num_envs
        self._state = AsyncState.DEFAULT

    def get_context_punishment(self, contexts):
        r"""Return a batch of out of context space punishments.

        Returns
        -------
        punishments : `np.ndarray` instance (dtype `np.float_`)
            A vector of punishments from the vectorized environment.
        """
        self.get_context_punishment_async(contexts)
        return self.get_context_punishment_wait()

    def get_context_punishment_async(self, contexts):
        """
        Parameters
        ----------
        contexts : iterable of samples possibly from `context_space`
            List of contexts.
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError('Calling `get_context_punishment_async` while waiting '
                                          'for a pending call to `{0}` to complete.'.format(
                self._state.value), self._state.value)

        for pipe, context in zip(self.parent_pipes, contexts):
            pipe.send(('get_context_punishment', context))
        self._active_pipes = contexts.shape[0]
        self._state = ContextualAsyncState.WAITING_GET_CONTEXT_PUNISHMENT

    def get_context_punishment_wait(self, timeout=None):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `step_wait` times out. If
            `None`, the call to `step_wait` never times out.

        Returns
        -------
        punishment : `np.ndarray` instance (dtype `np.float_`)
            A vector of punishments from the vectorized environment.
        """
        self._assert_is_running()
        if self._state != ContextualAsyncState.WAITING_GET_CONTEXT_PUNISHMENT:
            raise NoAsyncCallError('Calling `get_context_punishment_wait` without any prior call '
                                   'to `get_context_punishment_async`.',
                                   ContextualAsyncState.WAITING_GET_CONTEXT_PUNISHMENT.value)

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError('The call to `get_context_punishment_wait` has timed out after '
                                  '{0} second{1}.'.format(timeout, 's' if timeout > 1 else ''))

        punishments, successes = zip(*[pipe.recv() for pipe in self.parent_pipes[:self._active_pipes]])
        self._raise_if_errors(successes)
        self._active_pipes = self.num_envs
        self._state = AsyncState.DEFAULT

        return np.array(punishments)


def _contextual_worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                if "return_info" in data and data["return_info"] is True:
                    observation, info = env.reset(**data)
                    pipe.send(((observation, info), True))
                else:
                    observation = env.reset(**data)
                    pipe.send((observation, True))
            elif command == "step":
                observation, reward, done, info = env.step(data)
                if done:
                    info["terminal_observation"] = observation
                    observation = env.reset()
                pipe.send(((observation, reward, done, info), True))
            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))
            elif command == 'sample_contexts':
                ctxts = env.sample_contexts(data)
                pipe.send((ctxts, True))
            elif command == 'close':
                pipe.send((None, True))
                break
            elif command == 'set_context':
                env.set_context(data)
                pipe.send((None, True))
            elif command == 'get_context_punishment':
                punishment = env.get_context_punishment(data)
                pipe.send((punishment, True))
            elif command == '_check_observation_space':
                pipe.send((data == env.observation_space, True))
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                pipe.send(
                    (
                        (data[0] == env.observation_space, data[1] == env.action_space),
                        True,
                    )
                )
            else:
                raise RuntimeError(f'Received unknown command `{command}`. Must '
                                   'be one of {`reset`, `step`, `seed`, `close`, `set_context`, `get_context_punishment`, '
                                   '`_check_observation_space`}.')
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
