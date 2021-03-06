import multiprocessing as mp
import gym
import numpy as np
from .vec_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars


def worker(remote, parent_remote, env_fn_wrappers):
    def step_env(env, action):
        ob, reward, done, info = env.step(**action)
        if type(done) == list and done[0]:
            ob = env.reset()
            ob = ob[0]
        if type(done) != list and done:
            ob = env.reset()
        return ob, reward, done, info

    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'render':
                remote.send([env.render(mode='rgb_array') for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].action_space, envs[0].spec)))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None, context='spawn', in_series=1):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        in_series: number of environments to run in series in a single process
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        if(hasattr(env_fns[0](), 'sides')):
            self.sides = env_fns[0]().sides
        else:
            self.sides = 1
        env_fns = env_fns[:nenvs//self.sides]
        assert nenvs//self.sides % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs //self.sides // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nremotes)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv().x
        
        self.viewer = None
        VecEnv.__init__(self, nenvs, observation_space, action_space)

    def step_async(self, actions, hard_code_rate=1.0, **kwargs):
        self._assert_not_closed()
        actions = np.array_split(actions, self.nremotes)
        action_in_series = []
        for j in range(len(actions)):
            for action in actions[j]:
                action_in_series.append({'action': action, 'hard_code_rate': hard_code_rate, **kwargs})
                if(len(action_in_series) == self.in_series):
                    self.remotes[j].send(('step', action_in_series))
                    #TODO: Update readme as now actions are sent in dictionaries
                    action_in_series = []
                
        self.waiting = True
    def tactic_game_fix_results(self, results):
        for i in range(len(results)-1, -1, -1):
            for j in range(len(results[i])):
                results[i][j] = results[i//self.sides][j][i % self.sides]
        return results
    def step_wait(self):
        self._assert_not_closed()
        #do recv on the same remote several times
        results = [self.remotes[i//self.sides].recv() for i in range(len(self.remotes)*self.sides)]
        results = _flatten_list(results)
        data = results.copy()[self.sides-1::self.sides]
        results = np.asarray(results)
        results[:len(self.remotes)] = data
        #push the observations to the first portion of the results array.
        if self.sides > 1:
            results = self.tactic_game_fix_results(results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        print("Called reset")
        self._assert_not_closed()
        for i in range(len(self.remotes)):
            self.remotes[i].send(('reset', None))
        obs = [self.remotes[i].recv() for i, _ in enumerate(self.remotes)]
        obs = _flatten_list(obs)
        if self.sides > 1:
            obs += [[None] for _ in range(len(self.remotes)*(self.sides-1))]
            obs = self.tactic_game_fix_results(obs)     
            obs = zip(*obs)
            obs = obs.__next__()
        return _flatten_obs(obs)

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        imgs = _flatten_list(imgs)
        if self.sides > 1:
            imgs += [[None] for _ in range(len(self.remotes)*(self.sides-1))]
            imgs = self.tactic_game_fix_results(imgs)
            imgs = zip(*imgs)
            imgs = imgs.__next__()
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()

def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)

def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]
