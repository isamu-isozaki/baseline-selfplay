import numpy as np
from tqdm import tqdm
from baselines.common.runners import AbstractEnvRunner
import cv2

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, model_opponents, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.model_opponents = model_opponents
        self.opponent_states = [opponent.initial_state for opponent in self.model_opponents]
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.paths = []
    def run(self, hard_code_rate=1.0):
        import gc
        gc.collect()
        debug = False
        opponent_still = True
        only_current_model_data = True
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for i in range(len(self.model_opponents)):
            self.model_opponents[i].load_initial(self.paths, i+1)#Opponent has a random policy
        # For n in range number of steps
        for _ in tqdm(range(self.nsteps)):
            if debug:
                print("done step")
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            if debug:
                print(f'obs shape: {self.obs.shape}')
                print(f'states shape: {self.states.shape}')
                print(f'dones shape: {np.array(self.dones).shape}')
            actions, values, self.states, neglogpacs = self.model.step(self.obs[0::self.env.sides], S=self.states, M=self.dones[0::self.env.sides])
            #print(f"actions nan: {np.isnan(actions).any()} values nan: {np.isnan(values).any()} states nan: {np.isnan(self.states).any()}")
            # print(f"actions nan: {np.isnan(np.sum(actions))} action inf: {np.isinf(np.sum(actions))} values nan: {np.isnan(np.sum(values))} value inf: {np.isinf(np.sum(values))}")
            
            #self.states contains initial state of 
            if debug:
                print(f'actions: {actions.shape}')
                print(f'values: {values.shape}')
                print(f'neglogpacs: {neglogpacs.shape}')
            #Calculate for opponent models
            opponent_actions, opponent_values, opponent_neglogpacs = [], [], []
            for i in range(1, self.env.sides):
                opponent_action, opponent_value, self.opponent_states[i-1], opponent_neglogpac = self.model_opponents[i-1].step(self.obs[i::self.env.sides], S=self.opponent_states[i-1], M=self.dones[i::self.env.sides])
                if opponent_still:
                    opponent_action[:] = 0
                opponent_actions.append(opponent_action)
                opponent_values.append(opponent_value)
                opponent_neglogpacs.append(opponent_neglogpac)
            if opponent_still:
                assert np.array(opponent_actions).sum() == 0
            full_actions = np.concatenate([np.zeros_like(actions) for _ in range(self.env.sides)], axis=0)
            full_values = np.concatenate([np.zeros_like(values) for _ in range(self.env.sides)], axis=0)

            full_states = None
            if self.states is not None:
                full_states=np.concatenate([np.zeros_like(self.states) for _ in range(self.env.sides)], axis=0)
            full_neglogpacs = np.concatenate([np.zeros_like(neglogpacs) for _ in range(self.env.sides)], axis=0)
            full_actions[0::self.env.sides] = actions
            full_values[0::self.env.sides] = values
            if full_states is not None:
                full_states[0::self.env.sides] = self.states
            full_neglogpacs[0::self.env.sides] = neglogpacs
            for i in range(self.env.sides-1):
                full_actions[1+i::self.env.sides] = opponent_actions[i]
                full_values[1+i::self.env.sides] = opponent_values[i]
                if full_states is not None:
                    full_states[1+i::self.env.sides] = self.opponent_states[i]
                full_neglogpacs[1+i::self.env.sides] = opponent_neglogpacs[i]

            mb_obs.append(self.obs.copy())
            mb_actions.append(full_actions)
            mb_values.append(full_values)
            mb_neglogpacs.append(full_neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(full_actions, hard_code_rate=hard_code_rate)
            # print(f"obs nan: {np.isnan(np.sum(self.obs))} obs inf: {np.isinf(np.sum(self.obs))} rewards nan: {np.isnan(np.sum(rewards))} rewards inf: {np.isinf(np.sum(rewards))}")

            # print(f"obs max: {self.obs.max()}")
            self.obs[:] += 1e-3
            
            mb_rewards.append(rewards)
            mb_states = full_states
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
        #batch of steps to batch of rollouts
        # print(f"obs max {mb_obs.max()} obs min {mb_obs.min()} obs mean: {mb_obs.mean()} obs std: {mb_obs.std()}")
        mb_obs = np.asarray(mb_obs, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = np.zeros_like(mb_values[-1])
        for i in range(self.env.sides):
            if mb_states is not None:
                last_values[i::self.env.sides] = self.model.value(self.obs[i::self.env.sides], S=mb_states[i::self.env.sides], M=self.dones[i::self.env.sides])
            else:
                last_values[i::self.env.sides] = self.model.value(self.obs[i::self.env.sides], S=mb_states, M=self.dones[i::self.env.sides])


        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        if only_current_model_data:
            if mb_states is not None:
                return (*map(sf01, (mb_obs[0::self.env.sides], mb_returns[0::self.env.sides], mb_dones[0::self.env.sides], mb_actions[0::self.env.sides], mb_values[0::self.env.sides], mb_neglogpacs[0::self.env.sides])),
                    mb_states[0::self.env.sides], epinfos)
            else:
                return (*map(sf01, (mb_obs[0::self.env.sides], mb_returns[0::self.env.sides], mb_dones[0::self.env.sides], mb_actions[0::self.env.sides], mb_values[0::self.env.sides], mb_neglogpacs[0::self.env.sides])),
                    mb_states, epinfos)
        else:
            return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos)
    def save(self, path):
        self.paths.append(path)
        self.model.save(path)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

