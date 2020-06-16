import numpy as np
from tqdm import tqdm
from baselines.common.runners import AbstractEnvRunner

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
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.paths = []
    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for i in range(len(self.model_opponents)):
            self.model_opponents[i].load_from_random(self.paths)#Opponent has a random policy
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states[0::self.env.sides], neglogpacs = self.model.step(self.obs[0::self.env.sides], S=self.states[0::self.env.sides], M=self.dones[0::self.env.sides])
            opponent_actions, opponent_values, opponent_states, opponent_neglogpacs = [], [], [], []
            for i in range(1, self.env.sides):
                opponent_action, opponent_value, self.states[i::self.env.sides], opponent_neglogpac = self.model_opponents[i].step(self.obs[i::self.env.sides], S=self.states[i::self.env.sides], M=self.dones[i::self.env.sides])
                opponent_actions.append(opponent_action)
                opponent_values.append(opponent_value)
                opponent_neglogpacs.append(opponent_neglogpac)
            full_actions = np.concatenate([np.zeros_like(actions) for _ in range(self.env.sides)], axis=0)
            full_values = np.concatenate([np.zeros_like(values) for _ in range(self.env.sides)], axis=0)
            full_neglogpacs = np.concatenate([np.zeros_like(neglogpacs) for _ in range(self.env.sides)], axis=0)
            full_actions[0::self.env.sides] = actions
            full_values[0::self.env.sides] = values
            full_neglogpacs[0::self.env.sides] = neglogpacs
            for i in range(self.env.sides-1):
                full_actions[1+i::self.env.sides] = opponent_actions[i]
                full_values[1+i::self.env.sides] = opponent_values[i]
                full_neglogpacs[1+i::self.env.sides] = opponent_neglogpacs[i]
            mb_obs.append(self.obs.copy())
            mb_actions.append(full_actions)
            mb_values.append(full_values)
            mb_neglogpacs.append(full_neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

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
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
    def save(self, path):
        paths.append(path)
        self.model.save(path)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

