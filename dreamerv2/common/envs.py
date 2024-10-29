import atexit
import os
import sys
import threading
import traceback

import cloudpickle
import gym
import numpy as np
from PIL import Image

try:
    from mani_skill2.utils.sapien_utils import vectorize_pose

except:
    pass


class NormalizeActions:

    def __init__(self, env):
        self._env = env

        self._mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high))

        self._low = np.where(self._mask, env.action_space.low, -1)

        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)

        return self._env.step(original)


class NormObsWrapper:

    def __init__(self, env, obs_min, obs_max, keys=None):
        self._env = env
        self.obs_min = obs_min
        self.obs_max = obs_max
        self.keys = keys

    def __getattr__(self, name):
        return getattr(self._env, name)

    def norm_ob_dict(self, ob_dict):
        ob_dict = ob_dict.copy()
        if self.keys is None:
            for k, v in ob_dict.items():
                ob_dict[k] = (v - self.obs_min) / (self.obs_max - self.obs_min)
        else:
            for k in self.keys:
                v = ob_dict[k]
                ob_dict[k] = (v - self.obs_min) / (self.obs_max - self.obs_min)
        return ob_dict

    def step(self, action):
        obs, rew, done, info = self._env.step(action)
        return self.norm_ob_dict(obs), rew, done, info

    def reset(self):
        return self.norm_ob_dict(self._env.reset())

    def norm_ob(self, ob):
        return (ob - self.obs_min) / (self.obs_max - self.obs_min)

    def get_goals(self):
        goals = self._env.get_goals()
        norm_goals = np.stack([self.norm_ob(g) for g in goals])
        return norm_goals


class ConvertGoalEnvWrapper:
    """
    Given a GoalEnv that returns obs dict {'observation', 'achieved_goal', 'desired_goal'}, we modify obs dict to just contain {'observation', 'goal'} where 'goal' is desired goal.
    """

    def __init__(self, env, obs_key='observation', goal_key='goal'):
        self._env = env
        self.obs_key = obs_key
        self.goal_key = goal_key

        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
        assert self._obs_is_dict, "GoalEnv should have obs dict"

        self._act_is_dict = hasattr(self._env.action_space, 'spaces')

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):

        obs, reward, done, info = self._env.step(action)
        obs = {self.obs_key: obs[self.obs_key],
               self.goal_key: obs['desired_goal']}
        return obs, reward, done, info

    def reset(self):

        obs = self._env.reset()
        obs = {self.obs_key: obs[self.obs_key],
               self.goal_key: obs['desired_goal']}
        return obs

    @property
    def observation_space(self):

        return gym.spaces.Dict({self.obs_key: self._env.observation_space[self.obs_key], self.goal_key: self._env.observation_space["desired_goal"]})


class GymWrapper:
    """modifies obs space, action space,
    modifies step and reset fn, just returns ob dict now. ob dict has reward, done, etc.
    """

    def __init__(self, env, obs_key='image', act_key='action', info_to_obs_fn=None):
        self._env = env

        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')

        self._act_is_dict = hasattr(self._env.action_space, 'spaces')

        self._obs_key = obs_key
        self._act_key = act_key
        self.info_to_obs_fn = info_to_obs_fn

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}
        return {
            **spaces,
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

    @property
    def act_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()
        else:
            return {self._act_key: self._env.action_space}

    def step(self, action):
        if not self._act_is_dict:
            action = action[self._act_key]
        obs, reward, done, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs['reward'] = float(reward)
        obs['is_first'] = False
        obs['is_last'] = done
        obs['is_terminal'] = info.get('is_terminal', done)
        if self.info_to_obs_fn:
            obs = self.info_to_obs_fn(info, obs)
        return obs

    def reset(self):
        obs = self._env.reset()
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs['reward'] = 0.0
        obs['is_first'] = True
        obs['is_last'] = False
        obs['is_terminal'] = False
        if self.info_to_obs_fn:
            obs = self.info_to_obs_fn(None, obs)
        return obs


class GymnasiumWrapper:
    """modifies obs space, action space,
    modifies step and reset fn, just returns ob dict now. ob dict has reward, done, etc.
    """

    def __init__(self, env, if_eval=False, reset_with_seed=False, obs_key='observation', act_key='action', info_to_obs_fn=None, if_reduce_obs_dim=False):
        self._env = env

        self.if_eval = if_eval

        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')

        self._act_is_dict = hasattr(self._env.action_space, 'spaces')

        self._obs_key = obs_key
        self._act_key = act_key
        self.info_to_obs_fn = info_to_obs_fn

        self.if_reduce_obs_dim = if_reduce_obs_dim

        if self._env.spec.id == "PegInsertionSide-v0":

            self.goal_idx = 0

            if self.if_eval:

                self.seed_list = [0, 1, 2, 3, 5]

            else:

                self.seed_list = [0, 1, 2]

            self.seed_goal_dict = {
                0: np.array([1.81830227e-01, 5.74905396e-01, 1.37607260e-02, -1.85442078e+00, -1.62137486e-02, 2.42610025e+00, -6.88649297e-01, 2.20767539e-02, 2.20938660e-02, 7.43394066e-03, 1.04621425e-02, -9.62241646e-03, 2.08515078e-02, -6.22122642e-03, -1.18274558e-02, 2.59210151e-02, 6.11230716e-05, -3.63434170e-04, -6.15000010e-01, 8.73114914e-11, 0.00000000e+00, 1.00000012e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.47418911e-02, 1.23868600e-01, 1.05964966e-01, 3.19498242e-04, 6.67575121e-01, 7.44539142e-01, -2.19405768e-03, 9.45173297e-03, 1.73299462e-01, 1.05741017e-01, 6.68531239e-01, -8.95643374e-04, 2.01492128e-03, 7.43680894e-01, 1.02440678e-01, 2.21518930e-02, 2.21518930e-02, -3.70832253e-03, 2.83787638e-01, 1.06044292e-01, 6.65456831e-01, 0.00000000e+00, 0.00000000e+00, 7.46436357e-01, 2.51518935e-02]),
                1: np.array([1.3675854e-01, 4.9240422e-01, -1.2641929e-02, -2.0844460e+00, 6.8590986e-03, 2.5705941e+00, -3.4268767e-01, 2.2614639e-02, 2.1799462e-02, 1.0346685e-02, 8.1889814e-04, -1.1495682e-02, 1.0227492e-02, -1.1914693e-02, -3.2729562e-02, 2.6383577e-02, -7.8096142e-04, 5.2873651e-04, -6.1500001e-01, 8.7311491e-11, 0.0000000e+00, 1.0000001e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, -3.5463355e-02, 7.1944185e-02, 8.1312612e-02, -4.8786675e-04, 8.1157136e-01, 5.8424395e-01, -3.2708745e-03, -1.9536324e-02, 1.1992729e-01, 8.0993935e-02, 8.0845916e-01, 6.6424982e-04, 2.3691487e-03, 5.8854723e-01, 9.5851101e-02, 2.2203244e-02, 2.2203244e-02, 1.5177796e-02, 2.1776408e-01, 8.1293315e-02, 8.1078178e-01, 0.0000000e+00, 0.0000000e+00, 5.8534855e-01, 2.5203245e-02]),
                2: np.array([2.81332642e-01, 5.03252506e-01, -6.06334396e-02, -2.03278255e+00, 4.66373190e-02, 2.53016090e+00, -4.56216127e-01, 1.52025046e-02, 1.52098341e-02, 8.40214454e-03, 7.02408236e-03, -1.22000305e-02, 1.82699375e-02, -9.74855851e-03, -2.87350453e-02, 2.92915031e-02, 5.66016279e-05, -2.57728156e-04, -6.15000010e-01, 8.73114914e-11, 0.00000000e+00, 1.00000012e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -3.13415080e-02, 1.30619466e-01, 9.08941776e-02, 1.40967895e-05, 7.54601419e-01, 6.56178236e-01, -2.60222098e-03, -2.43821014e-02, 1.79845423e-01, 9.09571275e-02, 7.55040228e-01, -2.53557140e-04, 2.78897234e-03, 6.55672550e-01, 9.67997462e-02, 1.52592622e-02, 1.52592622e-02, -1.04813632e-02, 2.84611583e-01, 9.15259048e-02, 7.52615690e-01, 0.00000000e+00, 0.00000000e+00, 6.58460081e-01, 1.82592627e-02]),
                3: np.array([2.55297631e-01, 7.93555558e-01, 9.02452767e-02, -1.47290015e+00, -9.07772779e-02, 2.26013446e+00, -7.13582456e-01, 2.20035352e-02, 2.20212806e-02, 6.29775983e-04, 1.32671194e-02, 1.93476793e-04, 2.40237936e-02, -2.23881943e-04, -1.07875653e-02, 1.64373574e-04, 5.61112865e-05, -3.66784952e-04, -6.15000010e-01, 8.73114914e-11, 0.00000000e+00, 1.00000012e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.54250532e-02, 2.29851559e-01, 1.03733934e-01, 1.67685095e-03, 5.91544390e-01, 8.06266665e-01, -2.57765595e-03, 5.04159778e-02, 2.77883917e-01, 1.03277348e-01, 5.92378557e-01, -1.90370262e-03, 2.00106087e-03, 8.05655122e-01, 1.02539897e-01, 2.20814776e-02, 2.20814776e-02, 1.70979034e-02, 3.83741528e-01, 1.03411071e-01, 5.88962317e-01, 0.00000000e+00, 0.00000000e+00, 8.08160484e-01, 2.50814781e-02]),
                5: np.array([1.8329586e-01, 7.3086745e-01, 4.8267577e-02, -1.5576547e+00, -4.7488637e-02, 2.2851338e+00, -6.1979264e-01, 2.3610193e-02, 2.3638830e-02, 1.2327368e-02, 8.4310593e-03, -1.5619789e-02, 1.9815821e-02, -1.0325901e-02, -1.4378412e-02, 3.9492421e-02, 1.1098106e-04, -4.5170545e-04, -6.1500001e-01, 8.7311491e-11, 0.0000000e+00, 1.0000001e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 7.1043819e-02, 1.5521863e-01, 1.1257899e-01, 8.5883803e-04, 6.7648107e-01, 7.3645699e-01, -1.9695798e-03, 6.6846304e-02, 2.0562673e-01, 1.1203165e-01, 6.7720157e-01, -9.9172920e-04, 1.4944036e-03, 7.3579538e-01, 8.6099662e-02, 2.3707323e-02, 2.3707323e-02, 6.0089141e-02, 2.9928610e-01, 1.1221778e-01, 6.7540699e-01, 0.0000000e+00, 0.0000000e+00, 7.3744518e-01, 2.6707323e-02]),

            }

        if self._env.spec.id == "FetchPickAndPlace-v2":

            self.goal_idx = 0

            if self.if_eval:

                self.seed_list = [0, 1, 2, 3, 4]

            else:

                self.seed_list = [0, 1, 2]

            self.seed_goal_dict = {
                0: np.array([1.41970542e+00, 8.58713940e-01, 4.25692184e-01, 1.42290880e+00,
                             8.59712457e-01, 4.25503559e-01, 3.20337890e-03, 9.98516344e-04,
                             -1.88625639e-04, 2.50007962e-02, 2.30927866e-02, -1.68589989e-05,
                             1.51058999e-04, 1.65247348e-02, 8.91698115e-06, -3.75938282e-04,
                             -3.20900307e-05, 3.64877420e-04, 1.56182631e-04, -1.73043773e-05,
                             -1.16412669e-03, -2.17728998e-05, 2.20063191e-04, -4.29866370e-04,
                             4.28264637e-04]),
                1: np.array([1.24969523e+00, 8.84893160e-01, 7.73442435e-01, 1.25443954e+00,
                             8.85124579e-01, 7.74897922e-01, 4.74430373e-03, 2.31418889e-04,
                             1.45548717e-03, 2.41623220e-02, 2.39298243e-02, -2.97764825e-03,
                             1.04457657e-02, 1.63529404e-02, 1.28110068e-04, -2.44709411e-05,
                             -1.17227862e-04, -1.00415366e-03, -3.34987608e-03, 2.84104838e-05,
                             1.08792535e-03, -3.40962817e-04, -1.00724613e-03, -3.22000623e-05,
                             2.88664790e-05]),
                2: np.array([1.37249595e+00, 8.02781095e-01, 5.40148305e-01, 1.38033407e+00,
                             8.04612363e-01, 5.33043431e-01, 7.83812895e-03, 1.83126803e-03,
                             -7.10487331e-03, 2.57210005e-02, 2.23782774e-02, 1.20945956e-04,
                             3.38310220e-04, 1.62204301e-02, 1.21434953e-05, -5.41986876e-04,
                             -3.06010241e-05, -4.07799602e-04, 1.31152072e-05, -2.18024179e-05,
                             -3.91622844e-04, -4.21697911e-04, -7.14321199e-04, -5.26876632e-04,
                             5.27416988e-04]),
                3: np.array([1.41757608e+00, 7.73619861e-01, 6.22936043e-01, 1.41726091e+00,
                             7.73645542e-01, 6.20799761e-01, -3.15168334e-04, 2.56813605e-05,
                             -2.13628157e-03, 2.40549177e-02, 2.40397780e-02, 1.16134112e-04,
                             -5.51358332e-04, 1.62819996e-02, 5.06949905e-06, -5.07842104e-04,
                             -3.88607425e-05, -4.32037042e-04, -4.16345801e-06, 1.37342088e-05,
                             -2.38890178e-04, 5.47949582e-04, -1.20571991e-03, -4.99312967e-04,
                             4.99392061e-04]),
                4: np.array([1.44533107e+00, 6.30516562e-01, 7.68403964e-01, 1.45438591e+00,
                             6.30715710e-01, 7.68201408e-01, 9.05483681e-03, 1.99148230e-04,
                             -2.02556207e-04, 2.40320199e-02, 2.40621158e-02, -4.50319805e-03,
                             -2.01733083e-02, 1.63053742e-02, -2.68370345e-04, -4.90495122e-06,
                             -4.44670007e-04, 1.84149557e-03, 6.08004425e-03, 5.21478308e-05,
                             -2.71752311e-03, 6.21470704e-04, -6.91159582e-04, -1.42640256e-05,
                             1.81645120e-05]),

            }

        self.reset_with_seed = reset_with_seed
        self.goal_idx = 0
        self.reset_seed = self.seed_list[self.goal_idx]
        self.goal = self.seed_goal_dict[self.reset_seed]

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}

        if self.if_reduce_obs_dim:
            spaces[self._obs_key] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

        obs_space = {
            **spaces,
            'goal': spaces[self._obs_key],
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

        del obs_space['achieved_goal']
        del obs_space['desired_goal']

        return obs_space

    @property
    def act_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()
        else:
            return {self._act_key: self._env.action_space}

    def step(self, action):
        if not self._act_is_dict:
            action = action[self._act_key]
        obs, reward, terminated, truncated, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}

        obs['goal_idx'] = self.goal_idx
        obs['goal'] = self.goal
        obs['reward'] = float(reward)
        obs['is_first'] = False
        obs['is_last'] = truncated
        obs['is_terminal'] = terminated

        if self._env.spec.id == "PegInsertionSide-v0":
            obs['env_states'] = self._env.get_state()
        if self.info_to_obs_fn:
            obs = self.info_to_obs_fn(info, obs)

        if self.if_reduce_obs_dim:
            obs[self._obs_key] = self.reduce_obs(obs[self._obs_key])
            obs['goal'] = self.reduce_obs(obs['goal'])

        return obs

    def reset(self):
        if self.reset_with_seed:
            obs, info = self._env.reset(seed=self.reset_seed)
        else:
            goal_idx = np.random.choice(len(self.seed_list))
            self.set_goal_idx(goal_idx)
            obs, info = self._env.reset(seed=self.reset_seed)

        if not self._obs_is_dict:
            obs = {self._obs_key: obs}

        obs['goal_idx'] = self.goal_idx
        obs['goal'] = self.goal
        obs['reward'] = 0.0
        obs['is_first'] = True
        obs['is_last'] = False
        obs['is_terminal'] = False

        if self._env.spec.id == "PegInsertionSide-v0":
            obs['env_states'] = self._env.get_state()

        if self.info_to_obs_fn:
            obs = self.info_to_obs_fn(None, obs)

        if self.if_reduce_obs_dim:
            obs[self._obs_key] = self.reduce_obs(obs[self._obs_key])
            obs['goal'] = self.reduce_obs(obs['goal'])

        return obs

    def reduce_obs(self, obs):

        if self._env.spec.id == "PegInsertionSide-v0":

            obs = obs[25:42]

            return obs
        else:
            raise NotImplementedError

    def set_goal_idx(self, goal_idx):

        self.goal_idx = goal_idx
        self.reset_seed = self.seed_list[self.goal_idx]
        self.goal = self.seed_goal_dict[self.reset_seed]

    def get_seed_from_goal(self, goal):

        for key, value in self.seed_goal_dict.items():

            if np.all(value == goal):

                return key

        raise ValueError("Goal not found in goal_dict")


class GymnasiumWrapper_0:
    """modifies obs space, action space,
    modifies step and reset fn, just returns ob dict now. ob dict has reward, done, etc.
    """

    def __init__(self, env, if_eval=False, reset_with_seed=False, obs_key='observation', act_key='action', info_to_obs_fn=None, if_reduce_obs_dim=False):
        self._env = env

        self.if_eval = if_eval

        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')

        self._act_is_dict = hasattr(self._env.action_space, 'spaces')

        self._obs_key = obs_key
        self._act_key = act_key
        self.info_to_obs_fn = info_to_obs_fn

        self.if_reduce_obs_dim = if_reduce_obs_dim

        if self._env.spec.id == "FetchPickAndPlace-v2" or self._env.spec.id == "PegInsertionSide-v0" or self._env.spec.id == 'FetchPush-v2':

            self.goal_idx = 0

            if self.if_eval:

                self.seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            else:

                self.seed_list = [0, 1, 2, 3, 4,]

        self.reset_with_seed = reset_with_seed
        self.goal_idx = 0
        self.reset_seed = self.seed_list[self.goal_idx]

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}

        if self.if_reduce_obs_dim:
            spaces[self._obs_key] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

        obs_space = {
            **spaces,
            'goal': spaces['desired_goal'],
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

        del obs_space['achieved_goal']
        del obs_space['desired_goal']

        return obs_space

    @property
    def act_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()
        else:
            return {self._act_key: self._env.action_space}

    def step(self, action):
        if not self._act_is_dict:
            action = action[self._act_key]
        obs, reward, terminated, truncated, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}

        obs['goal_idx'] = self.goal_idx
        obs['goal'] = obs['desired_goal']
        obs['reward'] = float(reward)
        obs['is_first'] = False
        obs['is_last'] = truncated
        obs['is_terminal'] = info['is_success']

        if self._env.spec.id == "PegInsertionSide-v0":
            obs['env_states'] = self._env.get_state()

        if self.info_to_obs_fn:
            obs = self.info_to_obs_fn(info, obs)

        if self.if_reduce_obs_dim:
            obs[self._obs_key] = self.reduce_obs(obs[self._obs_key])

        return obs

    def reset(self):
        if self.reset_with_seed:
            obs, info = self._env.reset(seed=self.reset_seed)
        else:

            goal_idx = np.random.choice(len(self.seed_list))
            self.set_goal_idx(goal_idx)
            obs, info = self._env.reset(seed=self.reset_seed)

        if not self._obs_is_dict:
            obs = {self._obs_key: obs}

        obs['goal_idx'] = self.goal_idx
        obs['goal'] = obs['desired_goal']
        obs['reward'] = 0.0
        obs['is_first'] = True
        obs['is_last'] = False
        obs['is_terminal'] = False

        if self._env.spec.id == "PegInsertionSide-v0":
            obs['env_states'] = self._env.get_state()

        if self.info_to_obs_fn:
            obs = self.info_to_obs_fn(None, obs)

        if self.if_reduce_obs_dim:
            obs[self._obs_key] = self.reduce_obs(obs[self._obs_key])

        return obs

    def reduce_obs(self, obs):

        if self._env.spec.id == "PegInsertionSide-v0":

            obs = obs[25:42]

            return obs
        else:
            raise NotImplementedError

    def set_goal_idx(self, goal_idx):

        self.goal_idx = goal_idx
        self.reset_seed = self.seed_list[self.goal_idx]

    def render(self, width=200, height=200):

        image = self._env.render()

        compressed_image = Image.fromarray(
            image.astype('uint8')).resize((width, height))

        return compressed_image


class GymnasiumWrapper_1:
    """modifies obs space, action space,
    modifies step and reset fn, just returns ob dict now. ob dict has reward, done, etc.
    """

    def __init__(self, env, if_eval=False, obs_key='observation', act_key='action'):
        self._env = env

        self.if_eval = if_eval

        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
        self._act_is_dict = hasattr(self._env.action_space, 'spaces')

        self._obs_key = obs_key
        self._act_key = act_key

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}

        obs_space = {
            **spaces,
            'goal': gym.spaces.Box(-np.inf, np.inf, (7, ), dtype=np.float32) if self._env.spec.id == "PegInsertionSide-v0" else spaces['desired_goal'],
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

        if self._env.spec.id == "PegInsertionSide-v0":

            obs_space['goal'] = gym.spaces.Box(-np.inf,
                                               np.inf, (7, ), dtype=np.float32)

        elif self._env.spec.id == 'HandManipulateBlockRotateXYZ-v1' or self._env.spec.id == 'HandManipulatePenRotate-v1' or self._env.spec.id == 'HandManipulateEggRotate-v1':

            obs_space['goal'] = gym.spaces.Box(-np.inf,
                                               np.inf, (4, ), dtype=np.float32)

        else:

            obs_space['goal'] = spaces['desired_goal']

        if self._env.spec.id != "PegInsertionSide-v0":

            del obs_space['achieved_goal']
            del obs_space['desired_goal']

        return obs_space

    @property
    def act_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()
        else:
            return {self._act_key: self._env.action_space}

    def step(self, action):
        if not self._act_is_dict:
            action = action[self._act_key]
        obs, reward, terminated, truncated, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}

        if self._env.spec.id == "PegInsertionSide-v0":

            obs['goal'] = self.goal
            obs['env_states'] = self._env.get_state()

        elif self._env.spec.id == 'HandManipulateBlockRotateXYZ-v1' or self._env.spec.id == 'HandManipulatePenRotate-v1' or self._env.spec.id == 'HandManipulateEggRotate-v1':

            obs['goal'] = obs['desired_goal'][-4:]

        else:
            obs['goal'] = obs['desired_goal']

        obs['reward'] = float(reward)
        obs['is_first'] = False
        obs['is_last'] = truncated

        if self._env.spec.id == "PegInsertionSide-v0":
            obs['is_terminal'] = info['success']
        else:
            obs['is_terminal'] = info['is_success']

        return obs

    def reset(self):

        obs, info = self._env.reset()

        if not self._obs_is_dict:
            obs = {self._obs_key: obs}

        if self._env.spec.id == "PegInsertionSide-v0":

            self.goal = vectorize_pose(self._env.box_hole_pose)
            obs['goal'] = self.goal
            obs['env_states'] = self._env.get_state()

        elif self._env.spec.id == 'HandManipulateBlockRotateXYZ-v1' or self._env.spec.id == 'HandManipulatePenRotate-v1' or self._env.spec.id == 'HandManipulateEggRotate-v1':

            obs['goal'] = obs['desired_goal'][-4:]

        else:
            obs['goal'] = obs['desired_goal']

        obs['reward'] = 0.0
        obs['is_first'] = True
        obs['is_last'] = False
        obs['is_terminal'] = False

        return obs

    def render(self, width=200, height=200):

        image = self._env.render()

        compressed_image = Image.fromarray(
            image.astype('uint8')).resize((width, height))

        return compressed_image

    def render_with_obs(self, obs, goal, width=200, height=200):

        if self._env.spec.id == "FetchPickAndPlace-v2" or self._env.spec.id == 'FetchPush-v2':

            inner_env = self._env.env.env.env

            data = inner_env.data
            model = inner_env.model

            object_pos = obs[3:6]
            gripper_target = obs[:3]

            gripper_right_finger = obs[9]
            gripper_left_finger = obs[10]

            inner_env.goal = goal

            inner_env._utils.set_mocap_pos(
                model, data, "robot0:mocap", gripper_target)

            for _ in range(10):
                inner_env._mujoco.mj_step(
                    model, data, nstep=inner_env.n_substeps)

            inner_env._utils.set_joint_qpos(
                model, data, "robot0:r_gripper_finger_joint", gripper_right_finger)
            inner_env._utils.set_joint_qpos(
                model, data, "robot0:l_gripper_finger_joint", gripper_left_finger)

            object_qpos = inner_env._utils.get_joint_qpos(
                model, data, "object0:joint"
            )

            assert object_qpos.shape == (7,)

            object_qpos[:3] = object_pos

            inner_env._utils.set_joint_qpos(
                model, data, "object0:joint", object_qpos)

            inner_env._mujoco.mj_forward(model, data)

            image = self.render(width, height)

        elif "HandManipulateBlock" in self._env.spec.id or "HandManipulatePen" in self._env.spec.id or "HandManipulateEgg" in self._env.spec.id:

            if self._env.spec.id == 'HandManipulateBlockRotateXYZ-v1' or self._env.spec.id == 'HandManipulatePenRotate-v1' or self._env.spec.id == 'HandManipulateEggRotate-v1':

                goal = goal[-4:]
                goal = np.concatenate((np.array([1, 0.87, 0.17]), goal))

            inner_env = self._env.env.env.env

            data = inner_env.data
            model = inner_env.model

            block_qpos = obs[-7:]
            inner_env._utils.set_joint_qpos(
                model, data, "object:joint", block_qpos)

            hand_block_target_qpos = np.concatenate(
                (obs[:24], block_qpos, goal))

            inner_env.goal = goal

            data.qpos[:] = np.copy(hand_block_target_qpos)

            if model.na != 0:
                data.act[:] = None

            inner_env._mujoco.mj_forward(model, data)

            image = self.render(width, height)

            return image

        else:

            raise NotImplementedError

        return image


class DMC:

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):

        os.environ['MUJOCO_GL'] = 'egl'

        domain, task = name.split('_', 1)

        if domain == 'cup':
            domain = 'ball_in_cup'

        if domain == 'manip':
            from dm_control import manipulation
            self._env = manipulation.load(task + '_vision')

        elif domain == 'locom':
            from dm_control.locomotion.examples import basic_rodent_2020
            self._env = getattr(basic_rodent_2020, task)()

        else:
            from dm_control import suite
            self._env = suite.load(domain, task)

        self._action_repeat = action_repeat

        self._size = size

        if camera in (-1, None):
            camera = dict(
                quadruped_walk=2, quadruped_run=2, quadruped_escape=2,
                quadruped_fetch=2, locom_rodent_maze_forage=1,
                locom_rodent_two_touch=1,
            ).get(name, 0)
        self._camera = camera

        self._ignored_keys = []
        for key, value in self._env.observation_spec().items():
            if value.shape == (0,):
                print(f"Ignoring empty observation key '{key}'.")
                self._ignored_keys.append(key)

    @property
    def obs_space(self):

        spaces = {
            'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

        for key, value in self._env.observation_spec().items():

            if key in self._ignored_keys:
                continue

            if value.dtype == np.float64:
                spaces[key] = gym.spaces.Box(-np.inf,
                                             np.inf, value.shape, np.float32)

            elif value.dtype == np.uint8:
                spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)

            else:
                raise NotImplementedError(value.dtype)

        return spaces

    @property
    def act_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        return {'action': action}

    def step(self, action):

        assert np.isfinite(action['action']).all(), action['action']

        reward = 0.0

        for _ in range(self._action_repeat):
            time_step = self._env.step(action['action'])
            reward += time_step.reward or 0.0
            if time_step.last():
                break

        assert time_step.discount in (0, 1)

        obs = {
            'reward': reward,
            'is_first': False,
            'is_last': time_step.last(),
            'is_terminal': time_step.discount == 0,
            'image': self._env.physics.render(*self._size, camera_id=self._camera),
        }

        obs.update({
            k: v for k, v in dict(time_step.observation).items()
            if k not in self._ignored_keys})

        return obs

    def reset(self):
        time_step = self._env.reset()
        obs = {
            'reward': 0.0,
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
            'image': self._env.physics.render(*self._size, camera_id=self._camera),
        }
        obs.update({
            k: v for k, v in dict(time_step.observation).items()
            if k not in self._ignored_keys})
        return obs


class Atari:

    LOCK = threading.Lock()

    def __init__(
            self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
            life_done=False, sticky=True, all_actions=False):

        assert size[0] == size[1]

        import gym.wrappers
        import gym.envs.atari

        if name == 'james_bond':
            name = 'jamesbond'

        with self.LOCK:
            env = gym.envs.atari.AtariEnv(
                game=name, obs_type='image', frameskip=1,
                repeat_action_probability=0.25 if sticky else 0.0,
                full_action_space=all_actions)

        env._get_obs = lambda: None

        env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')

        self._env = gym.wrappers.AtariPreprocessing(
            env, noops, action_repeat, size[0], life_done, grayscale)

        self._size = size
        self._grayscale = grayscale

    @property
    def obs_space(self):
        shape = self._size + (1 if self._grayscale else 3,)
        return {
            'image': gym.spaces.Box(0, 255, shape, np.uint8),
            'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

    @property
    def act_space(self):
        return {'action': self._env.action_space}

    def step(self, action):
        image, reward, done, info = self._env.step(action['action'])
        if self._grayscale:
            image = image[..., None]
        return {
            'image': image,
            'ram': self._env.env._get_ram(),
            'reward': reward,
            'is_first': False,
            'is_last': done,
            'is_terminal': done,
        }

    def reset(self):
        with self.LOCK:
            image = self._env.reset()
        if self._grayscale:
            image = image[..., None]
        return {
            'image': image,
            'ram': self._env.env._get_ram(),
            'reward': 0.0,
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
        }

    def close(self):
        return self._env.close()


class Crafter:

    def __init__(self, outdir=None, reward=True, seed=None):
        import crafter
        self._env = crafter.Env(reward=reward, seed=seed)
        self._env = crafter.Recorder(
            self._env, outdir,
            save_stats=True,
            save_video=False,
            save_episode=False,
        )
        self._achievements = crafter.constants.achievements.copy()

    @property
    def obs_space(self):
        spaces = {
            'image': self._env.observation_space,
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'log_reward': gym.spaces.Box(-np.inf, np.inf, (), np.float32),
        }
        spaces.update({
            f'log_achievement_{k}': gym.spaces.Box(0, 2 ** 31 - 1, (), np.int32)
            for k in self._achievements})
        return spaces

    @property
    def act_space(self):
        return {'action': self._env.action_space}

    def step(self, action):
        image, reward, done, info = self._env.step(action['action'])
        obs = {
            'image': image,
            'reward': reward,
            'is_first': False,
            'is_last': done,
            'is_terminal': info['discount'] == 0,
            'log_reward': info['reward'],
        }
        obs.update({
            f'log_achievement_{k}': v
            for k, v in info['achievements'].items()})
        return obs

    def reset(self):
        obs = {
            'image': self._env.reset(),
            'reward': 0.0,
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
            'log_reward': 0.0,
        }
        obs.update({
            f'log_achievement_{k}': 0
            for k in self._achievements})
        return obs


class Dummy:

    def __init__(self):
        pass

    @property
    def obs_space(self):
        return {
            'image': gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

    @property
    def act_space(self):
        return {'action': gym.spaces.Box(-1, 1, (6,), dtype=np.float32)}

    def step(self, action):
        return {
            'image': np.zeros((64, 64, 3)),
            'reward': 0.0,
            'is_first': False,
            'is_last': False,
            'is_terminal': False,
        }

    def reset(self):
        return {
            'image': np.zeros((64, 64, 3)),
            'reward': 0.0,
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
        }


class TimeLimit:

    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            obs['is_last'] = True
            self._step = None
        return obs

    def reset(self):
        self._step = 0
        return self._env.reset()


class NormalizeAction:

    def __init__(self, env, key='action'):
        self._env = env
        self._key = key
        space = env.act_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * \
            (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])

        return self._env.step({**action, self._key: orig})


class OneHotAction:

    def __init__(self, env, key='action'):
        assert hasattr(env.act_space[key], 'n')
        self._env = env
        self._key = key
        self._random = np.random.RandomState()

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        shape = (self._env.act_space[self._key].n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        space.n = shape[0]
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        index = np.argmax(action[self._key]).astype(int)
        reference = np.zeros_like(action[self._key])
        reference[index] = 1
        if not np.allclose(reference, action[self._key]):
            raise ValueError(f'Invalid one-hot action:\n{action}')
        return self._env.step({**action, self._key: index})

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.act_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class ResizeImage:

    def __init__(self, env, size=(64, 64)):
        self._env = env
        self._size = size

        self._keys = [
            k for k, v in env.obs_space.items()
            if len(v.shape) > 1 and v.shape[:2] != size]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
            from PIL import Image
            self._Image = Image

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        for key in self._keys:
            shape = self._size + spaces[key].shape[2:]
            spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def reset(self):
        obs = self._env.reset()
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image


class RenderImage:

    def __init__(self, env, key='image'):
        self._env = env
        self._key = key
        self._shape = self._env.render().shape

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        obs[self._key] = self._env.render('rgb_array')
        return obs

    def reset(self):
        obs = self._env.reset()
        obs[self._key] = self._env.render('rgb_array')
        return obs


class Async:

    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _CLOSE = 4
    _EXCEPTION = 5

    def __init__(self, constructor, strategy='thread'):

        self._pickled_ctor = cloudpickle.dumps(constructor)

        if strategy == 'process':
            import multiprocessing as mp
            context = mp.get_context('spawn')

        elif strategy == 'thread':
            import multiprocessing.dummy as context

        else:
            raise NotImplementedError(strategy)

        self._strategy = strategy

        self._conn, conn = context.Pipe()

        self._process = context.Process(target=self._worker, args=(conn,))

        atexit.register(self.close)
        self._process.start()

        self._receive()

        self._obs_space = None
        self._act_space = None

    def access(self, name):
        self._conn.send((self._ACCESS, name))

        return self._receive

    def call(self, name, *args, **kwargs):
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            pass
        self._process.join(5)

    @property
    def obs_space(self):
        if not self._obs_space:
            self._obs_space = self.access('obs_space')()
        return self._obs_space

    @property
    def act_space(self):
        if not self._act_space:
            self._act_space = self.access('act_space')()
        return self._act_space

    def step(self, action, blocking=False):
        promise = self.call('step', action)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking=False):
        promise = self.call('reset')
        if blocking:
            return promise()
        else:
            return promise

    def _receive(self):
        try:
            message, payload = self._conn.recv()
        except (OSError, EOFError):
            raise RuntimeError('Lost connection to environment worker.')

        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)

        if message == self._RESULT:
            return payload
        raise KeyError(
            'Received message of unexpected type {}'.format(message))

    def _worker(self, conn):
        try:
            ctor = cloudpickle.loads(self._pickled_ctor)
            env = ctor()

            conn.send((self._RESULT, None))

            while True:
                try:

                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break

                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue

                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue

                if message == self._CLOSE:
                    break

                raise KeyError(
                    'Received message of unknown type {}'.format(message))

        except Exception:
            stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
            print('Error in environment process: {}'.format(stacktrace))
            conn.send((self._EXCEPTION, stacktrace))

        finally:
            try:
                conn.close()
            except IOError:
                pass
