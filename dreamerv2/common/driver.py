import imp
import numpy as np

import itertools
import random
import copy
import tensorflow as tf
import h5py
import time


class Driver:

    def __init__(self, envs, **kwargs):
        self._envs = envs
        self._kwargs = kwargs
        self._on_steps = []
        self._on_resets = []
        self._on_episodes = []
        self._act_spaces = [env.act_space for env in envs]
        self.reset()

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_reset(self, callback):
        self._on_resets.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def reset(self):
        self._obs = [None] * len(self._envs)
        self._eps = [None] * len(self._envs)
        self._state = None

    def __call__(self, policy, steps=0, episodes=0):

        step, episode = 0, 0

        while step < steps or episode < episodes:

            obs = {i: self._envs[i].reset() for i, ob in enumerate(
                self._obs) if ob is None or ob['is_last']}

            for i, ob in obs.items():

                self._obs[i] = ob() if callable(ob) else ob

                act = {k: np.zeros(v.shape)
                       for k, v in self._act_spaces[i].items()}

                tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
                [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]

                self._eps[i] = [tran]

            obs = {k: np.stack([o[k] for o in self._obs])
                   for k in self._obs[0]}

            actions, self._state = policy(obs, self._state, **self._kwargs)

            actions = [{k: np.array(actions[k][i]) for k in actions}
                       for i in range(len(self._envs))]

            assert len(actions) == len(self._envs)

            obs = [e.step(a) for e, a in zip(self._envs, actions)]

            obs = [ob() if callable(ob) else ob for ob in obs]

            for i, (act, ob) in enumerate(zip(actions, obs)):

                tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}

                [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]
                self._eps[i].append(tran)

                step += 1
                if ob['is_last']:
                    ep = self._eps[i]
                    ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
                    [fn(ep, **self._kwargs) for fn in self._on_episodes]
                    episode += 1

            self._obs = obs

    def _convert(self, value):

        value = np.array(value)

        if np.issubdtype(value.dtype, np.floating):
            return value.astype(np.float32)

        elif np.issubdtype(value.dtype, np.signedinteger):
            return value.astype(np.int32)

        elif np.issubdtype(value.dtype, np.uint8):
            return value.astype(np.uint8)

        return value


class GCDriver(Driver):
    def __init__(self, envs, goal_key, config, **kwargs):
        super().__init__(envs, **kwargs)
        self.config = config
        self.goal_key = goal_key
        self.all_transfer_goals = None

        self.all_3_block_train_goals_index = [10, 11, 14]

        self.if_eval_driver = False

        self.if_set_initial_state = False
        self.set_state_fn = None
        self.initial_state = None

    def reset(self):
        super().reset()
        self._subgoals = [None] * len(self._envs)
        self._use_policy_2 = [False] * len(self._envs)
        self._goal_time = [0] * len(self._envs)
        self._goal_dist = [0] * len(self._envs)
        self._goal_success = [0] * len(self._envs)

    def __call__(self, policy_1,
                 policy_2=None,
                 get_goal=None,
                 steps=0,
                 episodes=0,
                 goal_time_limit=None,
                 goal_checker=None,
                 if_multi_3_blcok_training_goal=False,
                 if_use_demo=False,
                 demo_path=None,
                 label='Normal'):
        """
        1. train: run gcp for entire rollout using goals from buffer/search.
        2. expl: run plan2expl for entire rollout
        3. 2pol: run gcp with goals from buffer/search and then expl policy

        LEXA is (1,2) and choosing goals from buffer.
        Ours can be (1,2,3), or (1,3) and choosing goals from search

        Args:
                policy_1 (_type_): 1st policy to run in episode
                policy_2 (_type_, optional): 2nd policy that runs after first policy is done. If None, then only run 1st policy.
                goal_strategy (_type_, optional): How to sample a goal
                steps (int, optional): _description_. Defaults to 0.
                episodes (int, optional): _description_. Defaults to 0.
                goal_time_limit (_type_, optional): _description_. Defaults to None.
                goal_checker (_type_, optional): _description_. Defaults to None.
        """

        step, episode = 0, 0
        while step < steps or episode < episodes:

            if if_use_demo and demo_path:

                demo_trajectory = sample_one_demo_trajectory(demo_path)
                self._eps[0] = []

                for step in range(len(demo_trajectory['observation'])):

                    tran = {k: self._convert(v[step])
                            for k, v in demo_trajectory.items()}

                    tran["label"] = label

                    [fn(tran, worker=0, **self._kwargs)
                        for fn in self._on_steps]

                    self._eps[0].append(tran)

                    if demo_trajectory['is_last'][step]:
                        ep = self._eps[0]
                        ep = {k: self._convert([t[k] for t in ep])
                              for k in ep[0]}
                        [fn(ep, **self._kwargs) for fn in self._on_episodes]
                        episode += 1

                        break

                continue

            obs = {}
            for i, ob in enumerate(self._obs):

                if ob is None or ob['is_last']:

                    if if_multi_3_blcok_training_goal:

                        self.training_goal_index = random.randint(1, 3)
                        training_env_goal_index = self.all_3_block_train_goals_index[
                            self.training_goal_index-1]

                        label = 'egc' + str(self.training_goal_index)

                        self._envs[i].set_goal_idx(training_env_goal_index)

                    obs[i] = self._envs[i].reset()

                    if self.if_eval_driver and self.if_set_initial_state:

                        self.set_state_fn(self._envs[i], self.initial_state)
                        obs[i] = self._envs[i].step(
                            {'action': np.zeros(self._act_spaces[i]['action'].shape)})

                        self.if_set_initial_state = False

            for i, ob in obs.items():

                self._obs[i] = ob() if callable(ob) else ob

                act = {k: np.zeros(v.shape)
                       for k, v in self._act_spaces[i].items()}

                tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}

                self._use_policy_2[i] = False
                self._goal_time[i] = 0

                if get_goal:
                    self._subgoals[i] = subgoal = get_goal(
                        obs, self._state, **self._kwargs)
                    tran[self.goal_key] = subgoal.numpy()

                tran["label"] = label

                [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]

                if goal_checker is not None:

                    self._goal_dist[i] = 0
                    self._goal_success[i] = 0.0

                self._eps[i] = [tran]

            obs = {}

            for k in self._obs[0]:
                if k == self.goal_key:
                    goals = [g if g is not None and get_goal else self._obs[i][k]
                             for (i, g) in enumerate(self._subgoals)]
                    obs[k] = np.stack(goals)
                else:
                    obs[k] = np.stack([o[k] for o in self._obs])

            policy = policy_2 if self._use_policy_2[0] else policy_1

            try:
                actions, self._state = policy(obs, self._state, **self._kwargs)
            except:
                print(policy, self._use_policy_2[0], policy_2)

            actions = [{k: np.array(actions[k][i]) for k in actions}
                       for i in range(len(self._envs))]

            assert len(actions) == len(self._envs)

            obs = [e.step(a) for e, a in zip(self._envs, actions)]
            obs = [ob() if callable(ob) else ob for ob in obs]

            if get_goal:
                for o in obs:
                    o[self.goal_key] = subgoal.numpy()

            for i, ob in enumerate(obs):

                if policy_2 is None or self._use_policy_2[i]:
                    continue

                self._goal_time[i] += 1
                subgoal = self._subgoals[i]
                out_of_time = goal_time_limit and self._goal_time[i] > goal_time_limit

                if self.config.if_actor_gs:

                    close_to_goal, goal_info = False, {}

                else:
                    close_to_goal, goal_info = goal_checker(obs)
                    self._goal_dist[i] += goal_info["subgoal_dist"]
                    self._goal_success[i] += goal_info["subgoal_success"]

                if out_of_time or close_to_goal:
                    self._use_policy_2[i] = True

            for i, (act, ob) in enumerate(zip(actions, obs)):

                tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}

                tran["label"] = label

                [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]
                self._eps[i].append(tran)
                step += 1

                if ob['is_last']:
                    ep = self._eps[i]
                    ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}

                    ep["log_subgoal_dist"] = np.array([self._goal_dist[i]])
                    ep["log_subgoal_success"] = np.array(
                        [float(self._goal_success[i] > 0)])
                    ep["log_subgoal_time"] = np.array([self._goal_time[i]])

                    [fn(ep, **self._kwargs) for fn in self._on_episodes]
                    episode += 1

            self._obs = obs


def sample_one_demo_trajectory(demo_path):

    with h5py.File(demo_path, 'r') as f:

        all_demo_list = list(f.keys())
        trajectory = random.choice(all_demo_list)

        trajectory = f[trajectory]

        tra_dict = {}

        for key in trajectory.keys():

            if key == 'actions':

                key_data = np.array(trajectory[key])

                initial_action = np.zeros((1, key_data.shape[1]))

                key_data = np.concatenate((initial_action, key_data), axis=0)

            elif key == 'success':

                key_data = np.array(trajectory[key])

                initial_action = np.array([0])

                key_data = np.concatenate((initial_action, key_data), axis=0)

            else:
                key_data = np.array(trajectory[key])

            tra_dict[key] = key_data

        tra_dict['observation'] = tra_dict['obs']
        tra_dict['action'] = tra_dict['actions']

        tra_dict['goal'] = np.repeat(
            tra_dict['obs'][-1].reshape(1, tra_dict['obs'].shape[1]), tra_dict['obs'].shape[0], axis=0)

        tra_dict['reward'] = np.zeros(
            tra_dict['success'].shape, dtype=np.float32)

        tra_dict['is_first'] = np.zeros_like(tra_dict['success'], dtype=bool)
        tra_dict['is_first'][0] = True

        tra_dict['is_last'] = np.zeros_like(tra_dict['success'], dtype=bool)
        tra_dict['is_last'][-1] = True

        tra_dict['is_terminal'] = tra_dict['success']

        del tra_dict['obs']
        del tra_dict['actions']
        del tra_dict['success']

        return tra_dict
