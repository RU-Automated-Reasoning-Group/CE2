import os

import tensorflow as tf
from dreamerv2 import common

import envs
import random
import numpy as np
import collections
import matplotlib.pyplot as plt
from dreamerv2.common.replay import convert
import pathlib
import sys
import ruamel.yaml as yaml
import collections
import logging
import os
import pathlib
import re
import sys
import warnings
import signal
import tensorflow as tf
from time import time
import matplotlib.pyplot as plt
from functools import partial
import pickle
from collections import defaultdict
from time import time
from tqdm import tqdm
import imageio
import numpy as np
import ruamel.yaml as yaml
import time

import gc_agent
import common
import dreamerv2.gc_goal_picker as gc_goal_picker


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))


class CE2:

    def __init__(self):
        self.cluster_ep_idx = 0
        pass

    def Set_Config(self,):

        configs = yaml.safe_load(
            (pathlib.Path(sys.argv[0]).parent.parent / 'Config/configs.yaml').read_text())

        parsed, remaining = common.Flags(
            configs=['defaults']).parse(known_only=True)

        config = common.Config(configs['defaults'])

        for name in parsed.configs:
            config = config.update(configs[name])
        config = old_config = common.Flags(config).parse(remaining)

        logdir = pathlib.Path(config.logdir).expanduser()
        if logdir.exists():
            print('Loading existing config')
            yaml_config = yaml.safe_load((logdir / 'config.yaml').read_text())
            new_keys = []
            for key in new_keys:
                if key not in yaml_config:
                    print(
                        f"{key} does not exist in saved config file, using default value from default config file")
                    yaml_config[key] = old_config[key]
            config = common.Config(yaml_config)
            config = common.Flags(config).parse(remaining)
            config.save(logdir / 'config.yaml')

        else:
            print('Creating new config')
            logdir.mkdir(parents=True, exist_ok=True)
            config.save(logdir / 'config.yaml')

        return config

    def make_env(self, config, if_eval=False):

        def wrap_mega_env(e, info_to_obs_fn=None):
            e = common.GymWrapper(e, info_to_obs_fn=info_to_obs_fn)
            if hasattr(e.act_space['action'], 'n'):
                e = common.OneHotAction(e)
            else:
                e = common.NormalizeAction(e)
            return e

        def wrap_lexa_env(e):
            e = common.GymWrapper(e)
            if hasattr(e.act_space['action'], 'n'):
                e = common.OneHotAction(e)
            else:
                e = common.NormalizeAction(e)

            if if_eval:
                e = common.TimeLimit(e, 300)
            else:
                e = common.TimeLimit(e, config.time_limit)
            return e

        if config.task in {'discwallsdemofetchpnp', 'wallsdemofetchpnp2', 'wallsdemofetchpnp3', 'demofetchpnp'}:

            from envs.customfetch.custom_fetch import DemoStackEnv, WallsDemoStackEnv, DiscreteWallsDemoStackEnv

            if 'walls' in config.task:
                if 'disc' in config.task:
                    env = DiscreteWallsDemoStackEnv(
                        max_step=config.time_limit, eval=if_eval, increment=0.01)
                else:
                    n = int(config.task[-1])
                    env = WallsDemoStackEnv(
                        max_step=config.time_limit, eval=if_eval, n=int(config.task[-1]))
            else:
                env = DemoStackEnv(max_step=config.time_limit, eval=if_eval)

            env = common.ConvertGoalEnvWrapper(env)

            info_to_obs = None

            def info_to_obs(info, obs):
                if info is None:
                    info = env.get_metrics_dict()
                obs = obs.copy()
                for k, v in info.items():
                    if eval:
                        if "metric" in k:
                            obs[k] = v
                    else:
                        if "above" in k:
                            obs[k] = v
                return obs

            class ClipObsWrapper:
                def __init__(self, env, obs_min, obs_max):
                    self._env = env
                    self.obs_min = obs_min
                    self.obs_max = obs_max

                def __getattr__(self, name):
                    return getattr(self._env, name)

                def step(self, action):
                    obs, rew, done, info = self._env.step(action)
                    new_obs = np.clip(obs['observation'],
                                      self.obs_min, self.obs_max)
                    obs['observation'] = new_obs
                    return obs, rew, done, info

            obs_min = np.ones(
                env.observation_space['observation'].shape) * -1e6
            pos_min = [1.0, 0.3, 0.35]
            if 'demofetchpnp' in config.task:
                obs_min[:3] = obs_min[5:8] = obs_min[8:11] = pos_min
                if env.n == 3:
                    obs_min[11:14] = pos_min

            obs_max = np.ones(
                env.observation_space['observation'].shape) * 1e6
            pos_max = [1.6, 1.2, 1.0]
            if 'demofetchpnp' in config.task:
                obs_max[:3] = obs_max[5:8] = obs_max[8:11] = pos_max
                if env.n == 3:
                    obs_max[11:14] = pos_max

            env = ClipObsWrapper(env, obs_min, obs_max)

            obs_min = np.concatenate(
                [env.workspace_min, [0., 0.], *[env.workspace_min for _ in range(env.n)]], 0)
            obs_max = np.concatenate(
                [env.workspace_max, [0.05, 0.05], *[env.workspace_max for _ in range(env.n)]], 0)

            env = common.NormObsWrapper(env, obs_min, obs_max)
            env = wrap_mega_env(env, info_to_obs)

        elif config.task in {'umazefulldownscale', 'hardumazefulldownscale'}:
            from envs.sibrivalry.ant_maze import AntMazeEnvFullDownscale, AntHardMazeEnvFullDownscale
            if 'hard' in config.task:
                env = AntHardMazeEnvFullDownscale(eval=if_eval)
            else:
                env = AntMazeEnvFullDownscale(eval=if_eval)
            env.max_steps = config.time_limit

            env = common.ConvertGoalEnvWrapper(env)
            info_to_obs = None
            if if_eval:
                def info_to_obs(info, obs):
                    if info is None:
                        info = env.get_metrics_dict()
                    obs = obs.copy()
                    for k, v in info.items():
                        if "metric" in k:
                            obs[k] = v
                    return obs
            env = wrap_mega_env(env, info_to_obs)

        elif 'pointmaze' in config.task:
            from envs.sibrivalry.toy_maze import MultiGoalPointMaze2D
            env = MultiGoalPointMaze2D(test=if_eval)
            env.max_steps = config.time_limit

            env = common.ConvertGoalEnvWrapper(env)

            info_to_obs = None
            if if_eval:
                def info_to_obs(info, obs):
                    if info is None:
                        info = env.get_metrics_dict()
                    obs = obs.copy()
                    for k, v in info.items():
                        if "metric" in k:
                            obs[k] = v
                    return obs
            env = wrap_mega_env(env, info_to_obs)

            class GaussianActions:
                """Add gaussian noise to the actions.
                """

                def __init__(self, env, std):
                    self._env = env
                    self.std = std

                def __getattr__(self, name):
                    return getattr(self._env, name)

                def step(self, action):
                    new_action = action
                    if self.std > 0:
                        noise = np.random.normal(scale=self.std, size=2)
                        if isinstance(action, dict):
                            new_action = {'action': action['action'] + noise}
                        else:
                            new_action = action + noise

                    return self._env.step(new_action)
            env = GaussianActions(env, std=0)

        elif 'dmc' in config.task:

            if if_eval:
                use_goal_idx = True
                log_per_goal = False

            else:
                use_goal_idx = False
                log_per_goal = True

            suite_task, obs = config.task.rsplit('_', 1)
            suite, task = suite_task.split('_', 1)
            if 'proprio' in config.task:
                env = envs.DmcStatesEnv(
                    task, config.render_size, config.action_repeat, use_goal_idx, log_per_goal)
                if 'humanoid' in config.task:
                    keys = ['qpos', 'goal']
                    env = common.NormObsWrapper(
                        env, env.obs_bounds[:, 0], env.obs_bounds[:, 1], keys)
            elif 'vision' in config.task:
                env = envs.DmcEnv(task, config.render_size,
                                  config.action_repeat, use_goal_idx, log_per_goal)

            env = wrap_lexa_env(env)

        elif config.task == "PegInsertionSide-v0" or config.task == "PickAndPlace" or config.task == "FetchPush-v2" or "HandManipulateBlock" in config.task or "HandManipulatePen" in config.task or "HandManipulateEgg" in config.task:

            import gymnasium as gym

            try:
                import mani_skill2.envs
            except:
                pass

            if config.task == "PegInsertionSide-v0":
                env = gym.make(config.task, obs_mode="state", control_mode="pd_joint_delta_pos",
                               render_mode="rgb_array", max_episode_steps=config.time_limit)

            elif config.task == "PickAndPlace":
                env = gym.make('FetchPickAndPlace-v2', render_mode="rgb_array",
                               max_episode_steps=config.time_limit)

            elif "HandManipulateBlock" in config.task or "HandManipulatePen" in config.task or "HandManipulateEgg" in config.task:
                env = gym.make(config.task,  render_mode="rgb_array",
                               max_episode_steps=config.time_limit)

            elif config.task == "FetchPush-v2":
                env = gym.make('FetchPush-v2',  render_mode="rgb_array",
                               max_episode_steps=config.time_limit)

            def gymnasium_env(e):

                if config.gc_input == 'state' or config.if_actor_gs:

                    if config.if_env_random_reset:

                        e = common.GymnasiumWrapper_1(e, if_eval=if_eval)

                    else:

                        e = common.GymnasiumWrapper_0(
                            e, if_eval=if_eval, reset_with_seed=if_eval)

                else:
                    e = common.GymnasiumWrapper(
                        e, if_eval=if_eval, reset_with_seed=if_eval)

                if hasattr(e.act_space['action'], 'n'):
                    e = common.OneHotAction(e)
                else:
                    e = common.NormalizeAction(e)

                return e

            env = gymnasium_env(env)

        else:
            raise NotImplementedError

        return env

    def make_images_render_fn(self, config):

        images_render_fn = None

        if 'demofetchpnp' in config.task:

            def images_render_fn(env, obs_list):

                sim = env.sim
                all_img = []
                env.reset()

                if env.n == 3:
                    out_of_way_state = np.array([4.40000000e+00,    4.04999349e-01,    4.79999636e-01,    2.79652104e-06, 1.56722299e-02, -3.41500342e+00, 9.11469058e-02, -1.27681180e+00,
                                                 -1.39750475e+00, 4.43858450e+00, 7.47892234e-01, 2.53633962e-01,
                                                 2.34366216e+00, 3.35102418e+00, 8.32919575e-04, 1.41610111e-03,
                                                 1.32999932e+00, 6.49999392e-01, 4.24784489e-01, 1.00000000e+00,
                                                 -2.28652597e-07, 2.56090909e-07, -1.20181003e-15, 1.32999955e+00,
                                                 8.49999274e-01, 4.24784489e-01, 1.00000000e+00, -2.77140579e-07,
                                                 1.72443027e-07, -1.77971404e-15, 1.39999939e+00, 7.49999392e-01,
                                                 4.24784489e-01, 1.00000000e+00, -2.31485576e-07, 2.31485577e-07,
                                                 -6.68816586e-16, -4.48284993e-08, -8.37398903e-09, 7.56100615e-07,
                                                 5.33433335e-03, 2.91848485e-01, 7.45623586e-05, 2.99902784e-01,
                                                 -7.15601860e-02, -9.44665089e-02, 1.49646097e-02, -1.10990294e-01,
                                                 -3.30174644e-03, 1.19462201e-01, 4.05130821e-04, -3.95036450e-04,
                                                 -1.53880539e-07, -1.37393338e-07, 1.07636483e-14, 5.51953825e-06,
                                                 -6.18188284e-06, 1.31307184e-17, -1.03617993e-07, -1.66528917e-07,
                                                 1.06089030e-14, 6.69000941e-06, -4.16267252e-06, 3.63225324e-17,
                                                 -1.39095626e-07, -1.39095626e-07, 1.10587840e-14, 5.58792469e-06,
                                                 -5.58792469e-06, -2.07082526e-17])

                sim.set_state_from_flattened(out_of_way_state)
                sim.forward()

                sites_offset = (sim.data.site_xpos - sim.model.site_pos)
                site_id = sim.model.site_name2id('gripper_site')

                def unnorm_ob(ob):
                    return env.obs_min + ob * (env.obs_max - env.obs_min)

                for obs in [*obs_list]:

                    obs = unnorm_ob(obs)
                    grip_pos = obs[:3]
                    gripper_state = obs[3:5]
                    all_obj_pos = np.split(obs[5:5+3*env.n], env.n)

                    sim.model.site_pos[site_id] = grip_pos - \
                        sites_offset[site_id]

                    for i, pos in enumerate(all_obj_pos):
                        sim.data.set_joint_qpos(f"object{i}:joint", [
                                                *pos, *[1, 0, 0, 0]])

                    sim.forward()
                    img = sim.render(height=200, width=200,
                                     camera_name="external_camera_0")[::-1]
                    all_img.append(img)

                all_imgs = np.stack(all_img[1:], 0)

                all_imgs = np.array(all_imgs)

                all_imgs = np.concatenate(all_imgs, axis=1)
                all_imgs = np.expand_dims(all_imgs, axis=0)

                return all_imgs

        elif config.task in {'umazefull', 'umazefulldownscale', 'hardumazefulldownscale'}:

            def images_render_fn(env, obs_list):

                ant_env = env.maze.wrapped_env

                inner_env = env._env._env._env

                all_img = []
                for obs in [*obs_list]:
                    inner_env.maze.wrapped_env.set_state(
                        obs[:15], np.zeros_like(obs[:14]))
                    inner_env.maze.wrapped_env.sim.forward()
                    img = env.render(mode='rgb_array')

                    all_img.append(img)

                all_imgs = np.stack(all_img[1:], 0)

                all_imgs = np.array(all_imgs)

                all_imgs = np.concatenate(all_imgs, axis=1)
                all_imgs = np.expand_dims(all_imgs, axis=0)

                return all_imgs

        elif 'pointmaze' in config.task:

            def images_render_fn(env, obs_list):

                all_img = []
                inner_env = env._env._env._env._env

                for xy in [*obs_list]:

                    inner_env.g_xy = xy
                    inner_env.s_xy = xy
                    img = env.render()
                    all_img.append(img)

                env.clear_plots()

                all_imgs = np.stack(all_img[1:], 0)

                all_imgs = np.array(all_imgs)

                all_imgs = np.concatenate(all_imgs, axis=1)
                all_imgs = np.expand_dims(all_imgs, axis=0)

                return all_imgs

        elif 'dmc_walker_walk_proprio' == config.task:

            def images_render_fn(env, obs_list):

                all_img = []
                inner_env = env._env._env._env._env
                for qpos in [*obs_list]:
                    size = inner_env.physics.get_state(
                    ).shape[0] - qpos.shape[0]
                    inner_env.physics.set_state(
                        np.concatenate((qpos, np.zeros([size]))))
                    inner_env.physics.step()
                    img = env.render()
                    all_img.append(img)

                all_imgs = np.stack(all_img[1:], 0)

                all_imgs = np.array(all_imgs)

                all_imgs = np.concatenate(all_imgs, axis=1)
                all_imgs = np.expand_dims(all_imgs, axis=0)

                return all_imgs

        elif config.task == "PickAndPlace" or config.task == "FetchPush-v2":

            def images_render_fn(env, obs_list):

                all_img = []

                for obs in [*obs_list]:

                    img = env.render_with_obs(
                        obs, obs[3:6], width=200, height=200)
                    all_img.append(img)

                all_imgs = np.stack(all_img[1:], 0)

                all_imgs = np.array(all_imgs)

                all_imgs = np.concatenate(all_imgs, axis=1)
                all_imgs = np.expand_dims(all_imgs, axis=0)

                return all_imgs

        elif "HandManipulateBlock" in config.task or "HandManipulatePen" in config.task or "HandManipulateEgg" in config.task:

            def images_render_fn(env, obs_list):

                all_img = []

                for obs in [*obs_list]:

                    if "HandManipulateBlockRotateXYZ" in config.task or "HandManipulatePenRotate" in config.task:
                        img = env.render_with_obs(
                            obs, obs[-4:], width=200, height=200)
                    else:
                        img = env.render_with_obs(
                            obs, obs[-7:], width=200, height=200)
                    all_img.append(img)

                all_imgs = np.stack(all_img[1:], 0)

                all_imgs = np.array(all_imgs)

                all_imgs = np.concatenate(all_imgs, axis=1)
                all_imgs = np.expand_dims(all_imgs, axis=0)

                return all_imgs

        return images_render_fn

    def make_ep_render_fn(self, config):

        episode_render_fn = None
        if config.no_render:
            return episode_render_fn

        if 'demofetchpnp' in config.task:
            import cv2

            def episode_render_fn_original(env, ep):
                sim = env.sim
                all_img = []

                env.reset()
                inner_env = env._env._env._env._env._env

                if env.n == 3:
                    out_of_way_state = np.array([4.40000000e+00,    4.04999349e-01,    4.79999636e-01,    2.79652104e-06,
                                                1.56722299e-02, -3.41500342e+00, 9.11469058e-02, -1.27681180e+00,
                                                -1.39750475e+00, 4.43858450e+00, 7.47892234e-01, 2.53633962e-01,
                                                2.34366216e+00, 3.35102418e+00, 8.32919575e-04, 1.41610111e-03,
                                                1.32999932e+00, 6.49999392e-01, 4.24784489e-01, 1.00000000e+00,
                                                -2.28652597e-07, 2.56090909e-07, -1.20181003e-15, 1.32999955e+00,
                                                8.49999274e-01, 4.24784489e-01, 1.00000000e+00, -2.77140579e-07,
                                                1.72443027e-07, -1.77971404e-15, 1.39999939e+00, 7.49999392e-01,
                                                4.24784489e-01, 1.00000000e+00, -2.31485576e-07, 2.31485577e-07,
                                                -6.68816586e-16, -4.48284993e-08, -8.37398903e-09, 7.56100615e-07,
                                                5.33433335e-03, 2.91848485e-01, 7.45623586e-05, 2.99902784e-01,
                                                -7.15601860e-02, -9.44665089e-02, 1.49646097e-02, -1.10990294e-01,
                                                -3.30174644e-03, 1.19462201e-01, 4.05130821e-04, -3.95036450e-04,
                                                -1.53880539e-07, -1.37393338e-07, 1.07636483e-14, 5.51953825e-06,
                                                -6.18188284e-06, 1.31307184e-17, -1.03617993e-07, -1.66528917e-07,
                                                1.06089030e-14, 6.69000941e-06, -4.16267252e-06, 3.63225324e-17,
                                                -1.39095626e-07, -1.39095626e-07, 1.10587840e-14, 5.58792469e-06,
                                                -5.58792469e-06, -2.07082526e-17])

                sim.set_state_from_flattened(out_of_way_state)
                sim.forward()
                inner_env.goal = ep['goal'][0]
                subgoal_time = ep['log_subgoal_time']
                sites_offset = (sim.data.site_xpos - sim.model.site_pos)
                site_id = sim.model.site_name2id('gripper_site')

                def unnorm_ob(ob):
                    return env.obs_min + ob * (env.obs_max - env.obs_min)

                for i, obs in enumerate(ep['observation']):

                    obs = unnorm_ob(obs)
                    grip_pos = obs[:3]
                    gripper_state = obs[3:5]
                    all_obj_pos = np.split(obs[5:5+3*env.n], env.n)

                    sim.model.site_pos[site_id] = grip_pos - \
                        sites_offset[site_id]

                    for i, pos in enumerate(all_obj_pos):
                        sim.data.set_joint_qpos(f"object{i}:joint", [
                                                *pos, *[1, 0, 0, 0]])

                    sim.forward()
                    img = sim.render(height=200, width=200,
                                     camera_name="external_camera_0")[::-1]
                    if subgoal_time > 0 and i >= subgoal_time:
                        img = img.copy()
                        cv2.putText(
                            img,
                            f"expl",
                            (16, 32),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    all_img.append(img)
                all_img = np.stack(all_img, 0)
                return all_img

            def episode_render_fn(env, ep):
                sim = env.sim
                all_img = []
                goals = []
                executions = []
                env.reset()

                if env.n == 3:
                    out_of_way_state = np.array([4.40000000e+00,    4.04999349e-01,    4.79999636e-01,    2.79652104e-06, 1.56722299e-02, -3.41500342e+00, 9.11469058e-02, -1.27681180e+00,
                                                 -1.39750475e+00, 4.43858450e+00, 7.47892234e-01, 2.53633962e-01,
                                                 2.34366216e+00, 3.35102418e+00, 8.32919575e-04, 1.41610111e-03,
                                                 1.32999932e+00, 6.49999392e-01, 4.24784489e-01, 1.00000000e+00,
                                                 -2.28652597e-07, 2.56090909e-07, -1.20181003e-15, 1.32999955e+00,
                                                 8.49999274e-01, 4.24784489e-01, 1.00000000e+00, -2.77140579e-07,
                                                 1.72443027e-07, -1.77971404e-15, 1.39999939e+00, 7.49999392e-01,
                                                 4.24784489e-01, 1.00000000e+00, -2.31485576e-07, 2.31485577e-07,
                                                 -6.68816586e-16, -4.48284993e-08, -8.37398903e-09, 7.56100615e-07,
                                                 5.33433335e-03, 2.91848485e-01, 7.45623586e-05, 2.99902784e-01,
                                                 -7.15601860e-02, -9.44665089e-02, 1.49646097e-02, -1.10990294e-01,
                                                 -3.30174644e-03, 1.19462201e-01, 4.05130821e-04, -3.95036450e-04,
                                                 -1.53880539e-07, -1.37393338e-07, 1.07636483e-14, 5.51953825e-06,
                                                 -6.18188284e-06, 1.31307184e-17, -1.03617993e-07, -1.66528917e-07,
                                                 1.06089030e-14, 6.69000941e-06, -4.16267252e-06, 3.63225324e-17,
                                                 -1.39095626e-07, -1.39095626e-07, 1.10587840e-14, 5.58792469e-06,
                                                 -5.58792469e-06, -2.07082526e-17])

                sim.set_state_from_flattened(out_of_way_state)
                sim.forward()

                sites_offset = (sim.data.site_xpos - sim.model.site_pos)
                site_id = sim.model.site_name2id('gripper_site')

                def unnorm_ob(ob):
                    return env.obs_min + ob * (env.obs_max - env.obs_min)

                for goal, obs in zip(ep['goal'], ep['observation']):

                    obs = unnorm_ob(obs)
                    grip_pos = obs[:3]
                    gripper_state = obs[3:5]
                    all_obj_pos = np.split(obs[5:5+3*env.n], env.n)

                    sim.model.site_pos[site_id] = grip_pos - \
                        sites_offset[site_id]

                    for i, pos in enumerate(all_obj_pos):
                        sim.data.set_joint_qpos(f"object{i}:joint", [
                                                *pos, *[1, 0, 0, 0]])

                    sim.forward()
                    img = sim.render(height=200, width=200,
                                     camera_name="external_camera_0")[::-1]

                    goal = unnorm_ob(goal)
                    grip_pos = goal[:3]
                    gripper_state = goal[3:5]
                    all_obj_pos = np.split(goal[5:5+3*env.n], env.n)

                    sim.model.site_pos[site_id] = grip_pos - \
                        sites_offset[site_id]

                    for i, pos in enumerate(all_obj_pos):
                        sim.data.set_joint_qpos(f"object{i}:joint", [
                                                *pos, *[1, 0, 0, 0]])

                    sim.forward()
                    goal_img = sim.render(
                        height=200, width=200, camera_name="external_camera_0")[::-1]

                    img = np.concatenate([goal_img, img], -3)

                    all_img.append(img)

                ep_img = np.stack(all_img, 0)
                T = ep_img.shape[0]

                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T),
                                    (0, 0), (0, 0), (0, 0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]

                return ep_img

        elif config.task in {'umazefull', 'umazefulldownscale', 'hardumazefulldownscale'}:

            def episode_render_fn(env, ep):

                ant_env = env.maze.wrapped_env

                ant_env.set_state(ep['goal'][0][:15], ep['goal'][0][:14])

                inner_env = env._env._env._env
                all_img = []
                for obs, goal in zip(ep['observation'], ep['goal']):
                    inner_env.maze.wrapped_env.set_state(
                        obs[:15], np.zeros_like(obs[:14]))
                    inner_env.g_xy = goal[:2]
                    inner_env.maze.wrapped_env.sim.forward()
                    img = env.render(mode='rgb_array')

                    inner_env.maze.wrapped_env.set_state(goal[:15], goal[:14])
                    inner_env.g_xy = goal[:2]
                    ant_env.sim.forward()
                    goal_img = env.render(mode='rgb_array')

                    img = np.concatenate([goal_img, img], -3)

                    all_img.append(img)

                ep_img = np.stack(all_img, 0)
                T = ep_img.shape[0]

                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T),
                                    (0, 0), (0, 0), (0, 0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]

                return ep_img

        elif 'pointmaze' in config.task:
            def episode_render_fn(env, ep):
                all_img = []
                inner_env = env._env._env._env._env
                for g_xy, xy in zip(ep['goal'], ep['observation']):
                    inner_env.g_xy = g_xy
                    inner_env.s_xy = xy
                    img = env.render()
                    all_img.append(img)
                env.clear_plots()

                ep_img = np.stack(all_img, 0)
                T = ep_img.shape[0]

                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T),
                                    (0, 0), (0, 0), (0, 0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]

                return ep_img

        elif 'dmc_walker_walk_proprio' == config.task:

            def episode_render_fn(env, ep):
                all_img = []
                inner_env = env._env._env._env._env

                for qpos, goal in zip(ep['qpos'], ep['goal']):
                    size = inner_env.physics.get_state(
                    ).shape[0] - qpos.shape[0]
                    inner_env.physics.set_state(
                        np.concatenate((qpos, np.zeros([size]))))
                    inner_env.physics.step()
                    img = env.render()

                    size = inner_env.physics.get_state(
                    ).shape[0] - goal.shape[0]
                    inner_env.physics.set_state(
                        np.concatenate((goal, np.zeros([size]))))
                    inner_env.physics.step()
                    goal_img = env.render()

                    img = np.concatenate([goal_img, img], -3)

                    all_img.append(img)

                ep_img = np.stack(all_img, 0)
                T = ep_img.shape[0]

                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T),
                                    (0, 0), (0, 0), (0, 0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]

                return ep_img

        elif 'dmc_humanoid_walk_proprio' == config.task:

            def episode_render_fn(env, ep):
                all_img = []
                inner_env = env._env._env._env._env._env

                def unnorm_ob(ob):
                    return env.obs_min + ob * (env.obs_max - env.obs_min)
                for qpos in ep['qpos']:
                    size = inner_env.physics.get_state(
                    ).shape[0] - qpos.shape[0]
                    qpos = unnorm_ob(qpos)
                    inner_env.physics.set_state(
                        np.concatenate((qpos, np.zeros([size]))))
                    inner_env.physics.step()
                    img = env.render()
                    all_img.append(img)

                ep_img = np.stack(all_img, 0)
                T = ep_img.shape[0]

                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T),
                                    (0, 0), (0, 0), (0, 0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]

                return ep_img

        elif config.task == "PegInsertionSide-v0":

            def episode_render_fn(env, ep):

                env.reset()
                all_img = []

                for env_state in ep['env_states']:
                    env.set_state(env_state)
                    image = env.render()
                    all_img.append(image)

                ep_img = np.stack(all_img, 0)
                T = ep_img.shape[0]

                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T),
                                    (0, 0), (0, 0), (0, 0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]

                return ep_img

        elif config.task == "PickAndPlace" or config.task == "FetchPush-v2" or "HandManipulateBlock" in config.task or "HandManipulatePen" in config.task or "HandManipulateEgg" in config.task:

            def episode_render_fn_original(env, ep):

                goal_idx = ep['goal_idx'][0]
                env._env.set_goal_idx(goal_idx)

                env.reset()

                all_img = []

                new_ep = []

                for action in ep['action']:

                    action = {'action': action}

                    obs = env.step(action)

                    new_ep.append(obs)

                    all_img.append(env.render())

                ep_img = np.stack(all_img, 0)

                T = ep_img.shape[0]

                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T),
                                    (0, 0), (0, 0), (0, 0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]

                return ep_img

            def episode_render_fn(env, ep):

                env.reset()

                all_img = []

                for obs, goal in zip(ep['observation'], ep['goal']):

                    img = env.render_with_obs(obs, goal, width=200, height=200)

                    all_img.append(img)

                ep_img = np.stack(all_img, 0)
                T = ep_img.shape[0]

                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T),
                                    (0, 0), (0, 0), (0, 0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]

                return ep_img

        return episode_render_fn

    def make_eval_fn(self, config):

        episode_render_fn = self.make_ep_render_fn(config)

        if 'demofetchpnp' in config.task:

            def eval_fn(driver, eval_policy, logger):
                env = driver._envs[0]

                eval_goal_idxs = range(len(env.get_goals()))

                num_eval_eps = 10
                all_metric_success = []

                ep_metrics = collections.defaultdict(list)

                all_ep_videos = []

                for ep_idx in range(num_eval_eps):

                    should_video = ep_idx == 0 and episode_render_fn is not None

                    for idx in eval_goal_idxs:
                        driver.reset()
                        env.set_goal_idx(idx)

                        driver(eval_policy, episodes=1)
                        """ rendering based on state."""
                        ep = driver._eps[0]

                        ep = {k: driver._convert(
                            [t[k] for t in ep]) for k in ep[0]}
                        score = float(ep['reward'].astype(np.float64).sum())
                        print(
                            f'Eval goal {idx} has {len(ep["reward"] - 1)} steps and return {score:.1f}.')

                        for k, v in ep.items():
                            if 'metric_success/goal_' in k:
                                ep_metrics[k].append(np.max(v))
                                all_metric_success.append(np.max(v))

                        if should_video:

                            ep_video = episode_render_fn(env, ep)
                            all_ep_videos.append(ep_video[None])

                    if should_video:

                        gc_video = np.concatenate(all_ep_videos, 0)

                        logger.video(f'eval_gc_policy{ep_idx}', gc_video)

                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_metric_success/goal_all',
                              all_metric_success)
                for key, value in ep_metrics.items():
                    logger.scalar(f'mean_eval_{key}', np.mean(value))
                logger.write()

            def eval_from_specific_goal(driver, eval_policy, logger):

                save_path = str(config.logdir) + \
                    "/ep_videos_from_specific_goal/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                env = driver._envs[0]

                eval_goal_idxs = range(len(env.get_goals()))

                goal_list = env.get_goals()
                num_eval_eps = 1
                all_metric_success = []

                ep_metrics = collections.defaultdict(list)

                def set_state(env, obs):

                    sim = env.sim

                    if env.n == 3:
                        out_of_way_state = np.array([4.40000000e+00,    4.04999349e-01,    4.79999636e-01,    2.79652104e-06, 1.56722299e-02, -3.41500342e+00, 9.11469058e-02, -1.27681180e+00,
                                                     -1.39750475e+00, 4.43858450e+00, 7.47892234e-01, 2.53633962e-01,
                                                     2.34366216e+00, 3.35102418e+00, 8.32919575e-04, 1.41610111e-03,
                                                     1.32999932e+00, 6.49999392e-01, 4.24784489e-01, 1.00000000e+00,
                                                     -2.28652597e-07, 2.56090909e-07, -1.20181003e-15, 1.32999955e+00,
                                                     8.49999274e-01, 4.24784489e-01, 1.00000000e+00, -2.77140579e-07,
                                                     1.72443027e-07, -1.77971404e-15, 1.39999939e+00, 7.49999392e-01,
                                                     4.24784489e-01, 1.00000000e+00, -2.31485576e-07, 2.31485577e-07,
                                                     -6.68816586e-16, -4.48284993e-08, -8.37398903e-09, 7.56100615e-07,
                                                     5.33433335e-03, 2.91848485e-01, 7.45623586e-05, 2.99902784e-01,
                                                     -7.15601860e-02, -9.44665089e-02, 1.49646097e-02, -1.10990294e-01,
                                                     -3.30174644e-03, 1.19462201e-01, 4.05130821e-04, -3.95036450e-04,
                                                     -1.53880539e-07, -1.37393338e-07, 1.07636483e-14, 5.51953825e-06,
                                                     -6.18188284e-06, 1.31307184e-17, -1.03617993e-07, -1.66528917e-07,
                                                     1.06089030e-14, 6.69000941e-06, -4.16267252e-06, 3.63225324e-17,
                                                     -1.39095626e-07, -1.39095626e-07, 1.10587840e-14, 5.58792469e-06,
                                                     -5.58792469e-06, -2.07082526e-17])

                    sites_offset = (sim.data.site_xpos - sim.model.site_pos)
                    site_id = sim.model.site_name2id('gripper_site')

                    def unnorm_ob(ob):
                        return env.obs_min + ob * (env.obs_max - env.obs_min)

                    obs = unnorm_ob(obs)
                    grip_pos = obs[:3]
                    gripper_state = obs[3:5]
                    all_obj_pos = np.split(obs[5:5+3*env.n], env.n)

                    sim.model.site_pos[site_id] = grip_pos - \
                        sites_offset[site_id]

                    for i, pos in enumerate(all_obj_pos):
                        sim.data.set_joint_qpos(f"object{i}:joint", [
                                                *pos, *[1, 0, 0, 0]])

                    sim.forward()

                driver.set_state_fn = set_state

                ep_num = 0
                for ep_idx in range(num_eval_eps):

                    should_video = True

                    for start_goal_idx in eval_goal_idxs:

                        start_goal = goal_list[start_goal_idx]

                        for idx in eval_goal_idxs:

                            if idx == start_goal_idx:
                                continue

                            driver.reset()

                            env.set_goal_idx(idx)
                            driver.if_set_initial_state = True
                            driver.initial_state = start_goal

                            driver(eval_policy, episodes=1)
                            """ rendering based on state."""
                            ep = driver._eps[0]

                            ep = {k: driver._convert(
                                [t[k] for t in ep]) for k in ep[0]}
                            score = float(
                                ep['reward'].astype(np.float64).sum())

                            for k, v in ep.items():
                                if 'metric_success/goal' in k:
                                    ep_metrics[k].append(np.max(v))
                                    all_metric_success.append(np.max(v))

                            if should_video:

                                ep_video = episode_render_fn(env, ep)

                                imageio.mimsave(
                                    save_path + f'ep_with_start_goal{start_goal_idx}_end_goal{idx}.gif', ep_video)

                            ep_num += 1

                all_metric_success = np.mean(all_metric_success)
                logger.scalar(
                    'mean_eval_metric_success_from_specific_goal', all_metric_success)

                for key, value in ep_metrics.items():
                    logger.scalar(f'mean_eval_end_goal_{key}', np.mean(value))

                logger.write()

        elif config.task in {'umazefull', 'umazefulldownscale', 'hardumazefulldownscale'}:

            def eval_fn(driver, eval_policy, logger):
                env = driver._envs[0]
                num_goals = len(env.get_goals()) if len(
                    env.get_goals()) > 0 else 5
                num_eval_eps = 10
                executions = []
                goals = []
                all_metric_success = []
                all_ep_videos = []

                ep_metrics = collections.defaultdict(list)
                for ep_idx in range(num_eval_eps):
                    should_video = ep_idx == 0 and episode_render_fn is not None
                    for idx in range(num_goals):
                        env.set_goal_idx(idx)
                        driver.reset()

                        driver(eval_policy, episodes=1)
                        """ rendering based on state."""
                        ep = driver._eps[0]
                        ep = {k: driver._convert(
                            [t[k] for t in ep]) for k in ep[0]}

                        score = float(ep['reward'].astype(np.float64).sum())
                        print(
                            f'Eval goal {idx} has {len(ep["reward"] - 1)} steps and return {score:.1f}.')

                        for k, v in ep.items():
                            if 'metric' in k:
                                ep_metrics[k].append(np.max(v))
                                all_metric_success.append(np.max(v))

                        if should_video:

                            ep_video = episode_render_fn(env, ep)
                            all_ep_videos.append(ep_video[None])

                    if should_video:

                        gc_video = np.concatenate(all_ep_videos, 0)

                        logger.video(f'eval_gc_policy{ep_idx}', gc_video)

                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_metric_success/goal_all',
                              all_metric_success)
                for key, value in ep_metrics.items():
                    logger.scalar(f'mean_eval_{key}', np.mean(value))
                logger.write()

            def eval_fn_collect(driver, eval_policy, logger):
                env = driver._envs[0]
                num_goals = len(env.get_goals()) if len(
                    env.get_goals()) > 0 else 5
                num_eval_eps = 10
                executions = []
                goals = []
                all_metric_success = []

                ep_metrics = collections.defaultdict(list)

                print(
                    "==============================collect the demonstration data=======================================")

                for idx in range(num_goals):

                    if idx < 15:

                        num_eval_eps = 50

                    else:
                        num_eval_eps = 100

                    for ep_idx in range(num_eval_eps):

                        env.set_goal_idx(idx)

                        driver.reset()

                        driver(eval_policy, episodes=1)
                        """ rendering based on state."""
                        ep = driver._eps[0]
                        ep = {k: driver._convert(
                            [t[k] for t in ep]) for k in ep[0]}
                        score = float(ep['reward'].astype(np.float64).sum())
                        print(
                            f'Eval goal {idx} has {len(ep["reward"] - 1)} steps and return {score:.1f}.')

                        for k, v in ep.items():
                            if 'metric' in k:
                                ep_metrics[k].append(np.max(v))
                                all_metric_success.append(np.max(v))

                                if np.max(v) > 0:
                                    np.savez(
                                        f'/common/home/yd374/ACH_Server/Experiment/Ant_Maze_Demo/goal_{idx}_ep_{ep_idx}.npz', observation=ep['observation'], action=ep['action'])
                                else:
                                    ep_idx -= 1

                all_metric_success = np.mean(all_metric_success)

                for key, value in ep_metrics.items():
                    print(f'mean_eval_{key}', np.mean(value))

                sys.exit()

            def eval_from_specific_goal(driver, eval_policy, logger):

                save_path = str(config.logdir) + \
                    "/ep_videos_from_specific_goal/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                env = driver._envs[0]

                eval_goal_idxs = [
                    i * 4 for i in range(int(len(env.get_goals())/4))]

                goal_list = env.get_goals()
                num_eval_eps = 1
                all_metric_success = []

                ep_metrics = collections.defaultdict(list)

                def set_state(env, obs):
                    ant_env = env.maze.wrapped_env

                    ant_env.set_state(obs[:15], obs[:14])

                    inner_env = env._env._env._env

                    inner_env.maze.wrapped_env.set_state(
                        obs[:15], np.zeros_like(obs[:14]))
                    inner_env.maze.wrapped_env.sim.forward()

                driver.set_state_fn = set_state

                ep_num = 0
                for ep_idx in range(num_eval_eps):

                    should_video = True

                    for start_goal_idx in eval_goal_idxs:

                        start_goal = goal_list[start_goal_idx]

                        for idx in eval_goal_idxs:

                            if idx == start_goal_idx:
                                continue

                            driver.reset()

                            env.set_goal_idx(idx)
                            driver.if_set_initial_state = True
                            driver.initial_state = start_goal

                            driver(eval_policy, episodes=1)
                            """ rendering based on state."""
                            ep = driver._eps[0]

                            ep = {k: driver._convert(
                                [t[k] for t in ep]) for k in ep[0]}
                            score = float(
                                ep['reward'].astype(np.float64).sum())

                            for k, v in ep.items():
                                if 'metric' in k:
                                    ep_metrics[k].append(np.max(v))
                                    all_metric_success.append(np.max(v))

                            if should_video:

                                ep_video = episode_render_fn(env, ep)

                                imageio.mimsave(
                                    save_path + f'ep_with_start_goal{start_goal_idx}_end_goal{idx}.gif', ep_video)

                            ep_num += 1

                all_metric_success = np.mean(all_metric_success)
                logger.scalar(
                    'mean_eval_metric_success_from_specific_goal', all_metric_success)

                for key, value in ep_metrics.items():
                    logger.scalar(f'mean_eval_end_goal{key}', np.mean(value))

                logger.write()

        elif 'pointmaze' in config.task:

            def eval_fn(driver, eval_policy, logger):
                env = driver._envs[0]
                num_goals = len(env.get_goals())
                num_eval_eps = 10
                all_ep_videos = []
                all_metric_success = []
                all_metric_success_cell = []
                ep_metrics = collections.defaultdict(list)
                for ep_idx in range(num_eval_eps):
                    should_video = ep_idx == 0 and episode_render_fn is not None
                    for idx in range(num_goals):
                        env.set_goal_idx(idx)
                        driver(eval_policy, episodes=1)
                        """ rendering based on state."""
                        ep = driver._eps[0]
                        ep = {k: driver._convert(
                            [t[k] for t in ep]) for k in ep[0]}

                        for k, v in ep.items():
                            if 'metric' in k:

                                ep_metrics[k].append(np.max(v))

                                if 'cell' in k.split('/')[0]:
                                    all_metric_success_cell.append(np.max(v))
                                else:
                                    all_metric_success.append(np.max(v))

                        if should_video:

                            ep_video = episode_render_fn(env, ep)
                            all_ep_videos.append(ep_video[None])

                    if should_video:

                        gc_video = np.concatenate(all_ep_videos, 0)

                        logger.video(f'eval_gc_policy{ep_idx}', gc_video)

                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_metric_success/goal_all',
                              all_metric_success)
                all_metric_success_cell = np.mean(all_metric_success_cell)
                logger.scalar(
                    'mean_eval_metric_success_cell/goal_all', all_metric_success_cell)

                for key, value in ep_metrics.items():
                    logger.scalar(f'mean_eval_{key}', np.mean(value))

                logger.write()

        elif 'dmc' in config.task:

            def eval_fn(driver, eval_policy, logger):
                env = driver._envs[0]
                num_goals = len(env.get_goals())
                num_eval_eps = 10
                all_ep_videos = []

                all_metric_success = []
                ep_metrics = collections.defaultdict(list)
                for ep_idx in range(num_eval_eps):
                    should_video = ep_idx == 0 and episode_render_fn is not None
                    for idx in range(num_goals):
                        env.set_goal_idx(idx)
                        env.reset()
                        driver(eval_policy, episodes=1)
                        ep = driver._eps[0]
                        ep = {k: driver._convert(
                            [t[k] for t in ep]) for k in ep[0]}
                        score = float(ep['reward'].astype(np.float64).sum())
                        print(
                            f'Eval goal {idx} has {len(ep["reward"] - 1)} steps and return {score:.1f}.')
                        for k, v in ep.items():
                            if 'metric_success' in k:
                                all_metric_success.append(np.max(v))
                                ep_metrics[k].append(np.max(v))
                            elif 'metric_reward' in k:
                                ep_metrics[k].append(np.sum(v))

                        if should_video:

                            ep_video = episode_render_fn(env, ep)
                            all_ep_videos.append(ep_video[None])

                    if should_video:

                        gc_video = np.concatenate(all_ep_videos, 0)

                        logger.video(f'eval_gc_policy{ep_idx}', gc_video)

                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_metric_success/goal_all',
                              all_metric_success)
                for key, value in ep_metrics.items():
                    if 'metric_success' in key:
                        logger.scalar(f'mean_eval_{key}', np.mean(value))
                    elif 'metric_reward' in key:
                        logger.scalar(f'sum_eval_{key}', np.mean(value))
                logger.write()

            def eval_from_specific_goal(driver, eval_policy, logger):

                save_path = str(config.logdir) + \
                    "/ep_videos_from_specific_goal/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                env = driver._envs[0]

                eval_goal_idxs = range(len(env.get_goals()))

                goal_list = env.get_goals()
                num_eval_eps = 1
                all_metric_success = []

                ep_metrics = collections.defaultdict(list)

                def set_state(env, qpos):
                    inner_env = env._env._env._env._env

                    size = inner_env.physics.get_state(
                    ).shape[0] - qpos.shape[0]
                    inner_env.physics.set_state(
                        np.concatenate((qpos, np.zeros([size]))))
                    inner_env.physics.step()

                driver.set_state_fn = set_state

                ep_num = 0
                for ep_idx in range(num_eval_eps):

                    should_video = True

                    for start_goal_idx in eval_goal_idxs:

                        start_goal = goal_list[start_goal_idx]

                        for idx in eval_goal_idxs:

                            if idx == start_goal_idx:
                                continue

                            driver.reset()

                            env.set_goal_idx(idx)
                            driver.if_set_initial_state = True
                            driver.initial_state = start_goal

                            driver(eval_policy, episodes=1)
                            """ rendering based on state."""
                            ep = driver._eps[0]

                            ep = {k: driver._convert(
                                [t[k] for t in ep]) for k in ep[0]}
                            score = float(
                                ep['reward'].astype(np.float64).sum())

                            for k, v in ep.items():
                                if 'metric_success' in k:
                                    ep_metrics[k].append(np.max(v))
                                    all_metric_success.append(np.max(v))

                            if should_video:

                                ep_video = episode_render_fn(env, ep)

                                imageio.mimsave(
                                    save_path + f'ep_with_start_goal{start_goal_idx}_end_goal{idx}.gif', ep_video)

                            ep_num += 1

                all_metric_success = np.mean(all_metric_success)
                logger.scalar(
                    'mean_eval_metric_success_from_specific_goal', all_metric_success)

                for key, value in ep_metrics.items():
                    logger.scalar(f'mean_eval_end_goal{key}', np.mean(value))

                logger.write()

        elif config.task == "PegInsertionSide-v0":

            def eval_fn_original(driver, eval_policy, logger):

                env = driver._envs[0]
                num_eval_eps = 10
                executions = []
                goals = []

                all_metric_success = []

                all_ep_video = []

                for ep_idx in range(num_eval_eps):

                    obs = env.reset()

                    for k in obs:
                        obs[k] = np.stack([obs[k]])

                    state = None
                    done = False

                    ep = []
                    ep_video = []
                    while not done:

                        actions, state = eval_policy(obs, state)

                        actions = [{k: np.array(actions[k][0])
                                    for k in actions}]

                        obs = env.step(actions[0])

                        for k in obs:
                            obs[k] = np.stack([obs[k]])

                        done = obs['is_last']

                        ep.append(obs)

                        ep_video.append(env.render())

                    ep_video = np.stack(ep_video, 0)
                    all_ep_video.append(ep_video[None])

                    ep = {k: driver._convert([t[k] for t in ep])
                          for k in ep[0]}
                    all_metric_success.append(max(ep['is_terminal']))

                all_ep_video = np.concatenate(all_ep_video, 0)

                logger.video(f'eval_gc_policy', all_ep_video)

                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_metric_success', all_metric_success)

                logger.write()

            def eval_fn_with_seed_reset(driver, eval_policy, logger):
                env = driver._envs[0]

                env = env._env
                eval_goal_num = 5
                num_eval_eps = 5
                executions = []
                goals = []

                all_metric_success = []
                ep_metrics = collections.defaultdict(list)

                for goal in range(eval_goal_num):

                    for ep_idx in range(num_eval_eps):

                        should_video = episode_render_fn is not None and ep_idx == 0

                        env.set_goal_idx(goal)

                        driver(eval_policy, episodes=1)
                        ep = driver._eps[0]
                        ep = {k: driver._convert(
                            [t[k] for t in ep]) for k in ep[0]}

                        k = 'goal_' + str(goal) + '_success'

                        ep_metrics[k].append(np.max(ep['is_terminal']))
                        all_metric_success.append(np.max(ep['is_terminal']))

                        if should_video:
                            """ rendering based on state."""

                            _executions = episode_render_fn(env, ep)
                            executions.extend(_executions)

                executions = np.concatenate(executions, 2)
                logger.video(f'eval_gc_policy', executions)

                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_goal_all_success', all_metric_success)

                for key, value in ep_metrics.items():

                    logger.scalar(f'mean_eval_{key}', np.mean(value))

                logger.write()

            def eval_fn(driver, eval_policy, logger):

                env = driver._envs[0]

                env = env._env
                num_eval_eps = 20

                all_metric_success = []
                all_ep_video = []
                ep_metrics = collections.defaultdict(list)

                for ep_idx in range(num_eval_eps):

                    should_video = episode_render_fn is not None and ep_idx < 5

                    driver(eval_policy, episodes=1)
                    ep = driver._eps[0]
                    ep = {k: driver._convert([t[k] for t in ep])
                          for k in ep[0]}

                    if should_video:
                        ep_video = episode_render_fn(env, ep)
                        all_ep_video.append(ep_video[None])

                    all_metric_success.append(np.max(ep['is_terminal']))

                all_ep_video = np.concatenate(all_ep_video, 0)

                logger.video(f'eval_gc_policy', all_ep_video)

                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_goal_all_success', all_metric_success)

                logger.write()

        elif config.task == "PickAndPlace" or "HandManipulateBlock" in config.task or "HandManipulatePen" in config.task or "HandManipulateEgg" in config.task:

            def eval_fn_with_random_reset(driver, eval_policy, logger):

                env = driver._envs[0]

                env = env._env
                num_eval_eps = 50

                all_metric_success = []
                all_ep_video = []
                ep_metrics = collections.defaultdict(list)

                for ep_idx in range(num_eval_eps):

                    should_video = ep_idx < 5

                    obs = env.reset()

                    ep_goal = obs['goal']

                    for k in obs:
                        obs[k] = np.stack([obs[k]])

                    state = None
                    done = False

                    ep = []
                    ep_video = []

                    while not done:

                        actions, state = eval_policy(obs, state)

                        actions = [{k: np.array(actions[k][0])
                                    for k in actions}]

                        obs = env.step(actions[0])

                        for k in obs:
                            obs[k] = np.stack([obs[k]])

                        done = obs['is_last']

                        ep.append(obs)

                        if should_video:
                            ep_video.append(env.render())

                    if should_video:
                        ep_video = np.stack(ep_video, 0)
                        all_ep_video.append(ep_video[None])

                    ep = {k: driver._convert([t[k] for t in ep])
                          for k in ep[0]}

                    if 0.42 <= ep_goal[2] < 0.52:
                        k = 'goal_low_success'

                    elif 0.52 <= ep_goal[2] < 0.62:
                        k = 'goal_medium_success'

                    elif 0.62 <= ep_goal[2]:
                        k = 'goal_high_success'

                    ep_metrics[k].append(np.max(ep['is_terminal']))

                    all_metric_success.append(np.max(ep['is_terminal']))

                all_ep_video = np.concatenate(all_ep_video, 0)

                logger.video(f'eval_gc_policy', all_ep_video)

                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_goal_all_success', all_metric_success)

                for key, value in ep_metrics.items():

                    logger.scalar(f'mean_eval_{key}', np.mean(value))

                logger.write()

            def eval_fn_with_seed_reset(driver, eval_policy, logger):

                env = driver._envs[0]

                env = env._env
                eval_goal_num = 10
                num_eval_eps = 5
                executions = []
                goals = []

                all_metric_success = []
                all_ep_video = []
                ep_metrics = collections.defaultdict(list)

                for goal in range(eval_goal_num):

                    for ep_idx in range(num_eval_eps):

                        should_video = ep_idx == 0

                        env.set_goal_idx(goal)

                        obs = env.reset()

                        for k in obs:
                            obs[k] = np.stack([obs[k]])

                        state = None
                        done = False

                        ep = []
                        ep_video = []

                        while not done:

                            actions, state = eval_policy(obs, state)

                            actions = [{k: np.array(actions[k][0])
                                        for k in actions}]

                            obs = env.step(actions[0])

                            for k in obs:
                                obs[k] = np.stack([obs[k]])

                            done = obs['is_last']

                            ep.append(obs)

                            if should_video:
                                ep_video.append(env.render())

                        if should_video:
                            ep_video = np.stack(ep_video, 0)
                            all_ep_video.append(ep_video[None])

                        ep = {k: driver._convert(
                            [t[k] for t in ep]) for k in ep[0]}
                        k = 'goal_' + str(goal) + '_success'
                        ep_metrics[k].append(np.max(ep['is_terminal']))

                        all_metric_success.append(np.max(ep['is_terminal']))

                all_ep_video = np.concatenate(all_ep_video, 0)

                logger.video(f'eval_gc_policy', all_ep_video)

                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_goal_all_success', all_metric_success)

                for key, value in ep_metrics.items():

                    logger.scalar(f'mean_eval_{key}', np.mean(value))

                logger.write()

            def eval_fn(driver, eval_policy, logger):

                env = driver._envs[0]

                env = env._env
                num_eval_eps = 50

                all_metric_success = []
                all_ep_video = []
                ep_metrics = collections.defaultdict(list)

                for ep_idx in range(num_eval_eps):

                    should_video = episode_render_fn is not None and ep_idx < 5

                    driver(eval_policy, episodes=1)
                    ep = driver._eps[0]
                    ep = {k: driver._convert([t[k] for t in ep])
                          for k in ep[0]}

                    if should_video:
                        ep_video = episode_render_fn(env, ep)
                        all_ep_video.append(ep_video[None])

                    all_metric_success.append(np.max(ep['is_terminal']))

                    if config.task == "PickAndPlace":
                        ep_goal = ep['goal'][0]

                        if 0.42 <= ep_goal[2] < 0.52:
                            k = 'goal_low_success'

                        elif 0.52 <= ep_goal[2] < 0.62:
                            k = 'goal_medium_success'

                        elif 0.62 <= ep_goal[2]:
                            k = 'goal_high_success'

                        ep_metrics[k].append(np.max(ep['is_terminal']))

                all_ep_video = np.concatenate(all_ep_video, 0)

                logger.video(f'eval_gc_policy', all_ep_video)

                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_goal_all_success', all_metric_success)

                for key, value in ep_metrics.items():

                    logger.scalar(f'mean_eval_{key}', np.mean(value))

                logger.write()

        else:
            raise NotImplementedError

        return eval_fn

    def make_obs2goal_fn(self, config):
        obs2goal = None
        if "demofetchpnp" in config.task:
            def obs2goal(obs):
                return obs

        if config.task == "PickAndPlace":
            def obs2goal(obs):
                return obs[..., 3:6]

        if config.task == "FetchPush-v2":
            def obs2goal(obs):
                return obs[..., 3:6]

        if "HandManipulateBlock" in config.task or "HandManipulatePen" in config.task or "HandManipulateEgg" in config.task:

            def obs2goal(obs):

                if "HandManipulateBlockRotateXYZ" in config.task or "HandManipulatePenRotate" in config.task or "HandManipulateEggRotate" in config.task:
                    return obs[..., -4:]
                else:
                    return obs[..., -7:]

        if config.task == "PegInsertionSide-v0":

            def obs2goal(obs):
                return obs[..., -18:-11]

        return obs2goal

    def make_space_explored_plot_fn(self, config):

        space_explored_plot_fn = None

        if 'demofetchpnp' in config.task:
            def space_explored_plot_fn(env, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size=50):
                wm = agnt.wm
                obs_key = agnt.state_key
                goal_key = agnt.goal_key
                episodes = list(complete_episodes.values())
                num_goals = min(int(config.eval_every), len(episodes))
                recent_episodes = episodes[-num_goals:]
                if len(episodes) > num_goals:
                    old_episodes = episodes[:-num_goals][::5]
                    old_episodes.extend(recent_episodes)
                    episodes = old_episodes
                else:
                    episodes = recent_episodes

                all_observations = []
                value_list = []
                all_goals = []

                def unnorm_ob(ob):
                    return env.obs_min + ob * (env.obs_max - env.obs_min)

                for ep_count, episode in enumerate(episodes):

                    if (ep_count % batch_size) == 0:
                        start = ep_count
                        chunk = collections.defaultdict(list)
                    sequence = {
                        k: convert(v[::step_subsample])
                        for k, v in episode.items() if not k.startswith('log_')}
                    data = wm.preprocess(sequence)
                    for key, value in data.items():
                        chunk[key].append(value)

                    if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes) - 1):
                        end = ep_count
                        chunk = {k: tf.stack(v) for k, v in chunk.items()}
                        embed = wm.encoder(chunk)
                        post, _ = wm.rssm.observe(
                            embed, chunk['action'], chunk['is_first'], None)
                        chunk['feat'] = wm.rssm.get_feat(post)
                        value_fn = agnt._expl_behavior.ac._target_critic
                        value = value_fn(chunk['feat']).mode()
                        value_list.append(value)

                        all_observations.append(tf.stack(chunk[obs_key]))
                        all_goals.append(tf.stack(chunk[goal_key]))

                all_observations = np.concatenate(all_observations)
                all_observations = all_observations.reshape(
                    -1, all_observations.shape[-1])
                all_observations = unnorm_ob(all_observations)
                all_obj_obs_pos = np.split(
                    all_observations[:, 5:5+3*env.n], env.n, axis=1)

                all_goals = np.concatenate(all_goals)[:, 0]
                all_goals = unnorm_ob(all_goals)
                all_obj_g_pos = np.split(
                    all_goals[:, 5:5+3*env.n], env.n, axis=1)

                plot_dims = [[1, 2]]
                plot_dim_name = dict([(0, 'x'), (1, 'y'), (2, 'z')])

                def plot_axes(axes, data, cmap, title, zorder):
                    for ax, pd in zip(axes, plot_dims):
                        ax.scatter(x=data[:, pd[0]],
                                   y=data[:, pd[1]],
                                   s=1,
                                   c=np.arange(len(data)),
                                   cmap=cmap,
                                   zorder=zorder,
                                   )
                        ax.set_title(f"{title} {plot_dim_name[pd[0]]}{plot_dim_name[pd[1]]}", fontdict={
                                     'fontsize': 10})

                fig, all_axes = plt.subplots(
                    1, 2+env.n, figsize=(1+(2+env.n*3), 2))

                g_ax = all_axes[0]
                p2evalue_ax = all_axes[-1]
                obj_axes = all_axes[1:-1]
                obj_colors = ['Reds', 'Blues', 'Greens']
                for obj_ax, obj_pos, obj_g_pos, obj_color in zip(obj_axes, all_obj_obs_pos, all_obj_g_pos, obj_colors):
                    plot_axes([obj_ax], obj_pos, obj_color, f"State ", 3)
                    plot_axes([g_ax], obj_g_pos, obj_color, f"Goal ", 3)

                limits = [[0.5, 1.0], [0.4, 0.6]] if 'walls' in config.task else [
                    [1, 1.6], [0.3, 0.7]]
                for _ax in all_axes:
                    _ax.set_xlim(limits[0])
                    _ax.set_ylim(limits[1])
                    _ax.axes.get_yaxis().set_visible(False)

                values = tf.concat(value_list, axis=0)
                values = values.numpy().flatten()
                cm = plt.cm.get_cmap("viridis")
                for obj_ax, obj_pos, obj_color in zip(obj_axes, all_obj_obs_pos, obj_colors):
                    p2e_scatter = p2evalue_ax.scatter(
                        x=obj_pos[:, plot_dims[0][0]],
                        y=obj_pos[:, plot_dims[0][1]],
                        s=1,
                        c=values,
                        cmap=cm,
                        zorder=3,
                    )
                plt.colorbar(p2e_scatter, ax=p2evalue_ax)
                p2evalue_ax.set_title('p2e value')

                fig.canvas.draw()
                image_from_plot = np.frombuffer(
                    fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(
                    fig.canvas.get_width_height()[::-1] + (3,))
                image_from_plot = np.expand_dims(image_from_plot, axis=0)
                logger.image('state_occupancy', image_from_plot)
                plt.cla()
                plt.clf()

        elif config.task in {'umazefulldownscale', 'a1umazefulldownscale', 'hardumazefulldownscale'}:
            def space_explored_plot_fn(maze, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size=50):

                wm = agnt.wm
                obs_key = agnt.state_key
                goal_key = agnt.goal_key
                episodes = list(complete_episodes.values())
                num_goals = min(int(config.eval_every), len(episodes))
                recent_episodes = episodes[-num_goals:]
                if len(episodes) > num_goals:
                    old_episodes = episodes[:-num_goals][::5]
                    old_episodes.extend(recent_episodes)
                    episodes = old_episodes
                else:
                    episodes = recent_episodes

                obs = []
                value_list = []
                goals = []
                for ep_count, episode in enumerate(episodes):

                    if (ep_count % batch_size) == 0:
                        start = ep_count
                        chunk = collections.defaultdict(list)
                    sequence = {
                        k: convert(v[::step_subsample])
                        for k, v in episode.items() if not k.startswith('log_')}
                    data = wm.preprocess(sequence)
                    for key, value in data.items():
                        chunk[key].append(value)

                    if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes) - 1):
                        end = ep_count
                        chunk = {k: tf.stack(v) for k, v in chunk.items()}
                        embed = wm.encoder(chunk)
                        post, _ = wm.rssm.observe(
                            embed, chunk['action'], chunk['is_first'], None)
                        chunk['feat'] = wm.rssm.get_feat(post)
                        value_fn = agnt._expl_behavior.ac._target_critic
                        value = value_fn(chunk['feat']).mode()
                        value_list.append(value)

                        obs.append(tf.stack(chunk[obs_key]))
                        goals.append(tf.stack(chunk[goal_key]))

                fig, (state_ax, dd_ax, p2evalue_ax) = plt.subplots(
                    1, 3, figsize=(1, 3))
                xlim = np.array([-1, 5.25])
                ylim = np.array([-1, 5.25])
                if config.task == 'a1umazefulldownscale':
                    xlim /= 2.0
                    ylim /= 2.0
                elif config.task == 'hardumazefulldownscale':
                    xlim = np.array([-1, 5.25])
                    ylim = np.array([-1, 9.25])

                state_ax.set(xlim=xlim, ylim=ylim)
                p2evalue_ax.set(xlim=xlim, ylim=ylim)
                goal_time_limit = round(
                    config.goal_policy_rollout_percentage * config.time_limit)
                obs_list = tf.concat(obs, axis=0)
                before = obs_list[:, :goal_time_limit, :]
                before = before[:, :, :2]
                ep_order_before = tf.range(before.shape[0])[:, None]
                ep_order_before = tf.repeat(
                    ep_order_before, before.shape[1], axis=1)
                before = tf.reshape(
                    before, [before.shape[0]*before.shape[1], 2])
                after = obs_list[:, goal_time_limit:, :]
                after = after[:, :, :2]
                ep_order_after = tf.range(after.shape[0])[:, None]
                ep_order_after = tf.repeat(
                    ep_order_after, after.shape[1], axis=1)
                after = tf.reshape(after, [after.shape[0]*after.shape[1], 2])

                ep_order_before = tf.reshape(
                    ep_order_before, [ep_order_before.shape[0]*ep_order_before.shape[1]])
                ep_order_after = tf.reshape(
                    ep_order_after, [ep_order_after.shape[0]*ep_order_after.shape[1]])
                goal_list = tf.concat(goals, axis=0)[:, 0, :2]
                goal_list = tf.reshape(goal_list, [-1, 2])

                state_ax.scatter(
                    x=before[:, 0],
                    y=before[:, 1],
                    s=1,
                    c=ep_order_before,
                    cmap='Blues',
                    zorder=3,
                )
                state_ax.scatter(
                    x=after[:, 0],
                    y=after[:, 1],
                    s=1,
                    c=ep_order_after,
                    cmap='Greens',
                    zorder=3,
                )
                state_ax.scatter(
                    x=goal_list[:, 0],
                    y=goal_list[:, 1],
                    s=1,
                    c=np.arange(goal_list.shape[0]),
                    cmap='Reds',
                    zorder=3,
                )
                x_min, x_max = xlim[0], xlim[1]
                y_min, y_max = ylim[0], ylim[1]
                x_div = y_div = 100
                if config.task in {'umazefulldownscale', 'hardumazefulldownscale'}:
                    other_dims = np.concatenate([[6.08193526e-01,    9.87496030e-01,
                                                  1.82685311e-03, -6.82827458e-03,    1.57485326e-01,    5.14617396e-02,
                                                  1.22386603e+00, -6.58701813e-02, -1.06980319e+00,    5.09069276e-01,
                                                  -1.15506861e+00,    5.25953435e-01,    7.11716520e-01], np.zeros(14)])
                elif config.task == 'a1umazefulldownscale':
                    other_dims = np.concatenate([[0.24556014,    0.986648,        0.09023235, -0.09100603,
                                                  0.10050705, -0.07250207, -0.01489305,    0.09989551, -0.05246516, -0.05311238,
                                                  -0.01864055, -0.05934234,    0.03910208, -
                                                  0.08356607,    0.05515265, -0.00453086,
                                                  -0.01196933], np.zeros(18)])
                gx = 0.0
                gy = 4.2
                if config.task == 'a1umazefulldownscale':
                    gx /= 2
                    gy /= 2
                elif config.task == 'hardumazefulldownscale':
                    gx = 4.2
                    gy = 8.2

                x = np.linspace(x_min, x_max, x_div)
                y = np.linspace(y_min, y_max, y_div)
                XY = X, Y = np.meshgrid(x, y)
                XY = np.stack([X, Y], axis=-1)
                XY = XY.reshape(x_div * y_div, 2)
                XY_plus = np.hstack(
                    (XY, np.tile(other_dims, (XY.shape[0], 1))))
                goal_vec = np.zeros((x_div*y_div, XY_plus.shape[-1]))
                goal_vec[:, 0] = goal_vec[:, 0] + gx
                goal_vec[:, 1] = goal_vec[:, 1] + gy
                goal_vec[:, 2:] = goal_vec[:, 2:] + other_dims
                obs = {"observation": XY_plus, "goal": goal_vec, "reward": np.zeros(
                    XY.shape[0]), "discount": np.ones(XY.shape[0]), "is_terminal": np.zeros(XY.shape[0])}
                temporal_dist = agnt.temporal_dist(obs)
                if config.gc_reward == 'dynamical_distance':
                    td_plot = dd_ax.tricontourf(
                        XY[:, 0], XY[:, 1], temporal_dist)
                    dd_ax.scatter(x=obs['goal'][0][0], y=obs['goal']
                                  [0][1], c="r", marker="*", s=20, zorder=2)
                    dd_ax.scatter(x=before[0][0], y=before[0]
                                  [1], c="b", marker=".", s=20, zorder=2)
                    plt.colorbar(td_plot, ax=dd_ax)
                    dd_ax.set_title('temporal distance')

                obs_list = obs_list[:, :, :2]
                obs_list = tf.reshape(
                    obs_list, [obs_list.shape[0]*obs_list.shape[1], 2])
                values = tf.concat(value_list, axis=0)
                values = values.numpy().flatten()
                cm = plt.cm.get_cmap("viridis")
                p2e_scatter = p2evalue_ax.scatter(
                    x=obs_list[:, 0],
                    y=obs_list[:, 1],
                    s=1,
                    c=values,
                    cmap=cm,
                    zorder=3,
                )
                plt.colorbar(p2e_scatter, ax=p2evalue_ax)
                p2evalue_ax.set_title('p2e value')

                fig = plt.gcf()
                fig.set_size_inches(10, 3)
                fig.canvas.draw()
                image_from_plot = np.frombuffer(
                    fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(
                    fig.canvas.get_width_height()[::-1] + (3,))
                image_from_plot = np.expand_dims(image_from_plot, axis=0)
                logger.image('state_occupancy', image_from_plot)
                plt.cla()
                plt.clf()

        elif 'pointmaze' in config.task:
            def space_explored_plot_fn(maze, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size=100):

                wm = agnt.wm
                obs_key = agnt.state_key
                goal_key = agnt.goal_key
                episodes = list(complete_episodes.values())
                obs = []
                goals = []
                reward_list = []
                for ep_count, episode in enumerate(episodes[::ep_subsample]):

                    if (ep_count % batch_size) == 0:
                        start = ep_count
                        chunk = collections.defaultdict(list)
                    sequence = {
                        k: convert(v[::step_subsample])
                        for k, v in episode.items() if not k.startswith('log_')}
                    data = wm.preprocess(sequence)
                    for key, value in data.items():
                        chunk[key].append(value)

                    if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes[::ep_subsample]) - 1):
                        end = ep_count

                        obs.append(tf.stack(chunk[obs_key]))
                        goals.append(tf.stack(chunk[goal_key]))

                fig, ax = plt.subplots(1, 1, figsize=(1, 1))
                ax.set(xlim=(-1, 11), ylim=(-1, 11))
                maze.maze.plot(ax)
                obs_list = tf.concat(obs, axis=0)
                ep_order = tf.range(obs_list.shape[0])[:, None]
                ep_order = tf.repeat(ep_order, obs_list.shape[1], axis=1)
                obs_list = tf.reshape(
                    obs_list, [obs_list.shape[0]*obs_list.shape[1], 2])
                ep_order = tf.reshape(
                    ep_order, [ep_order.shape[0]*ep_order.shape[1]])
                goal_list = tf.concat(goals, axis=0)[:, 0, :]
                goal_list = tf.reshape(goal_list, [-1, 2])
                plt.scatter(
                    x=obs_list[:, 0],
                    y=obs_list[:, 1],
                    s=1,
                    c=ep_order,
                    cmap='Blues',
                    zorder=3,
                )
                plt.scatter(
                    x=goal_list[:, 0],
                    y=goal_list[:, 1],
                    s=1,
                    c=np.arange(goal_list.shape[0]),
                    cmap='Reds',
                    zorder=3,
                )
                fig = plt.gcf()
                plt.title('states')
                fig.set_size_inches(8, 8)
                fig.canvas.draw()
                image_from_plot = np.frombuffer(
                    fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(
                    fig.canvas.get_width_height()[::-1] + (3,))
                image_from_plot = np.expand_dims(image_from_plot, axis=0)
                logger.image('state_occupancy', image_from_plot)

        elif 'dmc_walker_walk_proprio' == config.task:
            def space_explored_plot_fn(eval_env, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size=50):

                wm = agnt.wm
                obs_key = agnt.state_key
                goal_key = agnt.goal_key
                episodes = list(complete_episodes.values())
                num_goals = min(int(config.eval_every), len(episodes))
                recent_episodes = episodes[-num_goals:]
                if len(episodes) > num_goals:
                    old_episodes = episodes[:-num_goals][::5]
                    old_episodes.extend(recent_episodes)
                    episodes = old_episodes
                else:
                    episodes = recent_episodes

                obs = []
                value_list = []
                goals = []
                for ep_count, episode in enumerate(episodes):

                    if (ep_count % batch_size) == 0:
                        start = ep_count
                        chunk = collections.defaultdict(list)
                    sequence = {
                        k: convert(v[::step_subsample])
                        for k, v in episode.items() if not k.startswith('log_')}
                    data = wm.preprocess(sequence)
                    for key, value in data.items():
                        chunk[key].append(value)

                    if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes) - 1):
                        end = ep_count
                        chunk = {k: tf.stack(v) for k, v in chunk.items()}
                        embed = wm.encoder(chunk)
                        post, _ = wm.rssm.observe(
                            embed, chunk['action'], chunk['is_first'], None)
                        chunk['feat'] = wm.rssm.get_feat(post)
                        value_fn = agnt._expl_behavior.ac._target_critic
                        value = value_fn(chunk['feat']).mode()
                        value_list.append(value)

                        obs.append(tf.stack(chunk[obs_key]))
                        goals.append(tf.stack(chunk[goal_key]))

                fig, (state_ax, dd_ax, p2evalue_ax) = plt.subplots(
                    1, 3, figsize=(1, 3))

                xlim = np.array([-20.0, 20.0])
                ylim = np.array([-1.3, 1.0])

                state_ax.set(xlim=xlim, ylim=ylim)
                p2evalue_ax.set(xlim=xlim, ylim=ylim)

                goal_time_limit = round(
                    config.goal_policy_rollout_percentage * config.time_limit)
                obs_list = tf.concat(obs, axis=0)

                before = obs_list[:, :goal_time_limit, :]
                before = before[:, :, :2]

                ep_order_before = tf.range(before.shape[0])[:, None]
                ep_order_before = tf.repeat(
                    ep_order_before, before.shape[1], axis=1)
                before = tf.reshape(
                    before, [before.shape[0]*before.shape[1], 2])

                after = obs_list[:, goal_time_limit:, :]
                after = after[:, :, :2]

                ep_order_after = tf.range(after.shape[0])[:, None]
                ep_order_after = tf.repeat(
                    ep_order_after, after.shape[1], axis=1)
                after = tf.reshape(after, [after.shape[0]*after.shape[1], 2])

                ep_order_before = tf.reshape(
                    ep_order_before, [ep_order_before.shape[0]*ep_order_before.shape[1]])
                ep_order_after = tf.reshape(
                    ep_order_after, [ep_order_after.shape[0]*ep_order_after.shape[1]])
                goal_list = tf.concat(goals, axis=0)[:, 0, :2]
                goal_list = tf.reshape(goal_list, [-1, 2])

                state_ax.scatter(
                    y=before[:, 0],
                    x=before[:, 1],
                    s=1,
                    c=ep_order_before,
                    cmap='Blues',
                    zorder=3,
                )
                state_ax.scatter(
                    y=after[:, 0],
                    x=after[:, 1],
                    s=1,
                    c=ep_order_after,
                    cmap='Greens',
                    zorder=3,
                )
                state_ax.scatter(
                    y=goal_list[:, 0],
                    x=goal_list[:, 1],
                    s=1,
                    c=np.arange(goal_list.shape[0]),
                    cmap='Reds',
                    zorder=3,
                )

                state_ax.set_title('space explored')

                x_min, x_max = xlim[0], xlim[1]
                y_min, y_max = ylim[0], ylim[1]
                x_div = y_div = 100
                other_dims = np.array(
                    [0.34, 0.74, -1.34, -0., 1.1, -0.66, -0.1])
                gx = 5.0
                gy = 0.0

                x = np.linspace(x_min, x_max, x_div)
                y = np.linspace(y_min, y_max, y_div)
                XY = X, Y = np.meshgrid(y, x)
                XY = np.stack([X, Y], axis=-1)
                XY = XY.reshape(x_div * y_div, 2)
                XY_plus = np.hstack(
                    (XY, np.tile(other_dims, (XY.shape[0], 1))))

                goal_vec = np.zeros((x_div*y_div, XY_plus.shape[-1]))
                goal_vec[:, 0] = goal_vec[:, 0] + gy
                goal_vec[:, 1] = goal_vec[:, 1] + gx
                goal_vec[:, 2:] = goal_vec[:, 2:] + other_dims

                obs = {"qpos": XY_plus, "goal": goal_vec, "reward": np.zeros(
                    XY.shape[0]), "discount": np.ones(XY.shape[0]), "is_terminal": np.zeros(XY.shape[0])}
                temporal_dist = agnt.temporal_dist(obs)
                if config.gc_reward == 'dynamical_distance':
                    td_plot = dd_ax.tricontourf(
                        XY[:, 1], XY[:, 0], temporal_dist)
                    dd_ax.scatter(y=obs['goal'][0][0], x=obs['goal']
                                  [0][1], c="r", marker="*", s=20, zorder=2)
                    dd_ax.scatter(y=before[0][0], x=before[0]
                                  [1], c="b", marker=".", s=20, zorder=2)
                    plt.colorbar(td_plot, ax=dd_ax)
                    dd_ax.set_title('temporal distance')

                obs_list = obs_list[:, :, :2]
                obs_list = tf.reshape(
                    obs_list, [obs_list.shape[0]*obs_list.shape[1], 2])
                values = tf.concat(value_list, axis=0)
                values = values.numpy().flatten()
                cm = plt.cm.get_cmap("viridis")
                p2e_scatter = p2evalue_ax.scatter(
                    y=obs_list[:, 0],
                    x=obs_list[:, 1],
                    s=1,
                    c=values,
                    cmap=cm,
                    zorder=3,
                )
                plt.colorbar(p2e_scatter, ax=p2evalue_ax)
                p2evalue_ax.set_title('p2e value')

                fig = plt.gcf()
                fig.set_size_inches(12, 3)
                fig.canvas.draw()
                image_from_plot = np.frombuffer(
                    fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(
                    fig.canvas.get_width_height()[::-1] + (3,))
                image_from_plot = np.expand_dims(image_from_plot, axis=0)
                logger.image('state_occupancy', image_from_plot)
                plt.cla()
                plt.clf()

        elif 'dmc_humanoid_walk_proprio' == config.task:
            def space_explored_plot_fn(env, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size=50):

                wm = agnt.wm
                obs_key = agnt.state_key
                goal_key = agnt.goal_key
                episodes = list(complete_episodes.values())
                num_goals = min(int(config.eval_every), len(episodes))
                recent_episodes = episodes[-num_goals:]
                if len(episodes) > num_goals:
                    old_episodes = episodes[:-num_goals][::5]
                    old_episodes.extend(recent_episodes)
                    episodes = old_episodes
                else:
                    episodes = recent_episodes

                obs = []
                value_list = []
                goals = []
                for ep_count, episode in enumerate(episodes):

                    if (ep_count % batch_size) == 0:
                        start = ep_count
                        chunk = collections.defaultdict(list)
                    sequence = {
                        k: convert(v[::step_subsample])
                        for k, v in episode.items() if not k.startswith('log_')}
                    data = wm.preprocess(sequence)
                    for key, value in data.items():
                        chunk[key].append(value)

                    if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes) - 1):
                        end = ep_count
                        chunk = {k: tf.stack(v) for k, v in chunk.items()}
                        embed = wm.encoder(chunk)
                        post, _ = wm.rssm.observe(
                            embed, chunk['action'], chunk['is_first'], None)
                        chunk['feat'] = wm.rssm.get_feat(post)
                        value_fn = agnt._expl_behavior.ac._target_critic
                        value = value_fn(chunk['feat']).mode()
                        value_list.append(value)

                        obs.append(tf.stack(chunk[obs_key]))
                        goals.append(tf.stack(chunk[goal_key]))

                fig, (state_ax, dd_ax, p2evalue_ax) = plt.subplots(
                    1, 3, figsize=(1, 3))
                xlim = np.array([-0.2, 1.2])
                ylim = np.array([-0.2, 1.2])

                state_ax.set(xlim=xlim, ylim=ylim)
                p2evalue_ax.set(xlim=xlim, ylim=ylim)
                goal_time_limit = round(
                    config.goal_policy_rollout_percentage * config.time_limit)
                obs_list = tf.concat(obs, axis=0)
                before = obs_list[:, :goal_time_limit, :]
                before = before[:, :, :28]
                ep_order_before = tf.range(before.shape[0])[:, None]
                ep_order_before = tf.repeat(
                    ep_order_before, before.shape[1], axis=1)
                before = tf.reshape(
                    before, [before.shape[0]*before.shape[1], 28])
                after = obs_list[:, goal_time_limit:, :]
                after = after[:, :, :28]
                ep_order_after = tf.range(after.shape[0])[:, None]
                ep_order_after = tf.repeat(
                    ep_order_after, after.shape[1], axis=1)
                after = tf.reshape(after, [after.shape[0]*after.shape[1], 28])

                ep_order_before = tf.reshape(
                    ep_order_before, [ep_order_before.shape[0]*ep_order_before.shape[1]])
                ep_order_after = tf.reshape(
                    ep_order_after, [ep_order_after.shape[0]*ep_order_after.shape[1]])
                goal_list = tf.concat(goals, axis=0)[:, 0, :28]
                goal_list = tf.reshape(goal_list, [-1, 28])

                state_ax.scatter(
                    y=before[:, 0],
                    x=before[:, 1],
                    s=1,
                    c=ep_order_before,
                    cmap='Blues',
                    zorder=3,
                )
                state_ax.scatter(
                    y=after[:, 0],
                    x=after[:, 1],
                    s=1,
                    c=ep_order_after,
                    cmap='Greens',
                    zorder=3,
                )
                state_ax.scatter(
                    y=goal_list[:, 0],
                    x=goal_list[:, 1],
                    s=1,
                    c=np.arange(goal_list.shape[0]),
                    cmap='Reds',
                    zorder=3,
                )

                obs_list = obs_list[:, :, :28]
                obs_list = tf.reshape(
                    obs_list, [obs_list.shape[0]*obs_list.shape[1], 28])
                values = tf.concat(value_list, axis=0)
                values = values.numpy().flatten()
                cm = plt.cm.get_cmap("viridis")
                p2e_scatter = p2evalue_ax.scatter(
                    y=obs_list[:, 0],
                    x=obs_list[:, 1],
                    s=1,
                    c=values,
                    cmap=cm,
                    zorder=3,
                )
                plt.colorbar(p2e_scatter, ax=p2evalue_ax)
                p2evalue_ax.set_title('p2e value')

                fig = plt.gcf()
                fig.set_size_inches(10, 3)
                fig.canvas.draw()
                image_from_plot = np.frombuffer(
                    fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(
                    fig.canvas.get_width_height()[::-1] + (3,))
                image_from_plot = np.expand_dims(image_from_plot, axis=0)
                logger.image('state_occupancy', image_from_plot)
                plt.cla()
                plt.clf()

        return None

    def make_cem_vis_fn(self, config):

        vis_fn = None
        if config.task in {'umazefulldownscale', 'hardumazefulldownscale'}:
            num_vis = 10

            def vis_fn(elite_inds, elite_samples, seq, wm, eval_env, logger):
                elite_seq = tf.nest.map_structure(
                    lambda x: tf.gather(x, elite_inds[:num_vis], axis=1), seq)
                elite_obs = wm.heads['decoder'](wm.rssm.get_feat(elite_seq))[
                    'observation'].mode()
                goal_states = tf.repeat(
                    elite_samples[None, :num_vis], elite_obs.shape[0], axis=0).numpy()
                goal_list = goal_states[..., :2]
                goal_list = tf.reshape(goal_list, [-1, 2])

                fig, p2evalue_ax = plt.subplots(1, 1, figsize=(1, 3))
                p2evalue_ax.scatter(
                    x=goal_list[:, 0],
                    y=goal_list[:, 1],
                    s=1,
                    c='r',
                    zorder=5,
                )
                elite_obs = tf.transpose(elite_obs, (1, 0, 2))

                first_half = elite_obs[:, :-10]
                first_half = first_half[:, ::10]
                second_half = elite_obs[:, -10:]
                traj = tf.concat([first_half, second_half], axis=1)
                p2evalue_ax.plot(
                    traj[:, :, 0],
                    traj[:, :, 1],
                    c='b',
                    zorder=4,
                    marker='.'
                )

                if 'hard' in config.task:
                    p2evalue_ax.set(xlim=(-1, 5.25), ylim=(-1, 9.25))
                else:
                    p2evalue_ax.set(xlim=(-1, 5.25), ylim=(-1, 5.25))
                p2evalue_ax.set_title('elite goals and states')
                fig = plt.gcf()
                fig.set_size_inches(7, 6)
                fig.canvas.draw()
                image_from_plot = np.frombuffer(
                    fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(
                    fig.canvas.get_width_height()[::-1] + (3,))
                image_from_plot = np.expand_dims(image_from_plot, axis=0)
                logger.image(f'top_{num_vis}_cem', image_from_plot)
                logger.write()

        elif 'pointmaze' in config.task:
            num_vis = 10

            def vis_fn(elite_inds, elite_samples, seq, wm, eval_env, logger):
                elite_seq = tf.nest.map_structure(
                    lambda x: tf.gather(x, elite_inds[:num_vis], axis=1), seq)
                elite_obs = wm.heads['decoder'](wm.rssm.get_feat(elite_seq))[
                    'observation'].mode()

                goal_states = tf.repeat(
                    elite_samples[None, :num_vis], elite_obs.shape[0], axis=0).numpy()
                goal_states = goal_states.reshape(-1, 2)
                maze_states = elite_obs.numpy().reshape(-1, 2)
                inner_env = eval_env._env._env._env._env
                all_img = []
                for xy, g_xy in zip(maze_states, goal_states):
                    inner_env.s_xy = xy
                    inner_env.g_xy = g_xy
                    img = (eval_env.render().astype(np.float32) / 255.0) - 0.5
                    all_img.append(img)

                eval_env.clear_plots()
                imgs = np.stack(all_img, 0)
                imgs = imgs.reshape([*elite_obs.shape[:2], 100, 100, 3])
                T, B, H, W, C = imgs.shape

                imgs = imgs.transpose(0, 2, 1, 3, 4).reshape(
                    (T, H, B*W, C)) + 0.5
                metric = {f"top_{num_vis}_cem": imgs}
                logger.add(metric)
                logger.write()

        return vis_fn

    def make_sample_env_goals_fn(self, config, env):
        sample_env_goals_fn = None
        if config.task == 'hardumazefulldownscale' or 'demofetchpnp' in config.task:

            def sample_env_goals_fn(num_samples):
                all_goals = tf.convert_to_tensor(
                    env.get_goals(), dtype=tf.float32)
                N = len(all_goals)
                goal_ids = tf.random.categorical(
                    tf.math.log([[1/N] * N]), num_samples)

                return tf.gather(all_goals, goal_ids)[0]

        return sample_env_goals_fn

    def make_get_reverse_episode_video_fn(self, config):

        get_reverse_episode_video_fn = None

        def convert(value):

            value = np.array(value)

            if np.issubdtype(value.dtype, np.floating):
                return value.astype(np.float32)

            elif np.issubdtype(value.dtype, np.signedinteger):
                return value.astype(np.int32)

            elif np.issubdtype(value.dtype, np.uint8):
                return value.astype(np.uint8)

            return value

        if 'demofetchpnp' in config.task:

            def get_reverse_episode_video_fn(ep, agent, env, reverse_action_source='rac'):

                def set_env_to_last_obs(ep, env):

                    obs = ep['observation'][-1]

                    def unnorm_ob(ob):
                        return env.obs_min + ob * (env.obs_max - env.obs_min)

                    sim = env.sim

                    sites_offset = (sim.data.site_xpos - sim.model.site_pos)
                    site_id = sim.model.site_name2id('gripper_site')

                    obs = unnorm_ob(obs)
                    grip_pos = obs[:3]
                    gripper_state = obs[3:5]
                    all_obj_pos = np.split(obs[5:5+3*env.n], env.n)

                    sim.model.site_pos[site_id] = grip_pos - \
                        sites_offset[site_id]

                    for i, pos in enumerate(all_obj_pos):
                        sim.data.set_joint_qpos(f"object{i}:joint", [
                                                *pos, *[1, 0, 0, 0]])

                    sim.forward()

                set_env_to_last_obs(ep, env)

                if reverse_action_source == 'rac':
                    action = tf.nest.map_structure(tf.tensor, ep['action'])
                    reverse_action = agent._task_behavior.reverse_action_converter(
                        action).mode()

                reverse_action_list = reverse_action.numpy()

                reverse_action_list = np.flip(reverse_action_list, axis=0)

                reverse_episode = []

                for reverse_action in reverse_action_list:

                    act = {'action': reverse_action}

                    obs = env.step(act)

                    obs = obs() if callable(obs) else obs

                    tran = {k: convert(v) for k, v in {**obs, **act}.items()}

                    tran["label"] = 'Reverse'

                    reverse_episode.append(tran)

                reverse_episode = {k: convert(
                    [t[k] for t in reverse_episode]) for k in reverse_episode[0]}

                reverse_episode['goal'] = ep['goal']

                episode_render_fn = self.make_ep_render_fn(config)

                original_episode_video = episode_render_fn(env, ep, agent)
                reverse_episode_video = episode_render_fn(
                    env, reverse_episode, agent)

                concatenated_video = np.concatenate(
                    (original_episode_video, reverse_episode_video), axis=3)

                return concatenated_video

        elif config.task in {'umazefull', 'umazefulldownscale', 'hardumazefulldownscale'}:

            def get_reverse_episode_video_fn(ep, agent, env, reverse_action_source='rac'):

                def set_env_to_last_obs(ep, env):

                    ant_env = env.maze.wrapped_env
                    ant_env.set_state(ep['goal'][-1][:15], ep['goal'][-1][:14])

                    inner_env = env._env._env._env

                    inner_env.maze.wrapped_env.set_state(
                        ep[agent.state_key][-1][:15], np.zeros_like(ep[agent.state_key][-1][:14]))
                    inner_env.g_xy = ep['goal'][-1][:2]
                    inner_env.maze.wrapped_env.sim.forward()

                set_env_to_last_obs(ep, env)

                if reverse_action_source == 'rac':
                    action = tf.nest.map_structure(tf.tensor, ep['action'])
                    reverse_action = agent._task_behavior.reverse_action_converter(
                        action).mode()

                elif reverse_action_source == 'osp':

                    osp_obs = ep.copy()

                    shifted_observation = tf.map_fn(lambda x: tf.roll(
                        x, shift=1, axis=0), osp_obs[agent.state_key])

                    osp_obs['goal'] = shifted_observation

                    reverse_action = agent._task_behavior.osp_predict(
                        agent.wm, osp_obs)

                reverse_action_list = reverse_action.numpy()

                reverse_action_list = np.flip(reverse_action_list, axis=0)

                reverse_episode = []

                for reverse_action in reverse_action_list:

                    act = {'action': reverse_action}

                    obs = env.step(act)

                    obs = obs() if callable(obs) else obs

                    tran = {k: convert(v) for k, v in {**obs, **act}.items()}

                    tran["label"] = 'Reverse'

                    reverse_episode.append(tran)

                reverse_episode = {k: convert(
                    [t[k] for t in reverse_episode]) for k in reverse_episode[0]}

                reverse_episode['goal'] = ep['goal']

                episode_render_fn = self.make_ep_render_fn(config)

                original_episode_video = episode_render_fn(env, ep)
                reverse_episode_video = episode_render_fn(env, reverse_episode)

                concatenated_video = np.concatenate(
                    (original_episode_video, reverse_episode_video), axis=2)

                return concatenated_video

        elif 'pointmaze' in config.task:

            def get_reverse_episode_video_fn(ep, agent, env, reverse_action_source='rac'):

                def set_env_to_last_obs(ep, env):
                    inner_env = env._env._env._env._env
                    inner_env.g_xy = ep['goal'][-1]
                    inner_env.s_xy = ep['observation'][-1]

                set_env_to_last_obs(ep, env)

                if reverse_action_source == 'rac':
                    action = tf.nest.map_structure(tf.tensor, ep['action'])
                    reverse_action = agent._task_behavior.reverse_action_converter(
                        action).mode()

                elif reverse_action_source == 'osp':

                    osp_obs = ep.copy()

                    shifted_observation = tf.map_fn(lambda x: tf.roll(
                        x, shift=1, axis=0), osp_obs[agent.state_key])

                    osp_obs['goal'] = shifted_observation

                    reverse_action = agent._task_behavior.osp_predict(
                        agent.wm, osp_obs)

                reverse_action_list = reverse_action.numpy()

                reverse_action_list = np.flip(reverse_action_list, axis=0)

                reverse_episode = []

                for reverse_action in reverse_action_list:

                    act = {'action': reverse_action}

                    obs = env.step(act)

                    obs = obs() if callable(obs) else obs

                    tran = {k: convert(v) for k, v in {**obs, **act}.items()}

                    tran["label"] = 'Reverse'

                    reverse_episode.append(tran)

                reverse_episode = {k: convert(
                    [t[k] for t in reverse_episode]) for k in reverse_episode[0]}

                reverse_episode['goal'] = ep['goal']

                episode_render_fn = self.make_ep_render_fn(config)

                original_episode_video = episode_render_fn(env, ep)
                reverse_episode_video = episode_render_fn(env, reverse_episode)

                concatenated_video = np.concatenate(
                    (original_episode_video, reverse_episode_video), axis=2)

                return concatenated_video

        elif 'dmc_walker_walk_proprio' == config.task:

            def get_reverse_episode_video_fn(ep, agent, env, reverse_action_source='rac'):

                def set_env_to_last_obs(ep, env):

                    inner_env = env._env._env._env._env

                    qpos = ep['qpos'][-1]

                    size = inner_env.physics.get_state(
                    ).shape[0] - qpos.shape[0]
                    inner_env.physics.set_state(
                        np.concatenate((qpos, np.zeros([size]))))
                    inner_env.physics.step()

                set_env_to_last_obs(ep, env)

                if reverse_action_source == 'rac':
                    action = tf.nest.map_structure(tf.tensor, ep['action'])
                    reverse_action = agent._task_behavior.reverse_action_converter(
                        action).mode()

                reverse_action_list = reverse_action.numpy()

                reverse_action_list = np.flip(reverse_action_list, axis=0)

                reverse_episode = []

                for reverse_action in reverse_action_list:

                    act = {'action': reverse_action}

                    obs = env.step(act)

                    obs = obs() if callable(obs) else obs

                    tran = {k: convert(v) for k, v in {**obs, **act}.items()}

                    tran["label"] = 'Reverse'

                    reverse_episode.append(tran)

                reverse_episode = {k: convert(
                    [t[k] for t in reverse_episode]) for k in reverse_episode[0]}

                reverse_episode['goal'] = ep['goal']

                episode_render_fn = self.make_ep_render_fn(config)

                original_episode_video = episode_render_fn(env, ep)
                reverse_episode_video = episode_render_fn(env, reverse_episode)

                concatenated_video = np.concatenate(
                    (original_episode_video, reverse_episode_video), axis=2)

                return concatenated_video

        return get_reverse_episode_video_fn

    def train(self, env, eval_env, eval_fn, ep_render_fn, images_render_fn, space_explored_plot_fn, cem_vis_fn, obs2goal_fn, sample_env_goals_fn, config):

        logdir = pathlib.Path(config.logdir).expanduser()
        logdir.mkdir(parents=True, exist_ok=True)
        config.save(logdir / 'config.yaml')
        print(config, '\n')
        print('Logdir: ', logdir)

        if 'pointmaze' in config.task:
            video_fps = 10
        else:
            video_fps = 20

        outputs = [
            common.TerminalOutput(),
            common.JSONLOutput(config.logdir),
            common.TensorBoardOutput(config.logdir, video_fps),
        ]

        replay = common.Replay(logdir / 'train_episodes', **config.replay)

        eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
            capacity=config.replay.capacity // 10,
            minlen=config.dataset.length,
            maxlen=config.dataset.length))

        step = common.Counter(replay.stats['total_steps'])

        num_eps = common.Counter(replay.stats['total_episodes'])

        num_algo_updates = common.Counter(0)

        logger = common.Logger(step, outputs, multiplier=config.action_repeat)

        metrics = collections.defaultdict(list)

        should_train = common.Every(config.train_every)

        should_update_cluster = common.Every(config.cluster['update_every'])

        should_report = common.Every(config.report_every)

        should_video_train = common.Every(config.eval_every)

        should_eval = common.Every(config.eval_every)

        should_ckpt = common.Every(config.ckpt_every)

        should_goal_update = common.Every(config.goal_update_every)

        should_gcp_rollout = common.Every(config.gcp_rollout_every)

        should_exp_rollout = common.Every(config.exp_rollout_every)

        should_two_policy_rollout = common.Every(
            config.two_policy_rollout_every)

        should_cem_plot = common.Every(config.eval_every)

        if config.if_egc_env_sample:
            should_env_gcp_rollout = common.Every(config.env_gcp_rollout_every)

        self.next_ep_video = False

        def per_episode(ep, mode):

            length = len(ep['reward']) - 1
            score = float(ep['reward'].astype(np.float64).sum())
            print(
                f'{mode.title()} episode has {length} steps and return {score:.1f}.')
            logger.scalar(f'{mode}_return', score)
            logger.scalar(f'{mode}_length', length)

            for key, value in ep.items():
                if re.match(config.log_keys_sum, key):
                    logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
                if re.match(config.log_keys_mean, key):
                    logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
                if re.match(config.log_keys_max, key):
                    logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())

            if (should_video_train(num_eps) or self.next_ep_video) and mode == 'train':

                if self.next_ep_video:
                    self.next_ep_video = False
                else:
                    self.next_ep_video = True

                if ep_render_fn is None and 'none' not in config.log_keys_video:
                    for key in config.log_keys_video:
                        logger.video(f'{mode}_policy_{key}', ep[key])

                elif ep_render_fn is not None:

                    video = ep_render_fn(eval_env, ep)
                    if video is not None:
                        label = ep['label'][0]
                        logger.video(
                            f'{mode}_policy_{config.state_key}_{label}', video)

            _replay = dict(train=replay, eval=eval_replay)[mode]
            logger.add(_replay.stats, prefix=mode)
            logger.write()

        driver = common.GCDriver([env], config.goal_key, config)

        driver.on_episode(lambda ep: per_episode(ep, mode='train'))
        driver.on_episode(lambda ep: num_eps.increment())

        driver.on_step(lambda tran, worker: step.increment())
        driver.on_step(replay.add_step)
        driver.on_reset(replay.add_step)

        eval_driver = common.GCDriver([eval_env], config.goal_key, config)
        eval_driver.if_eval_driver = True
        eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
        eval_driver.on_episode(eval_replay.add_episode)

        if config.if_assign_centroids:
            centroids_sample_driver = common.GCDriver(
                [eval_env], config.goal_key, config)

        print('Create agent.')

        agnt = gc_agent.GCAgent(
            config, env.obs_space, env.act_space, step, obs2goal_fn, sample_env_goals_fn)

        should_space_explored_plot = common.Every(config.eval_every)

        def space_explored_plot():
            if should_space_explored_plot(num_eps) and space_explored_plot_fn != None:
                from time import time
                plt.cla()
                plt.clf()
                start = time()
                space_explored_plot_fn(eval_env, agnt=agnt, complete_episodes=replay._complete_eps,
                                       logger=logger, ep_subsample=1, step_subsample=1)
                print("plotting took ", time() - start)
                logger.write()
        driver.on_episode(lambda ep: space_explored_plot())

        should_reverse_episode_video = common.Every(config.eval_every)
        reverse_episode_video_fn = self.make_get_reverse_episode_video_fn(
            config)

        self.next_ep_reverse_video = False

        def reverse_episode_video_plot(ep):
            if (should_reverse_episode_video(num_eps) or self.next_ep_reverse_video) and reverse_episode_video_fn != None:

                if self.next_ep_reverse_video:
                    self.next_ep_reverse_video = False
                else:
                    self.next_ep_reverse_video = True

                video = reverse_episode_video_fn(
                    ep, agnt, eval_env, reverse_action_source='rac')
                if video is not None:
                    label = ep['label'][0]
                    logger.video(f'Reverse_{label}_from_rac', video)

                video = reverse_episode_video_fn(
                    ep, agnt, eval_env, reverse_action_source='osp')
                if video is not None:
                    label = ep['label'][0]
                    logger.video(f'Reverse_{label}_from_osp', video)

        if config.if_reverse_episode_video_plot:

            driver.on_episode(lambda ep: reverse_episode_video_plot(ep))

        prefill = max(0, config.prefill - replay.stats['total_episodes'])

        random_agent = common.RandomAgent(env.act_space)
        if prefill:
            print(f'Prefill dataset ({prefill} episodes).')
            driver(random_agent, episodes=prefill)
            driver.reset()

        dataset = iter(replay.dataset(**config.dataset))

        if config.if_egc_env_sample:
            if 'demofetchpnp' in config.task and config.if_multi_3_blcok_training_goal:
                self.cluster_assign_egc_goal_index = 1
                egc_dataset_1 = iter(replay.recent_dataset_specific_label(
                    **config.dataset, label='egc1'))
                egc_dataset_2 = iter(replay.recent_dataset_specific_label(
                    **config.dataset, label='egc2'))
                egc_dataset_3 = iter(replay.recent_dataset_specific_label(
                    **config.dataset, label='egc3'))

            else:
                egc_dataset = iter(replay.recent_dataset_specific_label(
                    **config.dataset, label='egc'))

        if config.gcp_train_factor > 1:
            gcp_dataset = iter(replay.dataset(**config.dataset))

        if config.replay.sample_recent:
            recent_dataset = iter(replay.recent_dataset(**config.dataset))

            if config.gcp_train_factor > 1:

                recent_gcp_dataset = iter(
                    replay.recent_dataset(**config.dataset))

        report_dataset = iter(replay.dataset(**config.dataset))

        train_agent = common.CarryOverState(agnt.train)

        train_gcp = common.CarryOverState(agnt.train_gcp)

        train_agent(next(dataset))

        if (logdir / 'variables.pkl').exists():
            print('Found existing checkpoint.')
            agnt.agent_load(logdir)
        else:
            print('Pretrain agent.')
            for _ in range(config.pretrain):
                for i in range(config.gcp_train_factor - 1):
                    train_gcp(next(gcp_dataset))
                train_agent(next(dataset))

        def train_step(tran, worker):

            if should_train(step):

                for _ in range(config.train_steps):

                    _data = next(dataset)

                    if config.train_cluster_use_Normal:
                        mets = train_agent(_data, train_cluster=True)
                    else:
                        mets = train_agent(_data)

                    [metrics[key].append(value) for key, value in mets.items()]

                    if config.if_egc_env_sample and config.train_cluster_use_egc:

                        if 'demofetchpnp' in config.task and config.if_multi_3_blcok_training_goal:

                            if self.cluster_assign_egc_goal_index == 1:

                                _egc_data = next(egc_dataset_1)

                            elif self.cluster_assign_egc_goal_index == 2:

                                _egc_data = next(egc_dataset_2)

                            elif self.cluster_assign_egc_goal_index == 3:

                                _egc_data = next(egc_dataset_3)

                            else:
                                raise ValueError(
                                    "Wrong value of self.cluster_assign_egc_goal_index!")

                        else:
                            _egc_data = next(egc_dataset)

                        mets = train_agent(_egc_data, train_cluster=True)

                    for i in range(config.gcp_train_factor - 1):
                        mets = train_gcp(next(gcp_dataset))
                        [metrics[key].append(value)
                         for key, value in mets.items()]

                    if config.replay.sample_recent:
                        _data = next(recent_dataset)
                        mets = train_agent(_data)
                        [metrics[key].append(value)
                         for key, value in mets.items()]

                        for i in range(config.gcp_train_factor - 1):
                            mets = train_gcp(next(recent_gcp_dataset))
                            [metrics[key].append(value)
                             for key, value in mets.items()]

            if should_report(step):
                for name, values in metrics.items():
                    logger.scalar(name, np.array(values, np.float64).mean())
                    metrics[name].clear()

                logger.add(agnt.report(next(report_dataset), eval_env))
                logger.write(fps=True)

        if config.if_actor_gs:
            train_gcpolicy = partial(agnt.policy_gs, mode='train')
            eval_gcpolicy = partial(agnt.policy_gs, mode='eval')
        else:
            train_gcpolicy = partial(agnt.policy, mode='train')
            eval_gcpolicy = partial(agnt.policy, mode='eval')

        driver.on_step(train_step)

        def eval_agent():
            if should_eval(num_eps):
                print('Start evaluation.')
                eval_fn(eval_driver, eval_gcpolicy, logger)
                agnt.agent_save(logdir)

                if images_render_fn is not None and config.goal_strategy == "Cluster_goal_Planner":

                    def get_decoded_centroids(agnt, config):

                        centroids = agnt.wm.cluster.centroids()

                        centroids = tf.convert_to_tensor(
                            centroids.numpy(), dtype=agnt.wm.dtype)

                        state = None
                        if state is None:
                            latent = agnt.wm.rssm.initial(
                                config.cluster['n_latent_landmarks'])
                            action = tf.zeros(
                                (config.cluster['n_latent_landmarks'],) + agnt.act_space.shape)
                            state = latent, action

                        latent, action = state

                        latent, _ = agnt.wm.rssm.obs_step(
                            latent, action, centroids, True, True)

                        decoder = agnt.wm.heads['decoder']

                        feat = agnt.wm.rssm.get_feat(latent)

                        centroids_decoded_dist = decoder(feat)

                        centroids_decoded = centroids_decoded_dist[agnt.state_key].mean(
                        )

                        centroids_decoded = centroids_decoded.numpy()

                        return centroids_decoded

                    decoded_centroids_list = get_decoded_centroids(
                        agnt, config)
                    all_images = images_render_fn(
                        eval_env, decoded_centroids_list)
                    logger.image('Cluster Centroids', all_images)

            if should_ckpt(num_eps):
                print('Checkpointing.')
                agnt.agent_save(logdir)

        def vis_fn(elite_inds, elite_samples, seq, wm):
            if should_cem_plot(num_eps) and cem_vis_fn is not None:
                cem_vis_fn(elite_inds, elite_samples,
                           seq, wm, eval_env, logger)

        print("Goal Pick Strategy for Exploration: ", config.goal_strategy)
        my_GC_goal_picker = gc_goal_picker.GC_goal_picker(
            config, agnt, replay, dataset, env, obs2goal_fn, sample_env_goals_fn, vis_fn)

        get_goal_fn = my_GC_goal_picker.get_goal_fn

        def assign_cluster_centrods():

            print("==========================================================clusters update==========================================================")

            if config.if_assign_use_batch:

                if config.centrods_assign_strategy == 'egc':

                    if 'demofetchpnp' in config.task and config.if_multi_3_blcok_training_goal:

                        self.cluster_assign_egc_goal_index = random.randint(
                            1, 3)

                        if self.cluster_assign_egc_goal_index == 1:

                            _egc_data = next(egc_dataset_1)

                        elif self.cluster_assign_egc_goal_index == 2:

                            _egc_data = next(egc_dataset_2)

                        elif self.cluster_assign_egc_goal_index == 3:

                            _egc_data = next(egc_dataset_3)

                        else:
                            raise ValueError(
                                "Wrong value of self.cluster_assign_egc_goal_index!")

                    else:
                        _egc_data = next(egc_dataset)

                    _egc_data[agnt.state_key] = _egc_data[agnt.state_key].numpy()
                    _egc_data[agnt.state_key] = _egc_data[agnt.state_key].reshape(
                        -1, _egc_data[agnt.state_key].shape[-1])
                    ep = _egc_data

                elif config.centrods_assign_strategy == 'exp':

                    _data = next(dataset)

                    _data[agnt.state_key] = _data[agnt.state_key].numpy()
                    _data[agnt.state_key] = _data[agnt.state_key].reshape(
                        -1, _data[agnt.state_key].shape[-1])
                    ep = _data

            else:

                centroids_sample_driver.reset()

                if config.centrods_assign_strategy == 'egc':

                    if 'demofetchpnp' in config.task and config.if_multi_3_blcok_training_goal:

                        centroids_sample_driver(
                            eval_gcpolicy, if_multi_3_blcok_training_goal=config.if_multi_3_blcok_training_goal, episodes=1, label='egc')

                        self.cluster_assign_egc_goal_index = centroids_sample_driver.training_goal_index

                    else:
                        centroids_sample_driver(
                            eval_gcpolicy, episodes=1, label='egc')

                elif config.centrods_assign_strategy == 'exp':

                    centroids_sample_driver(train_gcpolicy, expl_policy, get_goal_fn, episodes=1,
                                            goal_time_limit=goal_time_limit, goal_checker=temporal_dist)

                ep = centroids_sample_driver._eps[0]

                ep = {k: centroids_sample_driver._convert(
                    [t[k] for t in ep]) for k in ep[0]}

            agnt.wm.assign_cluster_centroids(
                data=ep, space=config.centroids_assign_space)

        def update_goal_strategy(*args):
            if should_goal_update(num_eps):
                if config.goal_strategy == "Greedy":
                    my_GC_goal_picker.goal_strategy.update_buffer_priorities()

                elif config.goal_strategy in {"MEGA", "Skewfit"}:
                    my_GC_goal_picker.goal_strategy.update_kde()

                elif config.goal_strategy == "SubgoalPlanner":

                    my_GC_goal_picker.goal_strategy.will_update_next_call = True

                elif config.goal_strategy == "Cluster_goal_Planner" and config.if_assign_centroids:
                    assign_cluster_centrods()

        driver.on_episode(lambda ep: update_goal_strategy())

        goal_time_limit = round(
            config.goal_policy_rollout_percentage * config.time_limit)

        def temporal_dist(obs):

            obs = tf.nest.map_structure(
                lambda x: tf.expand_dims(tf.tensor(x), 0), obs)[0]
            dist = agnt.temporal_dist(obs).numpy().item()
            success = dist < config.subgoal_threshold
            metric = {"subgoal_dist": dist, "subgoal_success": float(success)}
            return success, metric

        def expl_policy(obs, state, **kwargs):

            actions, state = agnt.expl_policy(obs, state, mode='train')

            if config.go_expl_rand_ac:
                actions, _ = random_agent(obs)

            return actions, state

        demo = False
        demo_path = '/common/home/yd374/ACH_Server/Experiment/Maniskill_PegInsert_demos/v0/rigid_body/PegInsertionSide-v0/trajectory.state.pd_joint_delta_pos.h5'

        while step < config.steps:

            logger.write()

            """ 1. train: run goal cond. policy for entire rollout"""
            if should_gcp_rollout(num_algo_updates):
                driver(train_gcpolicy, get_goal=get_goal_fn, episodes=1)
                eval_agent()

            """ 2. expl: run expl policy for entire rollout """
            if should_exp_rollout(num_algo_updates):
                driver(expl_policy, episodes=1)
                eval_agent()

            """ 3. 2pol: run goal cond. and then expl policy."""
            if should_two_policy_rollout(num_algo_updates):

                driver(train_gcpolicy, expl_policy, get_goal_fn, episodes=1,
                       goal_time_limit=goal_time_limit, goal_checker=temporal_dist)
                eval_agent()

            if config.if_egc_env_sample and should_env_gcp_rollout(num_algo_updates):

                if 'demofetchpnp' in config.task and config.if_multi_3_blcok_training_goal:

                    driver(eval_gcpolicy, episodes=3,
                           if_multi_3_blcok_training_goal=config.if_multi_3_blcok_training_goal, label='egc')

                else:

                    if config.task == "PegInsertionSide-v0" and demo:

                        if_use_demo_prob = random.random()

                        if if_use_demo_prob > step.value / config.steps:
                            if_use_demo = True
                        else:
                            if_use_demo = False

                        driver(eval_gcpolicy, episodes=1, label='egc',
                               if_use_demo=if_use_demo, demo_path=demo_path)

                    else:
                        driver(eval_gcpolicy, episodes=1, label='egc')

                eval_agent()

            num_algo_updates.increment()

    def main(self, gpu_index=0):
        """
        Pass in the config setting(s) you want from the configs.yaml. If there are multiple
        configs, we will override previous configs with later ones, like if you want to add
        debug mode to your environment.

        To override specific config keys, pass them in with --key value.

        python examples/run_goal_cond.py --configs <setting 1> <setting 2> ... --foo bar

        Examples:
            Normal scenario
                python examples/run_goal_cond.py --configs mega_fetchpnp_proprio
            Debug scenario
                python examples/run_goal_cond.py --configs mega_fetchpnp_proprio debug
            Override scenario
                python examples/run_goal_cond.py --configs mega_fetchpnp_proprio --seed 123
        """

        config = self.Set_Config()

        self.config = config

        env = self.make_env(config)
        eval_env = self.make_env(config, if_eval=True)

        obs = env.reset()

        sample_env_goals_fn = self.make_sample_env_goals_fn(config, eval_env)
        eval_fn = self.make_eval_fn(config)
        ep_render_fn = self.make_ep_render_fn(config)

        space_explored_plot_fn = self.make_space_explored_plot_fn(config)
        cem_vis_fn = self.make_cem_vis_fn(config)
        obs2goal_fn = self.make_obs2goal_fn(config)
        images_render_fn = self.make_images_render_fn(config)

        tf.config.run_functions_eagerly(not config.jit)

        message = 'No GPU found. To actually train on CPU remove this assert.'
        assert tf.config.experimental.list_physical_devices('GPU'), message

        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)

        if gpus:

            tf.config.experimental.set_visible_devices(gpus[gpu_index], 'GPU')
            print(f"Using GPU: {gpus[gpu_index].name}")
        else:
            print("No GPU devices found")

        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        assert config.precision in (16, 32), config.precision
        if config.precision == 16:
            from tensorflow.keras.mixed_precision import experimental as prec
            prec.set_policy(prec.Policy('mixed_float16'))

        self.train(env, eval_env, eval_fn, ep_render_fn, images_render_fn,
                   space_explored_plot_fn, cem_vis_fn, obs2goal_fn, sample_env_goals_fn, config)


if __name__ == "__main__":

    My_PEG = CE2()
    My_PEG.main(5)  # set the gpu index
