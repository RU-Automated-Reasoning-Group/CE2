from collections import defaultdict
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
from time import time
import random
import heapq


class GC_goal_picker:

    def __init__(self, config, agnt, replay, dataset, env, obs2goal_fn, sample_env_goals_fn, vis_fn):

        if config.goal_strategy == "Greedy":
            goal_strategy = Greedy(
                replay, agnt.wm, agnt._expl_behavior._intr_reward, config.state_key, config.goal_key, 1000)

        elif config.goal_strategy == "SampleReplay":
            goal_strategy = SampleReplay(
                agnt.wm, dataset, config.state_key, config.goal_key)

        elif config.goal_strategy == "MEGA":
            goal_strategy = MEGA(config, agnt, replay, env.act_space,
                                 config.state_key, config.time_limit, obs2goal_fn)

        elif config.goal_strategy == "Skewfit":
            goal_strategy = Skewfit(
                agnt, replay, env.act_space, config.state_key, config.time_limit, obs2goal_fn)

        elif config.goal_strategy == "SubgoalPlanner":

            goal_strategy = SubgoalPlanner(
                agnt,
                config,
                env,
                replay,
                obs2goal_fn=obs2goal_fn,
                sample_env_goals_fn=sample_env_goals_fn,
                vis_fn=vis_fn
            )

        elif config.goal_strategy == "Cluster_goal_Planner":

            goal_strategy = Cluster_goal_Planner(
                agnt,
                config,
                env,
                obs2goal_fn=obs2goal_fn,
            )

        else:
            raise NotImplementedError

        self.goal_strategy = goal_strategy

        self.get_goal_fn = self.make_get_goal_fn(
            config, agnt, sample_env_goals_fn)

    def make_get_goal_fn(self, config, agnt, sample_env_goals_fn):

        def get_goal(obs, state=None, mode='train'):

            obs = tf.nest.map_structure(lambda x: tf.expand_dims(
                tf.expand_dims(tf.tensor(x), 0), 0), obs)[0]
            obs = agnt.wm.preprocess(obs)
            if np.random.uniform() < config.planner.sample_env_goal_percent:
                goal = sample_env_goals_fn(1)
                return tf.squeeze(goal)

            if config.goal_strategy == "Greedy":
                goal = self.goal_strategy.get_goal()
                self.goal_strategy.will_update_next_call = False
            elif config.goal_strategy == "SampleReplay":
                goal = self.goal_strategy.get_goal(obs)
            elif config.goal_strategy in {"MEGA", "Skewfit"}:
                goal = self.goal_strategy.sample_goal(obs, state)
            elif config.goal_strategy == "SubgoalPlanner":
                goal = self.goal_strategy.search_goal(obs, state)
                self.goal_strategy.will_update_next_call = False
            elif config.goal_strategy == "Cluster_goal_Planner":
                goal = self.goal_strategy.search_goal(obs, state)
            else:
                raise NotImplementedError
            return tf.squeeze(goal)

        return get_goal


class Greedy:
    def __init__(self, replay, wm, reward_fn, state_key, goal_key, batch_size, topk=10, exp_weight=1.0):
        self.replay = replay
        self.wm = wm
        self.reward_fn = reward_fn
        self.state_key = state_key
        self.goal_key = goal_key
        self.batch_size = batch_size
        self.topk = topk
        self.exp_weight = exp_weight
        self.all_topk_states = None

    def update_buffer_priorities(self):
        start = time()

        @tf.function
        def process_batch(data, reward_fn):
            data = self.wm.preprocess(data)
            states = data[self.state_key]

            embed = self.wm.encoder(data)
            post, prior = self.wm.rssm.observe(
                embed, data['action'], data['is_first'], state=None)
            data['feat'] = self.wm.rssm.get_feat(post)

            reward = reward_fn(data).reshape((-1,))
            values, indices = tf.math.top_k(reward, self.topk)
            states = data[self.state_key].reshape(
                (-1, data[self.state_key].shape[-1]))
            topk_states = tf.gather(states, indices)

            return values, topk_states

        self.all_topk_states = []

        num_episodes = len(self.replay._complete_eps)
        chunk = defaultdict(list)
        count = 0
        for idx, ep_dict in enumerate(self.replay._complete_eps.values()):
            for k, v in ep_dict.items():
                chunk[k].append(v)
            count += 1
            if count >= self.batch_size or idx == num_episodes-1:
                count = 0
                data = {k: np.stack(v) for k, v in chunk.items()}

                chunk = defaultdict(list)

                values, top_states = process_batch(data, self.reward_fn)
                values_states = [(v, s) for v, s in zip(values, top_states)]
                self.all_topk_states.extend(values_states)
                self.all_topk_states.sort(key=lambda x: x[0], reverse=True)
                self.all_topk_states = self.all_topk_states[:self.topk]
        end = time() - start
        print("update buffer took", end)

    def get_goal(self):
        if self.all_topk_states is None:
            self.update_buffer_priorities()

        priorities = np.asarray([x[0] for x in self.all_topk_states])
        priorities += 1e-6
        np.exp(priorities * self.exp_weight)
        prob = np.squeeze(priorities) / priorities.sum()

        idx = np.random.choice(len(self.all_topk_states),
                               1, replace=True, p=prob)[0]
        value, state = self.all_topk_states[idx]
        return state.numpy()


class SampleReplay:
    def __init__(self, wm, dataset, state_key, goal_key):
        self.state_key = state_key
        self.goal_key = goal_key
        self._dataset = dataset
        self.wm = wm

    @tf.function
    def get_goal(self, obs):
        random_batch = next(self._dataset)
        random_batch = self.wm.preprocess(random_batch)
        random_goals = tf.reshape(
            random_batch[self.state_key], (-1,) + tuple(random_batch[self.state_key].shape[2:]))
        return random_goals[:obs[self.state_key].shape[0]]


class MEGA:
    def __init__(self, config, agnt, replay, act_space, state_key, ep_length, obs2goal_fn, goal_sample_fn=None):

        self.agnt = agnt
        self.config = config
        self.wm = agnt.wm
        self.obs2goal_fn = obs2goal_fn
        self.dtype = agnt.wm.dtype
        if self.config.if_actor_gs:
            self.actor = agnt._task_behavior.actor_gs
        else:
            self.actor = agnt._task_behavior.actor

        if self.config.expl_behavior == 'Plan2Explore':
            self.reward_fn = agnt._expl_behavior.planner_intr_reward

        p_cfg = config.planner
        self.min_goal = np.array(p_cfg.goal_min, dtype=np.float32)
        self.max_goal = np.array(p_cfg.goal_max, dtype=np.float32)
        self.planner = p_cfg.planner_type
        self.horizon = p_cfg.horizon
        self.batch = p_cfg.batch
        self.cem_elite_ratio = p_cfg.cem_elite_ratio
        self.optimization_steps = p_cfg.optimization_steps
        self.std_scale = p_cfg.std_scale
        self.mppi_gamma = p_cfg.mppi_gamma
        self.evaluate_only = p_cfg.evaluate_only
        self.repeat_samples = p_cfg.repeat_samples
        self.env_goals_percentage = p_cfg.init_env_goal_percent

        self.act_space = act_space
        if isinstance(self.act_space, dict):
            self.act_space = self.act_space['action']

        self.gc_input = config.gc_input
        self.state_key = config.state_key

        state = None
        if state is None:
            self.initial_latent = self.wm.rssm.initial(1)
            self.initial_action = tf.zeros((1,) + self.act_space.shape)

        self.decoder = self.wm.heads['decoder']

        self.replay = replay
        self.goal_sample_fn = goal_sample_fn

        self.dataset = iter(replay.dataset(batch=10, length=ep_length))

        from sklearn.neighbors import KernelDensity
        self.alpha = -1.0
        self.kernel = 'gaussian'
        self.bandwidth = 0.1
        self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        self.kde_sample_mean = 0.
        self.kde_sample_std = 1.

        self.state_key = state_key
        self.ready = False
        self.random = False
        self.ep_length = ep_length
        self.obs2goal = obs2goal_fn

        self.if_exploration_potential_filter = True
        self.if_eval_fitness = self.if_exploration_potential_filter
        self.elite_num = 10

    def update_kde(self):
        self.ready = True

        num_episodes = self.replay.stats['loaded_episodes']

        num_samples = min(10000, self.replay.stats['loaded_steps'])

        ep_idx = np.random.randint(0, num_episodes, num_samples)

        t_idx = np.random.randint(0, self.ep_length, num_samples)

        all_episodes = list(self.replay._complete_eps.values())
        if self.obs2goal is None:
            kde_samples = [all_episodes[e][self.state_key][t]
                           for e, t in zip(ep_idx, t_idx)]
        else:
            kde_samples = [self.obs2goal(
                all_episodes[e][self.state_key][t]) for e, t in zip(ep_idx, t_idx)]

        self.kde_sample_mean = np.mean(kde_samples, axis=0, keepdims=True)
        self.kde_sample_std = np.std(kde_samples, axis=0, keepdims=True) + 1e-4
        kde_samples = (kde_samples - self.kde_sample_mean) / \
            self.kde_sample_std

        self.fitted_kde = self.kde.fit(kde_samples)

    def evaluate_log_density(self, samples):
        assert self.ready, "ENSURE READY BEFORE EVALUATING LOG DENSITY"
        return self.fitted_kde.score_samples((samples - self.kde_sample_mean) / self.kde_sample_std)

    def sample_goal(self, obs, state=None, mode='train'):
        if not self.ready:
            self.update_kde()
        if self.goal_sample_fn:
            num_samples = 10000
            sampled_ags = self.goal_sample_fn(num_samples)
        else:

            num_episodes = self.replay.stats['loaded_episodes']

            num_samples = min(10000, self.replay.stats['loaded_steps'])

            ep_idx = np.random.randint(0, num_episodes, num_samples)

            t_idx = np.random.randint(0, self.ep_length, num_samples)

            all_episodes = list(self.replay._complete_eps.values())
            if self.obs2goal is None:
                sampled_ags = np.asarray(
                    [all_episodes[e][self.state_key][t] for e, t in zip(ep_idx, t_idx)])
            else:
                sampled_ags = np.asarray([self.obs2goal(
                    all_episodes[e][self.state_key][t]) for e, t in zip(ep_idx, t_idx)])

            if self.obs2goal is not None:
                sampled_ags = self.obs2goal(sampled_ags)

        q_values = None
        bad_q_idxs = None

        sampled_ag_scores = self.evaluate_log_density(sampled_ags)

        normalized_inverse_densities = softmax(sampled_ag_scores * self.alpha)
        normalized_inverse_densities *= -1.
        goal_values = normalized_inverse_densities

        if q_values is not None:
            goal_values[bad_q_idxs] = q_values[bad_q_idxs] * -1e-8

        if self.random:
            abs_goal_values = np.abs(goal_values)
            normalized_values = abs_goal_values / \
                np.sum(abs_goal_values, axis=0, keepdims=True)

            chosen_idx = np.random.choice(
                len(abs_goal_values), 1, replace=True, p=normalized_values)[0]
        else:

            if self.if_exploration_potential_filter:

                if state is None:
                    latent = self.wm.rssm.initial(1)
                    action = tf.zeros((1, 1,) + self.act_space.shape)
                    state = latent, action

                else:
                    latent, action = state
                    action = tf.expand_dims(action, 0)

                embed = self.wm.encoder(obs)

                post, prior = self.wm.rssm.observe(
                    embed, action, obs['is_first'], latent)

                init_start = {k: v[:, -1] for k, v in post.items()}

                @tf.function
                def eval_fitness(goal):

                    start = {k: v for k, v in init_start.items()}
                    start['feat'] = self.wm.rssm.get_feat(start)

                    start = tf.nest.map_structure(
                        lambda x: tf.repeat(x, goal.shape[0], 0), start)

                    if self.gc_input == "state" or self.config.if_actor_gs:
                        goal_input = tf.cast(goal, self.dtype)

                    elif self.gc_input == "embed":
                        goal_obs = start.copy()
                        goal_obs[self.state_key] = goal
                        goal_input = self.wm.encoder(goal_obs)

                    actor_inp = tf.concat([start['feat'], goal_input], -1)

                    start['action'] = tf.zeros_like(
                        self.actor(actor_inp).mode())
                    seq = {k: [v] for k, v in start.items()}

                    for _ in range(self.horizon):
                        actor_inp = tf.concat(
                            [seq['feat'][-1], goal_input], -1)
                        action = self.actor(actor_inp).sample()
                        state = self.wm.rssm.img_step(
                            {k: v[-1] for k, v in seq.items()}, action)
                        feat = self.wm.rssm.get_feat(state)
                        for key, value in {**state, 'action': action, 'feat': feat}.items():
                            seq[key].append(value)

                    seq = {k: tf.stack(v, 0) for k, v in seq.items()}

                    rewards = self.reward_fn(seq)

                    returns = tf.reduce_sum(rewards, 0)

                    return returns, seq

                min_n_indices = heapq.nsmallest(self.elite_num, range(
                    len(goal_values)), key=goal_values.__getitem__)

                elites = [sampled_ags[i] for i in min_n_indices]
                elites = np.array(elites)

                samples = tf.convert_to_tensor(elites, dtype=self.dtype)

                if self.config.expl_behavior == 'Plan2Explore' and self.if_eval_fitness:

                    fitness, seq = eval_fitness(samples)

                    weights = tf.expand_dims(tf.nn.softmax(
                        self.mppi_gamma * fitness), axis=1)

                    max_indices = int(tf.argmax(weights).numpy())

                    explore_goal = samples[max_indices]

                else:
                    explore_goal = random.choice(samples)

            else:

                chosen_idx = np.argmin(goal_values)

                explore_goal = sampled_ags[chosen_idx]

        self.sampled_ags = sampled_ags
        self.goal_values = goal_values

        return explore_goal


class Skewfit(MEGA):
    def __init__(self, agent, replay, act_space, state_key, ep_length, obs2goal_fn, goal_sample_fn):
        super().__init__(agent, replay, act_space, state_key,
                         ep_length, obs2goal_fn, goal_sample_fn)
        self.random = True


class SubgoalPlanner:

    def __init__(
            self,
            agnt,
            config,
            env,
            replay,
            obs2goal_fn=None,
            sample_env_goals_fn=None,
            vis_fn=None,):

        self.wm = agnt.wm
        self.config = config
        self.dtype = agnt.wm.dtype
        if self.config.if_actor_gs:
            self.actor = agnt._task_behavior.actor_gs
        else:
            self.actor = agnt._task_behavior.actor
        self.reward_fn = agnt._expl_behavior.planner_intr_reward

        p_cfg = config.planner
        self.min_goal = np.array(p_cfg.goal_min, dtype=np.float32)
        self.max_goal = np.array(p_cfg.goal_max, dtype=np.float32)
        self.planner = p_cfg.planner_type
        self.horizon = p_cfg.horizon
        self.batch = p_cfg.batch
        self.cem_elite_ratio = p_cfg.cem_elite_ratio
        self.optimization_steps = p_cfg.optimization_steps
        self.std_scale = p_cfg.std_scale
        self.mppi_gamma = p_cfg.mppi_gamma
        self.evaluate_only = p_cfg.evaluate_only
        self.repeat_samples = p_cfg.repeat_samples
        self.env_goals_percentage = p_cfg.init_env_goal_percent

        self.goal_dim = np.prod(env.obs_space[config.goal_key].shape)
        self.act_space = env.act_space
        if isinstance(self.act_space, dict):
            self.act_space = self.act_space['action']
        self.obs2goal = obs2goal_fn
        self.sample_env_goals = self.env_goals_percentage > 0
        self.sample_env_goals_fn = sample_env_goals_fn if self.sample_env_goals else None

        self.gc_input = config.gc_input
        self.state_key = config.state_key

        self.vis_fn = vis_fn
        self.will_update_next_call = True

        if p_cfg.init_candidates[0] == 123456789.0:
            init_cand = None
        else:
            init_cand = np.array(p_cfg.init_candidates, dtype=np.float32)

            goal_dim = np.prod(env.obs_space[config.state_key].shape)
            assert len(init_cand) == goal_dim, f"{len(init_cand)}, {goal_dim}"
            init_cand = np.split(init_cand, len(init_cand)//goal_dim)
            init_cand = tf.convert_to_tensor(init_cand)

        self.init_distribution = None
        if init_cand is not None:
            self.create_init_distribution(init_cand)

        goal_dataset = None

        if p_cfg.sample_replay:

            goal_dataset = iter(replay.dataset(
                batch=10000//(config.time_limit+1), length=config.time_limit+1))

        self.dataset = goal_dataset
        if self.evaluate_only:
            assert self.dataset is not None, "need to sample from replay buffer."

    def search_goal(self, obs, state=None):

        if self.will_update_next_call is False:
            return self.sample_goal()

        elite_size = int(self.batch * self.cem_elite_ratio)

        if state is None:
            latent = self.wm.rssm.initial(1)
            action = tf.zeros((1, 1,) + self.act_space.shape)
            state = latent, action

        else:
            latent, action = state
            action = tf.expand_dims(action, 0)

        embed = self.wm.encoder(obs)

        post, prior = self.wm.rssm.observe(
            embed, action, obs['is_first'], latent)

        init_start = {k: v[:, -1] for k, v in post.items()}

        @tf.function
        def eval_fitness(goal):

            start = {k: v for k, v in init_start.items()}
            start['feat'] = self.wm.rssm.get_feat(start)

            start = tf.nest.map_structure(
                lambda x: tf.repeat(x, goal.shape[0], 0), start)

            if self.gc_input == "state" or self.config.if_actor_gs:
                goal_input = tf.cast(goal, self.dtype)

            elif self.gc_input == "embed":
                goal_obs = start.copy()
                goal_obs[self.state_key] = goal
                goal_input = self.wm.encoder(goal_obs)

            actor_inp = tf.concat([start['feat'], goal_input], -1)

            start['action'] = tf.zeros_like(self.actor(actor_inp).mode())
            seq = {k: [v] for k, v in start.items()}

            for _ in range(self.horizon):
                actor_inp = tf.concat([seq['feat'][-1], goal_input], -1)
                action = self.actor(actor_inp).sample()
                state = self.wm.rssm.img_step(
                    {k: v[-1] for k, v in seq.items()}, action)
                feat = self.wm.rssm.get_feat(state)
                for key, value in {**state, 'action': action, 'feat': feat}.items():
                    seq[key].append(value)

            seq = {k: tf.stack(v, 0) for k, v in seq.items()}

            rewards = self.reward_fn(seq)

            returns = tf.reduce_sum(rewards, 0)

            return returns, seq

        if self.init_distribution is None:

            means, stds = self.get_distribution_from_obs(obs)
        else:

            means, stds = self.init_distribution

        opt_steps = 1 if self.evaluate_only else self.optimization_steps

        for i in range(opt_steps):

            if i == 0 and (self.dataset or self.sample_env_goals):

                if self.dataset:

                    random_batch = next(self.dataset)
                    random_batch = self.wm.preprocess(random_batch)

                    samples = tf.reshape(
                        random_batch[self.state_key], (-1,) + tuple(random_batch[self.state_key].shape[2:]))
                    if self.obs2goal is not None:
                        samples = self.obs2goal(samples)

                elif self.sample_env_goals:
                    num_cem_samples = int(
                        self.batch * self.env_goals_percentage)
                    num_env_samples = self.batch - num_cem_samples
                    cem_samples = tfd.MultivariateNormalDiag(
                        means, stds).sample(sample_shape=[num_cem_samples])
                    env_samples = self.sample_env_goals_fn(num_env_samples)
                    samples = tf.concat([cem_samples, env_samples], 0)

                means, vars = tf.nn.moments(samples, 0)

                samples = tfd.MultivariateNormalDiag(
                    means, stds).sample(sample_shape=[self.batch])

                samples = tf.clip_by_value(
                    samples, self.min_goal, self.max_goal)

            else:
                samples = tfd.MultivariateNormalDiag(
                    means, stds).sample(sample_shape=[self.batch])
                samples = tf.clip_by_value(
                    samples, self.min_goal, self.max_goal)

            if self.repeat_samples > 1:
                repeat_samples = tf.repeat(samples, self.repeat_samples, 0)
                repeat_fitness, seq = eval_fitness(repeat_samples)
                fitness = tf.reduce_mean(
                    tf.stack(tf.split(repeat_fitness, self.repeat_samples)), 0)
            else:
                fitness, seq = eval_fitness(samples)

            if self.planner == 'shooting_mppi':

                weights = tf.expand_dims(tf.nn.softmax(
                    self.mppi_gamma * fitness), axis=1)
                means = tf.reduce_sum(weights * samples, axis=0)
                stds = tf.sqrt(tf.reduce_sum(
                    weights * tf.square(samples - means), axis=0))

            elif self.planner == 'shooting_cem':

                elite_score, elite_inds = tf.nn.top_k(
                    fitness, elite_size, sorted=False)
                elite_samples = tf.gather(samples, elite_inds)

                means, vars = tf.nn.moments(elite_samples, 0)
                stds = tf.sqrt(vars + 1e-6)

        if self.planner == 'shooting_cem':
            self.vis_fn(elite_inds, elite_samples, seq, self.wm)
            self.elite_inds = elite_inds
            self.elite_samples = elite_samples
            self.final_seq = seq

        elif self.planner == 'shooting_mppi':

            self.elite_inds = None
            self.elite_samples = None
            self.final_seq = seq

        self.means = means
        self.stds = stds

        if self.evaluate_only:
            self.elite_samples = elite_samples
            self.elite_score = elite_score

        return self.sample_goal()

    def sample_goal(self, batch=1):

        if self.evaluate_only:

            weights = self.elite_score / self.elite_score.sum()
            idxs = tf.squeeze(tf.random.categorical(
                tf.math.log([weights]), batch), 0)
            samples = tf.gather(self.elite_samples, idxs)

        else:
            samples = tfd.MultivariateNormalDiag(
                self.means, self.stds).sample(sample_shape=[batch])

        return samples

    def create_init_distribution(self, init_candidates):
        """Create the starting distribution for seeding the planner.
        """
        def _create_init_distribution(init_candidates):
            means = tf.reduce_mean(init_candidates, 0)
            stds = tf.math.reduce_std(init_candidates, 0)

            if init_candidates.shape[0] == 1:
                stds = tf.ones(self.goal_dim, dtype=tf.float32) * \
                    self.std_scale
            return means, stds

        self.init_distribution = _create_init_distribution(init_candidates)

    def get_distribution_from_obs(self, obs):
        ob = tf.squeeze(obs[self.state_key])
        if self.gc_input == "state" or self.config.if_actor_gs:
            ob = self.obs2goal(ob)
        means = tf.cast(tf.identity(ob), tf.float32)
        assert np.prod(
            means.shape) == self.goal_dim, f"{np.prod(means.shape)}, {self.goal_dim}"
        stds = tf.ones(self.goal_dim, dtype=tf.float32) * self.std_scale
        init_distribution = tf.identity(means), tf.identity(stds)
        return init_distribution

    def get_init_distribution(self):
        if self.init_distribution is None:
            means = tf.zeros(self.goal_dim, dtype=tf.float32)
            stds = tf.ones(self.goal_dim, dtype=tf.float32) * self.std_scale
            self.init_distribution = tf.identity(means), tf.identity(stds)

        return self.init_distribution


class Cluster_goal_Planner:

    def __init__(self,
                 agnt,
                 config,
                 env,
                 obs2goal_fn=None,):

        self.agnt = agnt
        self.config = config
        self.wm = agnt.wm
        self.obs2goal_fn = obs2goal_fn
        self.cluster = agnt.wm.cluster
        self.dtype = agnt.wm.dtype
        if self.config.if_actor_gs:
            self.actor = agnt._task_behavior.actor_gs
        else:
            self.actor = agnt._task_behavior.actor

        if self.config.expl_behavior == 'Plan2Explore':
            self.reward_fn = agnt._expl_behavior.planner_intr_reward

        p_cfg = config.planner
        self.min_goal = np.array(p_cfg.goal_min, dtype=np.float32)
        self.max_goal = np.array(p_cfg.goal_max, dtype=np.float32)
        self.planner = p_cfg.planner_type
        self.horizon = p_cfg.horizon
        self.batch = p_cfg.batch
        self.cem_elite_ratio = p_cfg.cem_elite_ratio
        self.optimization_steps = p_cfg.optimization_steps
        self.std_scale = p_cfg.std_scale
        self.mppi_gamma = p_cfg.mppi_gamma
        self.evaluate_only = p_cfg.evaluate_only
        self.repeat_samples = p_cfg.repeat_samples
        self.env_goals_percentage = p_cfg.init_env_goal_percent

        self.goal_dim = np.prod(env.obs_space[config.goal_key].shape)
        self.act_space = env.act_space
        if isinstance(self.act_space, dict):
            self.act_space = self.act_space['action']

        self.gc_input = config.gc_input
        self.state_key = config.state_key

        state = None
        if state is None:
            self.initial_latent = self.wm.rssm.initial(1)
            self.initial_action = tf.zeros((1,) + self.act_space.shape)

        self.decoder = self.wm.heads['decoder']

        self.if_eval_fitness = False
        self.candidate_num = 1000
        self.sample_num = 100

    def search_goal(self, obs, state=None):

        if state is None:
            latent = self.wm.rssm.initial(1)
            action = tf.zeros((1, 1,) + self.act_space.shape)
            state = latent, action

        else:
            latent, action = state
            action = tf.expand_dims(action, 0)

        embed = self.wm.encoder(obs)

        post, prior = self.wm.rssm.observe(
            embed, action, obs['is_first'], latent)

        init_start = {k: v[:, -1] for k, v in post.items()}

        @tf.function
        def eval_fitness(goal):

            start = {k: v for k, v in init_start.items()}
            start['feat'] = self.wm.rssm.get_feat(start)

            start = tf.nest.map_structure(
                lambda x: tf.repeat(x, goal.shape[0], 0), start)

            actor_inp = tf.concat([start['feat'], goal], -1)

            start['action'] = tf.zeros_like(self.actor(actor_inp).mode())
            seq = {k: [v] for k, v in start.items()}

            for _ in range(self.horizon):
                actor_inp = tf.concat([seq['feat'][-1], goal], -1)

                action = self.actor(actor_inp).sample()
                state = self.wm.rssm.img_step(
                    {k: v[-1] for k, v in seq.items()}, action)
                feat = self.wm.rssm.get_feat(state)
                for key, value in {**state, 'action': action, 'feat': feat}.items():
                    seq[key].append(value)

            seq = {k: tf.stack(v, 0) for k, v in seq.items()}

            rewards = self.reward_fn(seq)

            returns = tf.reduce_sum(rewards, 0)

            return returns, seq

        samples = self.cluster.sample(self.candidate_num, self.sample_num)

        samples = tf.convert_to_tensor(samples.numpy(), dtype=self.dtype)

        if self.config.gc_input == "state" or self.config.if_actor_gs:

            initial_latent = tf.nest.map_structure(
                lambda x: tf.repeat(x, samples.shape[0], 0), self.initial_latent)
            initial_action = tf.nest.map_structure(
                lambda x: tf.repeat(x, samples.shape[0], 0), self.initial_action)

            latent, _ = self.wm.rssm.obs_step(
                initial_latent, initial_action, samples, True, True)

            feat = self.wm.rssm.get_feat(latent)

            samples_decoded_dist = self.decoder(feat)

            samples_decoded = samples_decoded_dist[self.wm.state_key].mean()

            samples_decoded = self.obs2goal_fn(samples_decoded)

            samples_decoded = tf.cast(samples_decoded, dtype=self.dtype)

            if self.config.expl_behavior == 'Plan2Explore' and self.if_eval_fitness:

                fitness, seq = eval_fitness(samples_decoded)
                weights = tf.expand_dims(tf.nn.softmax(
                    self.mppi_gamma * fitness), axis=1)

                max_indices = int(tf.argmax(weights).numpy())

                explore_goal_decoded = samples_decoded[max_indices]

            else:

                explore_goal_decoded = random.choice(samples_decoded)

        else:

            if self.config.expl_behavior == 'Plan2Explore' and self.if_eval_fitness:

                fitness, seq = eval_fitness(samples)
                weights = tf.expand_dims(tf.nn.softmax(
                    self.mppi_gamma * fitness), axis=1)

                max_indices = int(tf.argmax(weights).numpy())

                explore_goal = samples[max_indices]

            else:
                explore_goal = random.choice(samples)

            explore_goal = tf.convert_to_tensor(
                explore_goal.numpy(), dtype=self.wm.dtype)
            explore_goal = explore_goal[None]

            latent, _ = self.wm.rssm.obs_step(
                self.initial_latent, self.initial_action, explore_goal, True, True)

            feat = self.wm.rssm.get_feat(latent)

            explore_goal_decoded_dist = self.decoder(feat)

            explore_goal_decoded = explore_goal_decoded_dist[self.wm.state_key].mean(
            )

        return explore_goal_decoded


def softmax(X, theta=1.0, axis=None):
    """
        Compute the softmax of each element along an axis of X.

        Parameters
        ----------
        X: ND-Array. Probably should be floats.
        theta (optional): float parameter, used as a multiplier
                prior to exponentiation. Default = 1.0
        axis (optional): axis to compute values along. Default is the
                first non-singleton axis.

        Returns an array the same size as X. The result will sum to 1
        along the specified axis.
        """

    y = np.atleast_2d(X)

    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y * float(theta)

    y = y - np.max(y, axis=axis, keepdims=True)

    y = np.exp(y)

    ax_sum = np.sum(y, axis=axis, keepdims=True)

    p = y / ax_sum

    if len(X.shape) == 1:
        p = p.flatten()

    return p
