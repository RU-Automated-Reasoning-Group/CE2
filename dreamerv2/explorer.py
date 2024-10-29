

from copy import deepcopy
import tensorflow as tf
from tensorflow_probability import distributions as tfd

import nor_agent
import common
import numpy as np


class Random(common.Module):

    def __init__(self, config, act_space, wm, tfstep, reward):
        self.config = config
        self.act_space = act_space

    def actor(self, feat):
        shape = feat.shape[:-1] + self.act_space.shape
        if self.config.actor.dist == 'onehot':
            return common.OneHotDist(tf.zeros(shape))
        else:
            dist = tfd.Uniform(-tf.ones(shape), tf.ones(shape))
            return tfd.Independent(dist, 1)

    def train(self, start, context, data):
        return None, {}


class Plan2Explore(common.Module):

    def __init__(self, config, act_space, wm, tfstep, reward):
        self.config = config
        self.reward = reward
        self.wm = wm

        p2e_config = deepcopy(config)
        overrides = {
            "discount": config.p2e_discount
        }
        p2e_config = p2e_config.update(overrides)
        self.ac = nor_agent.ActorCritic(p2e_config, act_space, tfstep)
        self.actor = self.ac.actor
        stoch_size = config.rssm.stoch
        if config.rssm.discrete:
            stoch_size *= config.rssm.discrete
        size = {

            'embed': wm.encoder.embed_size,
            'stoch': stoch_size,
            'deter': config.rssm.deter,
            'feat': config.rssm.stoch + config.rssm.deter,
            'obs_decoded_gs': int(np.prod(wm.shapes[wm.goal_key])),
        }

        size = size[self.config.disag_target]

        print("explore ensembel ouput size: ", size)

        self._networks = [
            common.MLP(size, **config.expl_head)
            for _ in range(config.disag_models)]
        self.opt = common.Optimizer('expl', **config.expl_opt)
        self.extr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)
        self.intr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)

    def train(self, start, context, data):
        metrics = {}
        stoch = start['stoch']
        if self.config.rssm.discrete:
            stoch = tf.reshape(
                stoch, stoch.shape[:-2] + (stoch.shape[-2] * stoch.shape[-1]))
        target = {
            'embed': context['embed'],
            'stoch': stoch,
            'deter': start['deter'],
            'feat': context['feat'],
            'obs_decoded_gs': context.get('obs_decoded_gs', None),
        }[self.config.disag_target]

        inputs = context['feat']
        if self.config.disag_action_cond:
            action = tf.cast(data['action'], inputs.dtype)
            inputs = tf.concat([inputs, action], -1)
        metrics.update(self._train_ensemble(inputs, target))
        metrics.update(self.ac.train(
            self.wm, start, data['is_terminal'], self._intr_reward))

        return None, metrics

    def _intr_reward(self, seq):
        inputs = seq['feat']
        if self.config.disag_action_cond:
            action = tf.cast(seq['action'], inputs.dtype)
            inputs = tf.concat([inputs, action], -1)
        preds = [head(inputs).mode() for head in self._networks]
        disag = tf.tensor(preds).std(0).mean(-1)
        if self.config.disag_log:
            disag = tf.math.log(disag)
        reward = self.config.expl_intr_scale * self.intr_rewnorm(disag)[0]

        if self.config.expl_extr_scale:
            reward += self.config.expl_extr_scale * \
                self.extr_rewnorm(self.reward(seq))[0]

        return reward

    def planner_intr_reward(self, seq):

        inputs = feat = seq['feat']

        if self.config.disag_action_cond:
            action = tf.cast(seq['action'], inputs.dtype)
            inputs = tf.concat([inputs, action], -1)

        preds = [head(inputs).mode() for head in self._networks]
        disag = tf.tensor(preds).std(0).mean(-1)

        if self.config.disag_log:
            disag = tf.math.log(disag)

        reward = self.config.expl_intr_scale * self.intr_rewnorm(disag)[0]

        if self.config.expl_extr_scale:
            reward += self.config.expl_extr_scale * \
                self.extr_rewnorm(self.reward(seq))[0]

        if self.config.planner.cost_use_p2e_value:

            value = self.ac._target_critic(seq['feat']).mode()

            disc = self.config.p2e_discount * tf.ones_like(reward)

            returns = common.lambda_return(
                reward[:-1], value[:-1], disc[:-1],
                bootstrap=value[-1],
                lambda_=self.config.discount_lambda,
                axis=0)

            if self.config.planner.final_step_cost:
                returns = returns[-10:]

            return returns

        else:
            return reward

    def _train_ensemble(self, inputs, targets):

        if self.config.disag_offset:
            targets = targets[:, self.config.disag_offset:]
            inputs = inputs[:, :-self.config.disag_offset]
        targets = tf.stop_gradient(targets)
        inputs = tf.stop_gradient(inputs)
        with tf.GradientTape() as tape:
            preds = [head(inputs) for head in self._networks]
            preds_mode = [pred.mode() for pred in preds]
            loss = -sum([pred.log_prob(targets).mean() for pred in preds])
        metrics = self.opt(tape, loss, self._networks)
        return metrics


class ModelLoss(common.Module):

    def __init__(self, config, act_space, wm, tfstep, reward):
        self.config = config
        self.reward = reward
        self.wm = wm
        self.ac = nor_agent.ActorCritic(config, act_space, tfstep)
        self.actor = self.ac.actor
        self.head = common.MLP([], **self.config.expl_head)
        self.opt = common.Optimizer('expl', **self.config.expl_opt)

    def train(self, start, context, data):
        metrics = {}
        target = tf.cast(context[self.config.expl_model_loss], tf.float32)

        with tf.GradientTape() as tape:
            loss = -self.head(context['feat']).log_prob(target).mean()

        metrics.update(self.opt(tape, loss, self.head))

        metrics.update(self.ac.train(
            self.wm, start, data['is_terminal'], self._intr_reward))

        return None, metrics

    def _intr_reward(self, seq):
        reward = self.config.expl_intr_scale * self.head(seq['feat']).mode()

        if self.config.expl_extr_scale:
            reward += self.config.expl_extr_scale * self.reward(seq)

        return reward


class RND(common.Module):

    def __init__(self, config, act_space, wm, tfstep, reward):
        self.config = config
        self.reward = reward
        self.wm = wm

        p2e_config = deepcopy(config)
        overrides = {
            "discount": config.p2e_discount
        }
        p2e_config = p2e_config.update(overrides)
        self.ac = nor_agent.ActorCritic(p2e_config, act_space, tfstep)
        self.actor = self.ac.actor
        stoch_size = config.rssm.stoch
        if config.rssm.discrete:
            stoch_size *= config.rssm.discrete
        size = {

            'embed': wm.encoder.embed_size,
            'stoch': stoch_size,
            'deter': config.rssm.deter,
            'feat': config.rssm.stoch + config.rssm.deter,
            'obs_decoded_gs': int(np.prod(wm.shapes[wm.goal_key])),
        }

        size = size[self.config.disag_target]

        self._target_network = common.MLP(size, **config.expl_head)
        self._predictor_network = common.MLP(size, **config.expl_head)

        self.opt = common.Optimizer('expl', **config.expl_opt)
        self.extr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)

        self.intr_rewnorm = RunningMeanStd()

    def train(self, start, context, data):
        metrics = {}
        inputs = context['feat']
        _metrics = self._train_predictor(inputs)
        metrics.update(_metrics)

        metrics.update(self.ac.train(
            self.wm, start, data['is_terminal'], self._intr_reward_rnd))
        return None, metrics

    def _intr_reward_rnd(self, seq):
        inputs = seq['feat']

        f = self._target_network(inputs).mean()
        f_hat = self._predictor_network(inputs).mean()
        reward = self.config.expl_intr_scale * \
            tf.norm(f - f_hat, ord='euclidean', axis=-1)**2
        reward = self.intr_rewnorm.transform(reward)

        if self.config.expl_extr_scale:
            reward += self.config.expl_extr_scale * self.extr_rewnorm(
                self.reward(seq))[0]
        return reward

    def planner_intr_reward(self, seq):

        return self._intr_reward_rnd(seq)

    def _train_predictor(self, inputs):
        inputs = tf.stop_gradient(inputs)
        with tf.GradientTape() as tape:
            f = self._target_network(inputs)
            f_hat = self._predictor_network(inputs)
            loss = -f.log_prob(f_hat.mean()).mean()
            reward = self.config.expl_intr_scale * \
                tf.norm(f.mean() - f_hat.mean(), ord='euclidean', axis=-1)**2
            self.intr_rewnorm.update(tf.reshape(reward, [-1]))

        metrics = self.opt(tape, loss, self._predictor_network)
        return metrics


class RunningMeanStd(object):

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = tf.Variable(tf.zeros(shape, tf.float32), False)
        self.var = tf.Variable(tf.ones(shape, tf.float32), False)
        self.count = tf.Variable(epsilon, False, dtype=tf.float32)

    def update(self, x):
        batch_mean, batch_var = tf.nn.moments(x, 0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + (delta**2) * self.count * batch_count / (tot_count)
        new_var = M2 / (tot_count)

        self.mean.assign(new_mean)
        self.var.assign(new_var)
        self.count.assign(tot_count)

    def transform(self, inputs):
        return inputs / tf.math.sqrt(self.var)
