import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
import time

if "../" not in sys.path:
  sys.path.append("../")

from lib import plotting
from lib.deep_q_learner import *
from collections import deque, namedtuple

# Configuration
EVAL = True
env = gym.envs.make("Breakout-v0")
SAVE_DIR = "_dqn"
VALID_ACTIONS = [0, 1, 2, 3] # Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
PRINT_STEP = 100
REPLAY_MEMORY_SIZE = 200000
REPLAY_MEMORY_INIT_SIZE = 100


def main():
    tf.reset_default_graph()

    # Where we save our checkpoints and graphs
    experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id)+SAVE_DIR)

    # Create a glboal step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create estimators
    q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
    target_estimator = Estimator(scope="target_q")

    # State processor
    state_processor = StateProcessor()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # Create directories for checkpoints and summaries
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, "model")
        monitor_path = os.path.join(experiment_dir, "monitor")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(monitor_path):
            os.makedirs(monitor_path)

        saver = tf.train.Saver()
        # Load a previous checkpoint if we find one
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
        else:
            print("New model checkpoint")

        if EVAL:
            evaluating(sess, env, q_estimator=q_estimator, state_processor=state_processor, num_episodes=100)
        else:
            for t, stats in deep_q_learning(sess, saver,
                                            env,
                                            q_estimator=q_estimator,
                                            target_estimator=target_estimator,
                                            state_processor=state_processor,
                                            experiment_dir=experiment_dir,
                                            num_episodes=10000,
                                            replay_memory_size=REPLAY_MEMORY_SIZE,
                                            replay_memory_init_size=REPLAY_MEMORY_INIT_SIZE,
                                            update_target_estimator_every=10000,
                                            epsilon_start=1.0,
                                            epsilon_end=0.1,
                                            epsilon_decay_steps=500000,
                                            discount_factor=0.99,
                                            batch_size=32):

                print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))


class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calcualte the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])


    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
          action values.
        """
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss




if __name__ == "__main__":
    # execute only if run as a script
    main()
