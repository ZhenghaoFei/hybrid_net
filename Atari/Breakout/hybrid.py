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
EVAL = False
env = gym.envs.make("Breakout-v0")
SAVE_DIR = "_hybrid"
VALID_ACTIONS = [0, 1, 2, 3] # Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
PRINT_STEP = 100
REPLAY_MEMORY_SIZE = 200000
REPLAY_MEMORY_INIT_SIZE = 100
PLAN_LAYERS = 7

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


class StateProcessor():
    """
    Processes a raw Atari iamges. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })

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

        # Networks >>>>>>>>>>
        # Three convolutional layers
        features = tflearn.conv_2d(X, 32, 8, strides=4, activation='relu', name='conv1')
        features = tflearn.conv_2d(features, 64, 4, strides=2, activation='relu', name='conv2')
        features = tflearn.conv_2d(features, 64, 3, strides=1, activation='relu', name='conv3')

        # rnn
        features_rnn = tflearn.layers.core.flatten(features)
        fc1 = tflearn.fully_connected(features_rnn, 64)
        fc2 = tflearn.fully_connected(fc1, 32)
        fc_fb = tflearn.fully_connected(fc2, 64)

        net_rnn = tflearn.activation(tf.matmul(features_rnn,fc1.W) + fc1.b, activation='relu')
        for i in range(PLAN_LAYERS - 1):
            net_rnn = tflearn.activation(tf.matmul(net_rnn,fc2.W) + fc2.b, activation='relu')
            net_rnn = tflearn.activation(tf.matmul(net_rnn,fc_fb.W) + tf.matmul(features_rnn, fc1.W) + fc_fb.b + fc1.b, activation='relu')
        net_rnn = tflearn.activation(tf.matmul(net_rnn,fc2.W) + fc2.b, activation='relu')
        rnn_out = tflearn.fully_connected(net_rnn, 512)

        # Fully connected layers
        fc_out = tflearn.fully_connected(features, 512)

        #  Fusion
        alpha  = tflearn.fully_connected(features, 16)
        self.alpha  = tflearn.fully_connected(alpha, 1, activation='sigmoid')
        net = tf.add(tf.multiply(fc_out, self.alpha), tf.multiply(rnn_out, (1-self.alpha)))

        # Output Layer
        self.predictions = tflearn.fully_connected(net, len(VALID_ACTIONS))
        # <<<<<<<<<<<<<<<<<


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
            # tf.summary.histogram("loss_hist", self.losses),
            tf.summary.scalar("alpha", tf.reduce_mean(self.alpha)),
            # tf.summary.histogram("alpha_hist", self.alpha),
            # tf.summary.histogram("q_values_hist", self.predictions),
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
