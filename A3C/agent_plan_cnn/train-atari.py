#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train-atari.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import os
import sys
import re
import time
import random
import uuid
import argparse
import multiprocessing
import threading
import cv2
from collections import deque
import six
from six.moves import queue
import tensorflow as tf
import tensorflow.contrib.layers as layers

from tensorpack import *
from tensorpack.utils.concurrency import *
from tensorpack.utils.serialize import *
from tensorpack.utils.stats import *
from tensorpack.tfutils import symbolic_functions as symbf
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient

from tensorpack.RL import *
from simulator import *
import common
from common import (play_model, Evaluator, eval_model_multithread)


PLAN_ITER_NUM = 8

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
GAMMA = 0.99
CHANNEL = FRAME_HISTORY * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

LOCAL_TIME_MAX = 5
STEPS_PER_EPOCH = 6000
EVAL_EPISODE = 50
BATCH_SIZE = 128
SIMULATOR_PROC = 50
PREDICTOR_THREAD_PER_GPU = 2
PREDICTOR_THREAD = None
EVALUATE_PROC = min(multiprocessing.cpu_count() // 2, 20)

NUM_ACTIONS = None
ENV_NAME = None


def get_player(viz=False, train=False, dumpdir=None):
    pl = GymEnv(ENV_NAME, dumpdir=dumpdir)

    def func(img):
        return cv2.resize(img, IMAGE_SIZE[::-1])
    pl = MapPlayerState(pl, func)

    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()

    pl = HistoryFramePlayer(pl, FRAME_HISTORY)
    if not train:
        pl = PreventStuckPlayer(pl, 30, 1)
    pl = LimitLengthPlayer(pl, 40000)
    return pl


common.get_player = get_player


class MySimulatorWorker(SimulatorProcess):

    def _build_player(self):
        return get_player(train=True)


class Model(ModelDesc):
    def _get_inputs(self):
        assert NUM_ACTIONS is not None
        return [InputDesc(tf.float32, (None,) + IMAGE_SHAPE3, 'state'),
                InputDesc(tf.int64, (None,), 'action'),
                InputDesc(tf.float32, (None,), 'futurereward')]

    def _get_NN_prediction(self, image):
        image = image / 255.0
        # feature extraction
        features = layers.conv2d(image, num_outputs=16, kernel_size=[8,8], 
            stride=[4,4], padding="VALID",activation_fn=tf.nn.relu)
        features = layers.conv2d(features, num_outputs=32, kernel_size=[4,4], 
            stride=[2,2], padding="VALID",activation_fn=tf.nn.relu)
        features_flat = layers.flatten(features) 

        # plan module cnn
        #   weights init
        ch_h = 150
        bias = tf.Variable(np.random.randn(1,1,1,ch_h)*0.01, name='cnn_h_bias', dtype=tf.float32)
        cnn_h = tf.Variable(np.random.randn(1,1,32,ch_h)*0.01, name='cnn_h', dtype=tf.float32)
        cnn_1 = tf.Variable(np.random.randn(1,1,ch_h,64)*0.01, name='cnn_1', dtype=tf.float32)
        cnn_2 = tf.Variable(np.random.randn(1,1,64,64)*0.01, name='cnn_2', dtype=tf.float32)
        cnn_3 = tf.Variable(np.random.randn(1,1,64,64)*0.01, name='cnn_3', dtype=tf.float32)
        cnn_fb = tf.Variable(np.random.randn(1,1,64,64)*0.01, name='cnn_fb', dtype=tf.float32)

        #   net
        plan_h = tf.nn.relu(tf.nn.conv2d(features, cnn_h, [1,1,1,1], padding='SAME') + bias)
        plan_1 = tf.nn.relu(tf.nn.conv2d(plan_h, cnn_1, [1,1,1,1], padding='SAME'))
        plan_2 = tf.nn.relu(tf.nn.conv2d(plan_1, cnn_2, [1,1,1,1], padding='SAME'))
        plan_3 = tf.nn.relu(tf.nn.conv2d(plan_2, cnn_3, [1,1,1,1], padding='SAME'))

        for i in range(PLAN_ITER_NUM - 1):
            plan_13 = tf.concat([plan_1,plan_3],3)
            cnn_2_fb = tf.concat([cnn_2, cnn_fb], 2)
            plan2 = tf.nn.relu(tf.nn.conv2d(plan_13, cnn_2_fb, [1,1,1,1], padding='SAME'))
            plan_3 = tf.nn.relu(tf.nn.conv2d(plan_2, cnn_3, [1,1,1,1], padding='SAME'))

        plan_net = layers.conv2d(plan_3, num_outputs=16, kernel_size=[1,1], stride=[1,1], padding='SAME', activation_fn=tf.nn.relu)
        plan_net = layers.flatten(plan_net)

        # combine
        hidden_plan = layers.fully_connected(plan_net, 256, activation_fn=tf.nn.relu)
        hidden_reac = layers.fully_connected(features_flat, 256, activation_fn=tf.nn.relu)
        alpha  = tf.Variable(0.5, name = 'alpha', dtype=tf.float32)
        hidden = tf.add(tf.multiply(hidden_plan, alpha), tf.multiply(hidden_reac, (1-alpha)))

        logits = layers.fully_connected(hidden, NUM_ACTIONS)
        value = layers.fully_connected(hidden, 1)
        return logits, value


    def _build_graph(self, inputs):
        state, action, futurereward = inputs
        logits, self.value = self._get_NN_prediction(state)
        self.value = tf.squeeze(self.value, [1], name='pred_value')  # (B,)
        self.policy = tf.nn.softmax(logits, name='policy')

        expf = tf.get_variable('explore_factor', shape=[],
                               initializer=tf.constant_initializer(1), trainable=False)
        policy_explore = tf.nn.softmax(logits * expf, name='policy_explore')
        is_training = get_current_tower_context().is_training
        if not is_training:
            return
        log_probs = tf.log(self.policy + 1e-6)

        log_pi_a_given_s = tf.reduce_sum(
            log_probs * tf.one_hot(action, NUM_ACTIONS), 1)
        advantage = tf.subtract(tf.stop_gradient(self.value), futurereward, name='advantage')
        policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage, name='policy_loss')
        xentropy_loss = tf.reduce_sum(
            self.policy * log_probs, name='xentropy_loss')
        value_loss = tf.nn.l2_loss(self.value - futurereward, name='value_loss')

        pred_reward = tf.reduce_mean(self.value, name='predict_reward')
        advantage = symbf.rms(advantage, name='rms_advantage')
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        self.cost = tf.add_n([policy_loss, xentropy_loss * entropy_beta, value_loss])
        self.cost = tf.truediv(self.cost,
                               tf.cast(tf.shape(futurereward)[0], tf.float32),
                               name='cost')
        summary.add_moving_summary(policy_loss, xentropy_loss,
                                   value_loss, pred_reward, advantage, self.cost)

    def _get_optimizer(self):
        lr = symbf.get_scalar_var('learning_rate', 0.001, summary=True)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)

        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1)),
                     SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


class MySimulatorMaster(SimulatorMaster, Callback):
    def __init__(self, pipe_c2s, pipe_s2c, model):
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
        self.M = model
        self.queue = queue.Queue(maxsize=BATCH_SIZE * 8 * 2)

    def _setup_graph(self):
        self.async_predictor = MultiThreadAsyncPredictor(
            self.trainer.get_predictors(['state'], ['policy_explore', 'pred_value'],
                                        PREDICTOR_THREAD), batch_size=15)

    def _before_train(self):
        self.async_predictor.start()

    def _on_state(self, state, ident):
        def cb(outputs):
            distrib, value = outputs.result()
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib)
            client = self.clients[ident]
            client.memory.append(TransitionExperience(state, action, None, value=value))
            self.send_queue.put([ident, dumps(action)])
        self.async_predictor.put_task([state], cb)

    def _on_episode_over(self, ident):
        self._parse_memory(0, ident, True)

    def _on_datapoint(self, ident):
        client = self.clients[ident]
        if len(client.memory) == LOCAL_TIME_MAX + 1:
            R = client.memory[-1].value
            self._parse_memory(R, ident, False)

    def _parse_memory(self, init_r, ident, isOver):
        client = self.clients[ident]
        mem = client.memory
        if not isOver:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R = float(init_r)
        for idx, k in enumerate(mem):
            R = np.clip(k.reward, -1, 1) + GAMMA * R
            self.queue.put([k.state, k.action, R])

        if not isOver:
            client.memory = [last]
        else:
            client.memory = []


def get_config():
    dirname = os.path.join('train_log', 'train-atari-{}'.format(ENV_NAME))
    logger.set_logger_dir(dirname)
    M = Model()

    name_base = str(uuid.uuid1())[:6]
    PIPE_DIR = os.environ.get('TENSORPACK_PIPEDIR', '.').rstrip('/')
    namec2s = 'ipc://{}/sim-c2s-{}'.format(PIPE_DIR, name_base)
    names2c = 'ipc://{}/sim-s2c-{}'.format(PIPE_DIR, name_base)
    procs = [MySimulatorWorker(k, namec2s, names2c) for k in range(SIMULATOR_PROC)]
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    master = MySimulatorMaster(namec2s, names2c, M)
    dataflow = BatchData(DataFromQueue(master.queue), BATCH_SIZE)
    return TrainConfig(
        dataflow=dataflow,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(80, 0.0003), (120, 0.0001)]),
            ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            ScheduledHyperParamSetter('explore_factor',
                                      [(80, 2), (100, 3), (120, 4), (140, 5)]),
            master,
            StartProcOrThread(master),
            PeriodicTrigger(Evaluator(EVAL_EPISODE, ['state'], ['policy']), every_k_epochs=2),
        ],
        session_creator=sesscreate.NewSessionCreator(
            config=get_default_sess_config(0.5)),
        model=M,
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--env', help='env', required=True)
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train'], default='train')
    args = parser.parse_args()

    ENV_NAME = args.env
    assert ENV_NAME
    p = get_player()
    del p    # set NUM_ACTIONS

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.task != 'train':
        assert args.load is not None

    if args.task != 'train':
        cfg = PredictConfig(
            model=Model(),
            session_init=SaverRestore(args.load),
            input_names=['state'],
            output_names=['policy'])
        if args.task == 'play':
            play_model(cfg)
        elif args.task == 'eval':
            eval_model_multithread(cfg, EVAL_EPISODE)
    else:
        nr_gpu = get_nr_gpu()
        if nr_gpu > 0:
            if nr_gpu > 1:
                predict_tower = list(range(nr_gpu))[-nr_gpu // 2:]
            else:
                predict_tower = [0]
            PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
            train_tower = list(range(nr_gpu))[:-nr_gpu // 2] or [0]
            logger.info("[BA3C] Train on gpu {} and infer on gpu {}".format(
                ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))
            trainer = AsyncMultiGPUTrainer
        else:
            logger.warn("Without GPU this model will never learn! CPU is only useful for debug.")
            nr_gpu = 0
            PREDICTOR_THREAD = 1
            predict_tower, train_tower = [0], [0]
            trainer = QueueInputTrainer
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        config.tower = train_tower
        config.predict_tower = predict_tower
        trainer(config).train()