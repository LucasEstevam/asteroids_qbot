import sys
import os   
from collections import deque
sys.path.append('gen-py')
import tensorflow as tf
import numpy as np
import math
import random
import pickle
from botcoach import Coach

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

OBSERVATION_STEPS = 10000
MEMORY_SIZE = 60000
EXPLORE_STEPS = 20000

INITIAL_RANDOM_ACTION_PROB = 1
FINAL_RANDOM_ACTION_PROB = 0.05

MINI_BATCH_SIZE = 32
STATE_FRAMES = 1
LEARN_RATE = 1e-6
STORE_SCORES_LEN = 200
FUTURE_REWARD_DISCOUNT = 0.99

NUM_ACTIONS = 3*3*3*4

STATE_DIM = 340
HIDDEN_DIM = 192

CHECKPOINT_PATH = "./chk/"
OBSERVATIONS_STORE = "./obs.pkl"
SAVE_EVERY_X_STEPS = 10000

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def inferenceGraph():
    input_layer = tf.placeholder("float", [None, STATE_DIM])

    # fc1
    with tf.variable_scope('fc1') as scope:
        weights = _variable_with_weight_decay('weights', shape=[STATE_DIM, HIDDEN_DIM],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [HIDDEN_DIM], tf.constant_initializer(0.01))

        fc1 = tf.nn.relu(tf.matmul(input_layer, weights) + biases, name=scope.name)
        _activation_summary(fc1)

    # fc2
    with tf.variable_scope('fc2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[HIDDEN_DIM, NUM_ACTIONS],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [NUM_ACTIONS], tf.constant_initializer(0.01))

        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)
        _activation_summary(fc2)

    return(input_layer, fc2)



class CoachHandler:
    def __init__(self):
        self.session = tf.Session()
        self.observations = deque()

        if os.path.isfile(OBSERVATIONS_STORE):
            obs_load = open(OBSERVATIONS_STORE,'rb')
            self.observations = pickle.load(obs_load)
            obs_load.close()


        self.target = tf.placeholder("float", [None])
        self.action = tf.placeholder("float", [None, NUM_ACTIONS])

        self.input_layer, self.output_layer = inferenceGraph()
        
        readout_action = tf.reduce_sum(tf.mul(self.output_layer, self.action), reduction_indices=1)

        cost = tf.reduce_mean(tf.square(self.target - readout_action))
        self.train_operation = tf.train.AdamOptimizer(LEARN_RATE).minimize(cost)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        self.summary_op = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(CHECKPOINT_PATH,
                                        graph_def=self.session.graph.as_graph_def(add_shapes=True))

        self.session.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        if not os.path.exists(CHECKPOINT_PATH):
            os.mkdir(CHECKPOINT_PATH)
        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_PATH)

        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Loaded checkpoints %s" % checkpoint.model_checkpoint_path)


        self.random_prob = INITIAL_RANDOM_ACTION_PROB
        self.time = 0

    def botInit(self):
        print 'received bot init'

    def newObservation(self, lastState,lastAction, currentState, reward):               
        self.observations.append((lastState,lastAction,reward,currentState))
        if(len(self.observations) > MEMORY_SIZE):
            self.observations.popleft()

        if len(self.observations) > OBSERVATION_STEPS:
            self.train()
            self.time += 1

        self.last_state = currentState
        self.last_action = self.nextAction()

        if(len(self.observations) % 10000 == 0):
            print 'saving observations'
            obs_write = open(OBSERVATIONS_STORE,'wb')
            pickle.dump(self.observations,obs_write)
            obs_write.close()

        if(len(self.observations) % 1000 == 0):
            print 'observations: ' + str(len(self.observations))

        return(self.last_action)

    def nextAction(self):
        new_action = np.zeros([NUM_ACTIONS])

        if(random.random() <= self.random_prob):
            action_index = random.randrange(NUM_ACTIONS)
        else:
            output = self.session.run(self.output_layer, feed_dict={self.input_layer: [self.last_state]})[0]
            action_index = np.argmax(output)
        new_action[action_index] = 1
        return new_action


    def train(self):
        mini_batch = random.sample(self.observations, MINI_BATCH_SIZE)

        previous_states = [x[0] for x in mini_batch]
        actions = [x[1] for x in mini_batch]
        rewards = [x[2] for x in mini_batch]
        current_states = [x[3] for x in mini_batch]

        agents_expected_reward = []
        agents_reward_per_action, summary_str = self.session.run([self.output_layer, self.summary_op], feed_dict = {self.input_layer: current_states})
        for i in range(len(mini_batch)):
            agents_expected_reward.append(rewards[i] + FUTURE_REWARD_DISCOUNT*np.max(agents_reward_per_action[i]))

        self.session.run(self.train_operation, feed_dict={
            self.input_layer: previous_states,
            self.action: actions,
            self.target: agents_expected_reward
            })

        if self.time % 10 ==0:            
            self.summary_writer.add_summary(summary_str, self.time)

        if self.time % SAVE_EVERY_X_STEPS == 0:
            self.saver.save(self.session,CHECKPOINT_PATH + '/network', global_step=self.time)

if __name__ == '__main__':
    handler = CoachHandler()
    processor = Coach.Processor(handler)
    transport = TSocket.TServerSocket(port=9090)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    server.serve()