import tensorflow as tf
from bot_interface import *
from collections import deque
import os
import numpy as np
import random
import pickle
import signal
import sys
import math
sys.path.append('/home/lucas/Dev/asteroids/top-asteroids-challenge/Bots/python/gen-py')

from botcoach import Coach

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol


STATE_DIM = 313

HIDDEN_DIM = 192

NUM_ACTIONS = 3*3*3*4

PLAYBACK_MODE = False
FUTURE_REWARD_DISCOUNT = 0.99
OBSERVATION_STEPS = 600000
EXPLORE_STEPS = 2000000
INITIAL_RANDOM_ACTION_PROB = 1
FINAL_RANDOM_ACTION_PROB = 0.05
MEMORY_SIZE = 600000
MINI_BATCH_SIZE = 32
STATE_FRAMES = 1
LEARN_RATE = 1e-6
STORE_SCORES_LEN = 200

CHECKPOINT_PATH = "./chk/"

OBSERVATIONS_STORE = "/mnt/ramdisk/"





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
    input_layer = tf.placeholder("float", [1, STATE_DIM])

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


class Qbot(BotBase):

    
    def __init__(self):
        transport = TSocket.TSocket('localhost', 9090)

        # Buffering is critical. Raw sockets are very slow
        transport = TTransport.TBufferedTransport(transport)

        # Wrap in a protocol
        protocol = TBinaryProtocol.TBinaryProtocol(transport)

        # Create a client to use the protocol encoder
        self.client = Coach.Client(protocol)
        
        transport.open()
        self.client.botInit()
        self.corruptions = 0
        self.session = tf.Session()
        self.input_layer, self.output_layer = inferenceGraph()

        self.target = tf.placeholder("float", [None])
        self.action = tf.placeholder("float", [None, NUM_ACTIONS])

        readout_action = tf.reduce_sum(tf.mul(self.output_layer, self.action), reduction_indices=1)

        cost = tf.reduce_mean(tf.square(self.target - readout_action))
        self.train_operation = tf.train.AdamOptimizer(LEARN_RATE).minimize(cost)
        
        self.observations = deque()
        # filecount = 0
        # # load pickled observations, if any
        # for item in os.listdir(OBSERVATIONS_STORE):
        #     filecount = filecount + 1
        #     if(os.stat(OBSERVATIONS_STORE + item).st_size > 0):
        #         try:
        #             with open(OBSERVATIONS_STORE + item,'rb') as f:
        #                 while True:
        #                     try:
        #                         self.observations.extend(pickle.load(f))
        #                     except ValueError:
        #                         self.corruptions = self.corruptions + 1
        #                         # pickle is corrupted, change previous state
        #         except EOFError:
        #             pass

       
        self.current_observations = deque()
        # self.obs_file = open(OBSERVATIONS_STORE + str(filecount), 'wb')

        self.initial_obs_size = len(self.observations)
        self.last_scores = deque()

        self.last_action = np.zeros(NUM_ACTIONS)
        self.last_action[1] = 1

        self.last_state = None
        self.random_prob = INITIAL_RANDOM_ACTION_PROB
        self.time = 0

        self.previous_score = 0
        self.saved = False

        self.session.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        if not os.path.exists(CHECKPOINT_PATH):
            os.mkdir(CHECKPOINT_PATH)
        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_PATH)

        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Loaded checkpoints %s" % checkpoint.model_checkpoint_path)


    def process(self, gamestate):
        gamestate.log('corruptions' + str(self.corruptions))
        stateVector = []
        stateVector.append(gamestate.arenaRadius)
        stateVector.append(len(gamestate.ships))
        stateVector.append(len(gamestate.rocks))
        stateVector.append(len(gamestate.lasers))
        self.gamestate_cache = gamestate
        # my state
        stateVector.extend([self.ang, self.velang, self.charge,
                            self.posx, self.posy, self.velx, self.vely, self.radius])

        # up to 10 ships
        for i in xrange(0, 9):
            if(len(gamestate.ships) > i):
                ship = gamestate.ships.values()[i]
                if(ship.uid != self.uid):
                    stateVector.extend([ship.ang, ship.velang, ship.charge, ship.score,
                                        ship.posx, ship.posy, ship.velx, ship.vely, ship.radius])
            else:
                stateVector.extend([0, 0, 0, 0, 0, 0, 0, 0, 0])

        # up to 21 asteroids
        for i in xrange(0, 20):
            if(len(gamestate.rocks) > i):
                rock = gamestate.rocks.values()[i]
                stateVector.extend(
                    [rock.posx, rock.posy, rock.velx, rock.vely, rock.radius])
            else:
                stateVector.extend([0, 0, 0, 0, 0])

        # up to 21 lasers
        for i in xrange(0, 20):
            if(len(gamestate.lasers) > i):
                laser = gamestate.lasers.values()[i]
                stateVector.extend(
                    [laser.posx, laser.posy, laser.velx, laser.vely, laser.radius, laser.lifetime])
            else:
                stateVector.extend([0, 0, 0, 0, 0, 0])

        # stateVector is ready for action

        if self.last_state is None:
            self.last_state = stateVector

        about_to_die = self.about_to_die(gamestate.rocks, gamestate.lasers, gamestate)

        if(about_to_die):
            reward = self.score - self.previous_score
        else:
            reward = -1
        self.previous_score = self.score
        current_state = stateVector


        # self.observations.append((self.last_state, self.last_action, reward, current_state))
        self.client.newObservation(self.last_state, self.last_action, current_state, reward)
        

        if len(self.observations) > OBSERVATION_STEPS:
            self.train()
            self.time += 1

        self.last_state = current_state
        self.last_action = self.choose_next_action()

        # if(about_to_die):
        #     if(not self.saved):
        #         pickle.dump((self.last_state, self.last_action, reward, current_state), self.obs_file)
        #         self.obs_file.close()
        #         self.saved = True
        #     gamestate.log('panic!!!!!!!!!!!!!!!!!')
        # else:
        #     if(not self.saved):
        #         pickle.dump((self.last_state, self.last_action, reward, current_state), self.obs_file)
        gamestate.log(len(self.observations))

        return self.action_to_Action(self.last_action)

    def dot_prod(self, x1, x2, y1, y2):
        return x1*y1 + x2*y2

    def about_to_die(self, rocks, lasers, gamestate):
        threshold = 0.07 # seconds till collision
        forecast_interval = 0.1
        forecast_ship_pos = [self.posx + self.velx*forecast_interval, self.posy + self.vely*forecast_interval]
        for rock in rocks.values():
            forecast_rock_pos = [rock.posx + rock.velx*forecast_interval, rock.posy + rock.vely*forecast_interval]
            forecast_dist = math.sqrt((forecast_rock_pos[0] - forecast_ship_pos[0])**2 + (forecast_rock_pos[1] - forecast_ship_pos[1])**2)
            if(forecast_dist < self.radius + rock.radius):
                return(True)
            # dist = math.sqrt((rock.posx - self.posx)**2 + (rock.posy - self.posy)**2)

            # if(dist < 8):
            #     dist = dist - self.radius - rock.radius
            #     # decompose velocity vector to line between ship and rock
            #     collision_vec = [self.posx - rock.posx, self.posy-rock.posy]
            #     collision_vel = self.dot_prod(collision_vec[0], collision_vec[1], rock.velx, rock.vely)/dist
            #     collision_vel = collision_vel - self.dot_prod(collision_vec[0], collision_vec[1], self.velx, self.vely)/dist
            #     time_to_colllision = dist/collision_vel
            #     if(time_to_colllision < threshold and time_to_colllision > 0):
            #         return(True)

        for rock in lasers.values():
            forecast_rock_pos = [rock.posx + rock.velx*forecast_interval, rock.posy + rock.vely*forecast_interval]
            forecast_dist = math.sqrt((forecast_rock_pos[0] - forecast_ship_pos[0])**2 + (forecast_rock_pos[1] - forecast_ship_pos[1])**2)
            if(forecast_dist < self.radius + rock.radius):
                return(True)
            # if(rock.owner != self.uid):
            #     dist = math.sqrt((rock.posx - self.posx)**2 + (rock.posy - self.posy)**2)
            #     if(dist < 10):
            #         dist = dist - self.radius - rock.radius
            #         # decompose velocity vector to line between ship and rock
            #         collision_vec = [self.posx - rock.posx, self.posy-rock.posy]
            #         collision_vel = self.dot_prod(collision_vec[0], collision_vec[1], rock.velx, rock.vely)/dist
            #         collision_vel = collision_vel - self.dot_prod(collision_vec[0], collision_vec[1], self.velx, self.vely)/dist
            #         time_to_colllision = dist/collision_vel                    
            #         if((time_to_colllision < threshold) and (time_to_colllision > 0)):
            #             return(True)

        return(False)


    def choose_next_action(self):
        new_action = np.zeros([NUM_ACTIONS])

        if(random.random() <= self.random_prob):
            action_index = random.randrange(NUM_ACTIONS)
        else:
            output = self.session.run(self.output_layer, feed_dict={self.input_layer: [self.last_state]})[0]
            action_index = np.argmax(output)
        new_action[action_index] = 1
        return new_action

    def action_to_Action(self, action):
        action_index = np.argmax(action)
        thrust_value = action_index % 3 - 1

        x = action_index % 9
        if(x < 3):
            thrust_front_value = -1
        elif (x < 6):
            thrust_front_value = 0
        else:
            thrust_front_value = 1

        x = action_index % 27
        if(x < 9):
            thrust_back_value = -1
        elif (x < 18):
            thrust_back_value = 0
        else:
            thrust_back_value = 1

        if(action_index < 27):
            shoot = 0
        elif(action_index < 54):
            shoot = 1
        elif(action_index < 81):
            shoot = 2
        else:
            shoot = 3

        return(Action(thrust_value, thrust_front_value, thrust_back_value, shoot))



GameState(Qbot()).connect()
