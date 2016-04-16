from bot_interface import *
import os
import numpy as np
import random
import pickle
import signal
import sys
import math
sys.path.append('/home/lucas/Dev/asteroids/top-asteroids-challenge/Bots/python/gen-py')
from collections import deque
from botcoach import Coach

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

NUM_ACTIONS = 3*3*3*4
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
       
        self.last_scores = deque()

        self.last_action = np.zeros(NUM_ACTIONS)
        self.last_action[13] = 1

        self.last_state = None        
        self.previous_score = 0

        
    def process(self, gamestate):
        stateVector = []
        stateVector.append(gamestate.arenaRadius)
        stateVector.append(len(gamestate.ships))
        stateVector.append(len(gamestate.rocks))
        stateVector.append(len(gamestate.lasers))
        self.gamestate_cache = gamestate

        # my state
        stateVector.extend([self.ang, self.velang, self.charge,
                            self.posx, self.posy, self.velx, self.vely, self.radius])

        # my shots, up to 6
        i = 0
        for laser in gamestate.lasers.values():
            if(laser.owner == self.uid):                
                if(i < 6):
                    i = i + 1
                    stateVector.extend([laser.posx, laser.posy, laser.velx, laser.vely, laser.radius, laser.lifetime])

        while(i < 6):
            stateVector.extend([0, 0, 0, 0, 0, 0])            
            i = i + 1

        # up to 10 ships, skipping me
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

        # up to 21 lasers, skipping mine
        for i in xrange(0, 20):
            if(len(gamestate.lasers) > i):
                laser = gamestate.lasers.values()[i]
                if(laser.owner != self.uid):
                    stateVector.extend(
                        [laser.posx, laser.posy, laser.velx, laser.vely, laser.radius, laser.lifetime])
                else:
                    stateVector.extend([0, 0, 0, 0, 0, 0])    
            else:
                stateVector.extend([0, 0, 0, 0, 0, 0])

        # stateVector is ready for action

        if self.last_state is None:
            self.last_state = stateVector

        about_to_die = self.about_to_die(gamestate.rocks, gamestate.lasers, gamestate)

        if(about_to_die):
            reward = -1
            gamestate.log('dying!!!')
        else:
            reward = self.score - self.previous_score
        self.previous_score = self.score
        current_state = stateVector
      
        next_action = self.client.newObservation(self.last_state, self.last_action, current_state, reward)
        
        self.last_state = current_state
        self.last_action = next_action

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

        for rock in lasers.values():
            forecast_rock_pos = [rock.posx + rock.velx*forecast_interval, rock.posy + rock.vely*forecast_interval]
            forecast_dist = math.sqrt((forecast_rock_pos[0] - forecast_ship_pos[0])**2 + (forecast_rock_pos[1] - forecast_ship_pos[1])**2)
            if(forecast_dist < self.radius + rock.radius):
                return(True)

        return(False)
 

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
