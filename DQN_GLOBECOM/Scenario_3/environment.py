# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 18:33:54 2019

@author: Aidin
"""

import numpy as np

from tf_agents.environments import time_step as ts
from tf_agents.environments import py_environment

from tf_agents.specs import array_spec


FIRST = ts.StepType.FIRST
MID = ts.StepType.MID
LAST = ts.StepType.LAST

class UAVGridEnvTime(py_environment.PyEnvironment):

  def __init__(self, num_IoTD, w, IoTD_loc, AoI_max, E_max, boundary , tau, start_loc, end_loc, sigma, C , beta, endtime_penalty,height,wh_cost, dist_conv,energy_conv):
      self.num_IoTD = num_IoTD # number of IoT devices
      self.w = w# the importance level of IoT devices
      self.AoI_max = AoI_max # the vector with components equal to the maximum allowed AoI of each IoTD
      self.E_max = E_max # the vector with components equal to the maximum available energy of each IoTD
      self.tau = tau # the time limit
      self.t = tau # counter 
      self.start_loc = start_loc # start location [x,y]
      self.end_loc = end_loc # end location [x,y]
      self.height = height # height at which the UAV is flying (nodes are at 0)
      self.IoTD_loc = IoTD_loc # the vector which indicates the location of each IoTD [[x1,y1], ...]
      self.C = C # required rate capacity
      self.sigma = sigma # noise level
      self.boundary = boundary # environment boundary
      self.beta = beta # weight for differentiating between the AoI and the distance from the destination
      self.endtime_penalty = endtime_penalty # the penalty when the UAV does not reach dest.
      self.wh_cost = wh_cost # cost of hitting the wall (negative)
      self.dist_conv = dist_conv #distance conversion
      self.energy_conv = energy_conv # used for quantizing the energy
    
      '''
      Action space consists of the following:
          - Directions: left:0 ,right: 1, up: 2, down: 3, stay steady: 4 
          - Scheduling: 0, 1, ..., num_IoTD
      '''
      self._action_spec = array_spec.BoundedArraySpec(
              shape=(1,), dtype=np.int32, minimum=0, maximum=4 + 5 * self.num_IoTD, name='action')
      
      '''
      State space consists of the following:
          - AoI for every IoTD: 0 - AoI_max[i] -> size: num_IoTD
          - Energy level for every IoTD: 0 - E_max[i] -> size: num_IoTD
          - UAV Location: x,y: 0,..., boundary[0] & 0,...,boundary[1] -> size: 2
          - Time left: Time constraint - Required Time: 0, tau #Note that at every 
          time step we have to reduce the time constrant -> size: 1
      '''
      self._observation_spec = array_spec.BoundedArraySpec(
              shape=(1,self.num_IoTD + self.num_IoTD + 2 + 1), dtype=np.int32, 
              minimum = [1] * self.num_IoTD + [0] * (self.num_IoTD + 2 + 1), maximum = AoI_max + self.E_max + self.boundary + [tau ],
              name='observation')
      self._state = [1] * self.num_IoTD + self.E_max + self.start_loc + [self.t - self.req_time(self.start_loc)] # state (AoI,E,Loc,t)
      self._episode_ended = False
#      self._current_time_step = ts.transition(
#              np.array([self._state], dtyspe=np.int32), reward = 0, discount=1.0)

  def action_spec(self):
      return self._action_spec

  def observation_spec(self):
      return self._observation_spec
  
#  def time_step_spec(self):
#      return ts.time_step_spec((self._observation_spec))

  def _reset(self):
        self.t  = self.tau
        self._state = [1] * self.num_IoTD + self.E_max + self.start_loc + [self.t - self.req_time(self.start_loc)] # state (AoI,E,Loc,t)
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))
#        return ts.transition(
#              np.array([self._state], dtype=np.int32), reward = 0, discount=1.0)

  def _step(self, act):
        
        

        if self._episode_ended:
          # The last action ended the episode. Ignore the current action and start
          # a new episode.
              return self._reset()
        wall_hit_cost = 0
        action = [ act[0]%5 , act[0]//5 ]
        
        
        if action[1]< 0 or action[1]>self.num_IoTD:
            raise ValueError('scehduling index is wrong')
        else:
            self.update(action[1])
            
        if action[0] == 0: # indicates left
            if self._state[self.num_IoTD*2]< 1: # hits the wall
                wall_hit_cost = self.wh_cost
            else:
                self._state[self.num_IoTD*2] = self._state[self.num_IoTD*2] - 1 # self.num_IoTD*2 is the index of x location
        elif action[0] == 1:  # indicates right
            if self._state[self.num_IoTD*2] > self.boundary[0]-1:# hits the wall
                wall_hit_cost = self.wh_cost
            else:
                self._state[self.num_IoTD*2] = self._state[self.num_IoTD*2] + 1 # self.num_IoTD*2 is the index of x location
        elif action[0] == 2: # indicates up
            if self._state[self.num_IoTD*2 +1] > self.boundary[1]-1 : # hits the wall
                wall_hit_cost = self.wh_cost
            else:
                self._state[self.num_IoTD*2 +1] = self._state[self.num_IoTD*2+ 1] + 1 # self.num_IoTD*2 + 1 is the index of y location
        elif action[0] == 3: # indicates down
            if self._state[self.num_IoTD*2 +1] < 1: # hits the wall
                wall_hit_cost = self.wh_cost
            else:
                self._state[self.num_IoTD*2 +1] = self._state[self.num_IoTD*2+ 1] - 1 # self.num_IoTD*2 + 1 is the index of y location
#        else:
#            raise ValueError('direction of movement should be l, r, u, or d and UAV should not go out of the boundaries')
        
        
        
        reward = - np.dot(self.w, self._state[0:self.num_IoTD]) - self.beta * self.distance(self._state[self.num_IoTD*2:self.num_IoTD*2+2]) + wall_hit_cost
        req_t = self.req_time(self._state[self.num_IoTD*2:self.num_IoTD*2+2])
        self._state[-1] = self.t - req_t
        self.t -=1
        if self._state[self.num_IoTD*2] == self.end_loc[0] and self._state[self.num_IoTD*2+1] == self.end_loc[1]:
            self._episode_ended = True #reached destination
            if self.t>0:
                reward += self.calculate_penalty()
#                reward += self.endtime_penalty
            
        elif self._state[-1] < 1: # minimum required time constraint will not satisfy
            self._episode_ended = True
#            reward += self.endtime_penalty
            reward += self.calculate_penalty() + self._state[-1]
            
        

    
        if self._episode_ended:
          return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
          return ts.transition(
              np.array([self._state], dtype=np.int32), reward = reward, discount=1.0)
      
      
      
  def req_time(self, loc):
        return np.abs(loc[0] - self.end_loc[0]) + np.abs(loc[1] - self.end_loc[1])
  
  def distance(self, loc):
        return np.abs(loc[0] - self.end_loc[0]) + np.abs(loc[1] - self.end_loc[1])
    
    
    
  def update(self, node_idx):
    for i in range(self.num_IoTD):
         if (self._state[i] < self.AoI_max[i]):
             self._state[i] = self._state[i] + 1
    if node_idx != 0:
        node_idx = node_idx - 1
        ### add meter conversion
        h = 1/(self.height**2 + (self.IoTD_loc[node_idx][0] - self._state[2*self.num_IoTD])**2  # 2*self.num_IoTD is the index of uav_x in the state vector
               + (self.IoTD_loc[node_idx][1] - self._state[2*self.num_IoTD+1])**2)/self.dist_conv**2 # 2*self.num_IoTD+1 is the index of uav_y in the state vector
        ### quantize E_bar
        E_bar = ((2**(self.C)-1)/h * self.sigma**2)//self.energy_conv + 1
        if self._state[self.num_IoTD + node_idx] > E_bar - 1: # self.num_IoTD + node_idx is the index of energy of node in the state vector
            self._state[self.num_IoTD + node_idx] -= E_bar    
            self._state[node_idx] = 1
    
  def calculate_penalty(self):
       penalty = 0
       AOI = self._state[0:self.num_IoTD]
       for _ in range(self.t):
           for i in range(self.num_IoTD):
               if (AOI[i] < self.AoI_max[i]):
                   AOI[i] = AOI[i] + 1
               penalty += AOI[i]*self.w[i]
       return -penalty