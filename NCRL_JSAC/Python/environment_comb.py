# -*- coding: utf-8 -*-
"""
Created on Thu Apr  18 18:33:54 2020

@author: Aidin
"""

import numpy as np

import matlab.engine

eng = matlab.engine.start_matlab()

from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment

from tf_agents.specs import array_spec


FIRST = ts.StepType.FIRST
MID = ts.StepType.MID
LAST = ts.StepType.LAST

class UAVGridEnvTime(py_environment.PyEnvironment):

  def __init__(self, num_IoTD, E_max,E_min,T_max,T_min,V_max,V_min,height_max,height_min,x_max,y_max):
      self.num_IoTD = num_IoTD # number of IoT devices
      self.E_max = E_max # the vector with components equal to the maximum available energy of each IoTD
      self.E_min = E_min # the vector with components equal to the minimum available energy of each IoTD
      self.T_max = T_max # the time limit max
      self.T_min = T_min # the time limit min
      self.V_max = V_max # speed maximum 
      self.V_min = V_min # speed minimum 
      self.height_min = height_min # minimum height at which the UAV is flying (nodes are at 0)
      self.height_max = height_max # minimum height at which the UAV is flying (nodes are at 0)
      self.x_max = x_max # max x coordinate
      self.y_max = y_max # max y coordinate
      self.nodes_so_far = []
      self.prev_aoi = 1
      self.penalty = -10
    
      '''
      Action space consists of the following:
          - Scheduling: 1, ..., num_IoTD
      '''
      self._action_spec = array_spec.BoundedArraySpec(
              shape=(1,), dtype=np.int32, minimum=0, maximum=self.num_IoTD, name='action')
      
      '''
      State space consists of the following:
          [0] : Update time
          [1:num_IoTD+1] : Energy_level of all IoTDs
          [num_IoTD + 1] : x_coordinate difference between UAV and final location
          [num_IoTD + 2 : 2*num_IoTD + 2]: x_coordinate difference of IoT devices and UAV
          [2 * num_IoTD + 2] : y_coordinate difference between UAV and final location
          [2 * num_IoTD + 3 : 3*num_IoTD + 3]: y_coordinate difference of IoT devices and UAV
          [3*num_IoTD + 3 : 4*num_IoTD + 3]: IoTD values
          [4*num_IoTD + 3]: Time constraint
          [4*num_IoTD + 4]: speed 
          [4*num_IoTD + 5]: height
          
      '''
      self._observation_spec = array_spec.BoundedArraySpec(
              shape=(1,self.num_IoTD * 4 + 6), dtype=np.float32, 
              minimum =  [-1] * (self.num_IoTD * 4 + 6), maximum =  [1] * (self.num_IoTD * 4 + 6),
              name='observation')
      self._state = [0] * (self.num_IoTD * 4 + 6) # state (t,E,Loc,T,vmax,h,lambda)
      self._state[0] = -1
      self.E = np.random.uniform(self.E_min/self.E_max,1,(self.num_IoTD))
      self._state[1:num_IoTD+1] = self.E * 2 - 1
      self.xs = np.random.uniform(0,1,1)
      self.ys = np.random.uniform(0,1,1)
      self.xi = np.random.uniform(0,1,self.num_IoTD + 1)
      self.yi = np.random.uniform(0,1,self.num_IoTD + 1)
      self._state[num_IoTD + 1 : 2*num_IoTD + 2] = self.xi - self.xs
      self._state[2 * num_IoTD + 2 : 3*num_IoTD + 3] = self.yi - self.ys
      self.w = np.random.uniform(0,1,self.num_IoTD)
      self.w = self.w/np.sum(self.w)
      self._state[3*num_IoTD + 3 : 4*num_IoTD + 3] = self.w*2-1
      self._state[4*num_IoTD + 3] = np.random.uniform(-1,1,1)
      self._state[4*num_IoTD + 4] = np.random.uniform(-1,1,1)
      self._state[4*num_IoTD + 5] = np.random.uniform(-1,1,1)
      self._episode_ended = False
      
      self._current_time_step = ts.transition(
              np.array([self._state], dtype=np.float32), reward = 0, discount=1.0)

  def action_spec(self):
      return self._action_spec

  def observation_spec(self):
      return self._observation_spec
  
#  def time_step_spec(self):
#      return ts.time_step_spec((self._observation_spec))

  def _reset(self):
        self._state = [0] * (self.num_IoTD * 4 + 6) # state (t,E,Loc,T,vmax,h,lambda)
        self._state[0] = -1
        self.E = np.random.uniform(self.E_min/self.E_max,1,(self.num_IoTD))
        self._state[1:self.num_IoTD+1] = self.E * 2 - 1
        self.xs = np.random.uniform(0,1,1)
        self.ys = np.random.uniform(0,1,1)
        self.xi = np.random.uniform(0,1,self.num_IoTD + 1)
        self.yi = np.random.uniform(0,1,self.num_IoTD + 1)
        self._state[self.num_IoTD + 1 : 2*self.num_IoTD + 2] = self.xi - self.xs
        self._state[2 * self.num_IoTD + 2 : 3*self.num_IoTD + 3] = self.yi - self.ys
        self.w = np.random.uniform(0,1,self.num_IoTD)
        self.w = self.w/np.sum(self.w)
        self._state[3*self.num_IoTD + 3 : 4*self.num_IoTD + 3] = self.w*2-1
        self._state[4*self.num_IoTD + 3] = np.random.uniform(-1,1,1)
        self._state[4*self.num_IoTD + 4] = np.random.uniform(-1,1,1)
        self._state[4*self.num_IoTD + 5] = np.random.uniform(-1,1,1)
        self._episode_ended = False
        self.nodes_so_far = []
        self.prev_aoi = 1
        return ts.restart(np.array([self._state], dtype=np.float32))
#        return ts.transition(
#              np.array([self._state], dtype=np.int32), reward = 0, discount=1.0)

  def _step(self, act):
        
        

        
            
        if self._episode_ended:
          # The last action ended the episode. Ignore the current action and start
          # a new episode.
              return self._reset()
        
        
        
        if act [0].tolist() == 0:
            self._episode_ended = True
            reward = 0
        
        else:
            
            self.nodes_so_far.append(act[0].tolist())
            
    
            aoi, has_solution = self.optimal_aoi()
            
    #        print(self.nodes_so_far)  
    #        print(self.prev_aoi)
            
    #        print(has_solution)
            
            if has_solution:
                reward = self.prev_aoi - aoi
                self.prev_aoi = aoi
    #            self._episode_ended = self.check_terminal()
            
            else:
                self._episode_ended = True
                reward = self.prev_aoi - aoi
                self.prev_aoi = 1
    #        print(self._episode_ended)   
        
        
    
        if self._episode_ended:
          return ts.termination(np.array([self._state], dtype=np.float32), reward)
        else:
          return ts.transition(
              np.array([self._state], dtype=np.float32), reward = reward, discount=1.0)
      
      
      
  def optimal_aoi(self):
      T = (self._state[4*self.num_IoTD + 3] + 1) / 2 * (self.T_max - self.T_min) + self.T_min
      T =  matlab.double(T.tolist(), size=(1, 1))
      vmax = (self._state[4*self.num_IoTD + 4] + 1) / 2 * (self.V_max - self.V_min) + self.V_min
      vmax = matlab.double(vmax.tolist(), size=(1, 1))
      E = (self.E + 1) / 2 * self.E_max
      E = matlab.double(E.tolist() , size=(1, len(E)))
      x0 = self.xs * self.x_max
      x0 = matlab.double(x0.tolist(), size=(1, 1))
      y0 = self.ys * self.y_max
      y0 = matlab.double(y0.tolist(), size=(1, 1))
      xf = self.xi[-1] * self.x_max
      xf = matlab.double([xf], size=(1, 1))
      yf = self.yi[-1] * self.y_max
      yf = matlab.double([yf], size=(1, 1))
      xi = self.xi[0:-1] * self.x_max
      xi = matlab.double(xi.tolist(), size=(1, len(xi)))
      yi = self.yi[0:-1] * self.y_max
      yi = matlab.double(yi.tolist(), size=(1, len(yi)))
      h = (self._state[-1] + 1)  / 2 * (self.height_max - self.height_min) + self.height_min
      h = matlab.double(h.tolist(), size=(1, 1))
      w = matlab.double(self.w.tolist(), size=(1, len(self.w)))
      u = matlab.double(self.nodes_so_far, size=(1, len(self.nodes_so_far)))
      t,x,y,e,aoi = eng.AoIminimizer_multiIoT(T,vmax,E,x0,y0,xf,yf,xi,yi,h,w,u,nargout=5)
      t = np.array(t, ndmin = 2)
      x = np.array(x,ndmin = 2)
      y = np.array(y,ndmin = 2)
      e = np.array(e,ndmin = 1)
#      print(e)
      aoi = np.array(aoi)
#      print(self._state)
      if np.isnan(aoi) or np.isinf(aoi):
          has_solution = False
          self._state[0] = -1
          self._state[1:self.num_IoTD+1] = self.E
          self._state[self.num_IoTD + 1 : 2*self.num_IoTD + 2] = self.xi - self.xs
          self._state[2 * self.num_IoTD + 2 : 3*self.num_IoTD + 3] = self.yi - self.ys
          aoi = 1
          
      else:
          has_solution = True
          
#          if t.shape[0] > 1:
#              print(t)
#              self._state[0] = t[0][-1]/T[0][0] * 2 - 1
#          else:
          self._state[0] = t[:,-1]/T[0][0]*2-1
          self._state[1:self.num_IoTD+1] = e[:,-1]/self.E_max * 2 - 1
          self._state[self.num_IoTD + 1 : 2*self.num_IoTD + 2] = self.xi - x[:,-1]/self.x_max
          self._state[2 * self.num_IoTD + 2 : 3*self.num_IoTD + 3] = self.yi - y[:,-1]/self.y_max
          
#      print(self._state)
      return aoi, has_solution
  
    
  def check_terminal(self):
      T = (self._state[4*self.num_IoTD + 3] + 1) / 2 * (self.T_max - self.T_min) + self.T_min
      T =  matlab.double(T.tolist(), size=(1, 1))
      vmax = (self._state[4*self.num_IoTD + 4] + 1) / 2 * (self.V_max - self.V_min) + self.V_min
      vmax = matlab.double(vmax.tolist(), size=(1, 1))
      E = (self.E + 1) / 2 * self.E_max
      E = matlab.double(E.tolist() , size=(1, len(E)))
      x0 = self.xs * self.x_max
      x0 = matlab.double(x0.tolist(), size=(1, 1))
      y0 = self.ys * self.y_max
      y0 = matlab.double(y0.tolist(), size=(1, 1))
      xf = self.xi[-1] * self.x_max
      xf = matlab.double([xf], size=(1, 1))
      yf = self.yi[-1] * self.y_max
      yf = matlab.double([yf], size=(1, 1))
      xi = self.xi[0:-1] * self.x_max
      xi = matlab.double(xi.tolist(), size=(1, len(xi)))
      yi = self.yi[0:-1] * self.y_max
      yi = matlab.double(yi.tolist(), size=(1, len(yi)))
      h = (self._state[-1] + 1)  / 2 * (self.height_max - self.height_min) + self.height_min
      h = matlab.double(h.tolist(), size=(1, 1))
      w = matlab.double(self.w.tolist(), size=(1, len(self.w)))
      u = matlab.double(self.nodes_so_far, size=(1, len(self.nodes_so_far)))
      n = matlab.double([self.num_IoTD], size=(1,1))
      
      is_terminal = eng.check_terminal(T,vmax,E,x0,y0,xf,yf,xi,yi,h,w,u,n,nargout=1)
      
      if is_terminal == 1:
          return True
      else:
          return False