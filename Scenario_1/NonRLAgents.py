# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:10:24 2019

@author: Aidin
"""
import random
import numpy as np

class RandomWalkAgent:
    def __init__(self, num_IoTD):
        
        self.num_IoTD = num_IoTD
        self.dir = range(5)
        self.sch = range(num_IoTD + 1)
        
        
    def action(self):
        
        direction = random.choice(self.dir)
        schedule = random.choice(self.sch)
        
        return direction + 5 * schedule
    
    
class DistanceAgent:
    def __init__(self, num_IoTD, IoTD_loc):
        
        self.num_IoTD = num_IoTD
        self.dir = range(5)
        self.sch = range(num_IoTD)
        self.IoTD_loc = IoTD_loc
    
    def action(self, state):
        '''
          State space consists of the following:
              - AoI for every IoTD: 0 - AoI_max[i] -> size: num_IoTD
              - Energy level for every IoTD: 0 - E_max[i] -> size: num_IoTD
              - UAV Location: x,y: 0,..., boundary[0] & 0,...,boundary[1] -> size: 2
              - Time left: Time constraint - Required Time: 0, tau #Note that at every 
              time step we have to reduce the time constrant -> size: 1
         '''
        AoI = state[0 : self.num_IoTD]
        E = state[self.num_IoTD:self.num_IoTD * 2]
        UAV_Loc = state[self.num_IoTD * 2:self.num_IoTD * 2 + 2]
        dist = self.distance(UAV_Loc)
        sortedDistIndex = sorted(range(len(dist)), key=lambda k: dist[k])
        sortedAoIIndex = sorted(range(len(AoI)), key=lambda k: AoI[k])
        if dist[sortedDistIndex[0]] < 2:
            schedule = sortedDistIndex[0]
        else:
            schedule = -1
        direction = random.choice(self.dir)
        for i in range(self.num_IoTD):
            if E[sortedAoIIndex[self.num_IoTD - 1 - i]] > 0:
                xdist = UAV_Loc[0] - self.IoTD_loc[sortedAoIIndex[self.num_IoTD - 1 - i]][0]
                ydist = UAV_Loc[1] - self.IoTD_loc[sortedAoIIndex[self.num_IoTD - 1 - i]][1]
                if np.abs(xdist) == 0 and  np.abs(ydist) == 0:
                    direction = 4 #stand steal 
                    
                elif np.abs(xdist) > np.abs(ydist):
                    if xdist > 0: # should go to the left
                        direction = 0
                    else: # should go to the right
                        direction = 1
                else:
                    if ydist > 0: # should go down
                        direction = 3
                    else: # should go up
                        direction = 2
                
                break
        
        return direction + (schedule + 1) * 5
                    
        
        
    def distance(self, loc):
        dist = [0] * self.num_IoTD
        for i in range(self.num_IoTD):
            dist[i] = np.abs(loc[0] - self.IoTD_loc[i][0]) + np.abs(loc[1] - self.IoTD_loc[i][1])
        return dist
         