# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:14:11 2019

@author: Aidin
"""

from environment import UAVGridEnvTime as env
import NonRLAgents
import scipy

num_iterations = 1000

def run_episode(environment,agent):
    time_step = environment.reset()
    episode_return = 0.0
    
    while not time_step.is_last():
      action_step = [agent.action(time_step.observation[0])]
      time_step = environment.step(action_step)
      episode_return += time_step.reward
    
    return episode_return

def run_episode_random(environment,agent):
    time_step = environment.reset()
    episode_return = 0.0
    
    while not time_step.is_last():
      action_step = [agent.action()]
      time_step = environment.step(action_step)
      episode_return += time_step.reward
    
    return episode_return

if __name__ =="__main__":
    # Environment
    num_IoTD = 1
    w =  [1]
    IoTD_loc = [[5,9]]
    AoI_max = [50]
    E_max = [100]
    boundary= [10,10]
    tau = 100
    start_loc= [0,5]
    end_loc =[10,5] 
    sigma = 10**(-7)
    C = 20
    beta = 0
    endtime_penalty = -100000
    height = 1
    wall_hit_cost  = 0
    energy_conv = 1.048575* 10**(-4)
    dist_conv = 100
    file_name = './results/Scenario_1_Dist_5_noRL.mat'
    random_walk_env = env(num_IoTD, w, IoTD_loc, AoI_max, E_max, boundary , tau, start_loc, end_loc, sigma, C , beta, endtime_penalty, height, wall_hit_cost,dist_conv,energy_conv)
    dist_based_env = env(num_IoTD, w, IoTD_loc, AoI_max, E_max, boundary , tau, start_loc, end_loc, sigma, C , beta, endtime_penalty, height, wall_hit_cost,dist_conv, energy_conv)
    
    agent_random = NonRLAgents.RandomWalkAgent(num_IoTD)
    agent_distance = NonRLAgents.DistanceAgent(num_IoTD , IoTD_loc)
    return_random = 0
    for _ in range(num_iterations):
        return_random += run_episode_random(random_walk_env,agent_random)
    
    return_random = return_random/num_iterations
    print('random_agent_return = {0}'.format(return_random))
    return_dist = run_episode(dist_based_env,agent_distance)
    print('dist_agent_return = {0}'.format(return_dist))
    scipy.io.savemat(file_name, dict(return_random = return_random,
                                     return_dist = return_dist))