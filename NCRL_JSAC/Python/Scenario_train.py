# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:30:04 2019

@author: Aidin
"""

#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io
import sys
import os


sys.path.insert(1, '../')
from environment_comb import UAVGridEnvTime as env
from environment_comb import UAVGridEnvTime as env_eval

#from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import trajectory

from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent

from tf_agents.policies import random_tf_policy

from tf_agents.replay_buffers import tf_uniform_replay_buffer

from tf_agents.utils import common


tf.compat.v1.enable_v2_behavior()

tf.version.VERSION


# Global hyperparams

initial_collect_steps = 1000
collect_steps_per_iteration = 1
replay_buffer_capacity = 100000

fc_layer_params = (200,)

learning_rate = 1e-3
batch_size = 128


num_iterations = 1000000
num_eval_episodes = 100
eval_interval = 5000
log_interval = 1000
save_interval = 5000
target_update_period = 10


# Agent
class Agent:
    def __init__(self,train_env,fc_layer_params,learning_rate):

        q_net = q_network.QNetwork(
                train_env.observation_spec(),
                train_env.action_spec(),
                fc_layer_params=fc_layer_params)
        
        
        target_q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)
    
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        train_step_counter = tf.compat.v2.Variable(0)
    
        self.tf_agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            target_q_network = target_q_net,
            target_update_period = target_update_period,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)

    

# Policy 
class Policy:
    def __init__(self,tf_agent,train_env):
        
        self.eval_policy = tf_agent.policy
        self.collect_policy = tf_agent.collect_policy
        
        self.random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                        train_env.action_spec())


# Average Return

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    
    while not time_step.is_last():
      action_step =  policy.action(time_step)
          
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += (1-episode_return)

  avg_return = float(total_return) / float(num_episodes)
#  print ("average return is: {}".format(avg_return))
  return avg_return


def state_action(environment, policy):



    time_step = environment.reset()
    state = [time_step.observation.numpy()]
    actions = []
    while not time_step.is_last():
      
      action_step =  policy.action(time_step)
      actions.append(action_step.action.numpy())
      time_step = environment.step(action_step.action)
      state.append(time_step.observation.numpy())

  
    return state, actions



            
            


# Data Collection

def collect_step(environment, policy,RB):
  time_step = environment.current_time_step()
#  print(time_step)
     
  action_step =  policy.action(time_step)
  
#  print(action_step)
  
  next_time_step = environment.step(action_step.action)
  # Add trajectory to the replay buffer
  
  traj = trajectory.from_transition(time_step, action_step, next_time_step)
  RB.add_batch(traj)

  

if __name__ =="__main__":
    # Environment
    num_IoTD = int(sys.argv[1])
    
    sigma_2 = 10**(-13)
    B = 1
    S = 10
    beta = 0.00002
    e_max = 1
    e_min = 0.1
    
    E_max = e_max * beta/(2**(S/B)-1)/sigma_2
    E_min = e_min * beta/(2**(S/B)-1)/sigma_2
    
    
    T_max = 1200
    T_min = 600 
    V_max = 30
    V_min = 20
    height_max = 90
    height_min = 70
    x_max = 1000
    y_max = 1000
    dir_name = './results'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    log_file_name = dir_name + '/num_IoTD_{}.txt'.format(num_IoTD) 
    
    
    file_name = dir_name + '/num_IoTD_{}.mat'.format(num_IoTD)
    train_py_env = env(num_IoTD, E_max,E_min,T_max,T_min,V_max,V_min,height_max,height_min,x_max,y_max)
    eval_py_env = env_eval(num_IoTD, E_max,E_min,T_max,T_min,V_max,V_min,height_max,height_min,x_max,y_max)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    train_env.reset()
    eval_env.reset()

    
    
    Agent = Agent(train_env,fc_layer_params,learning_rate)
    tf_agent = Agent.tf_agent
    tf_agent.initialize()
    
    # Replay Buffer
    RB = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=tf_agent.collect_data_spec,
                batch_size=train_env.batch_size,
                max_length=replay_buffer_capacity)
    
    
    collect_policy = tf_agent.collect_policy
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                            train_env.action_spec())
    
    for ie in range(initial_collect_steps):
        print("Initial step {}".format(ie))
        collect_step(train_env, random_policy,RB)


    
    dataset = RB.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
    
    print("step 1 passed")
    iterator = iter(dataset)


        # Training 
        
    tf_agent.train = common.function(tf_agent.train)

        # Reset agent
    tf_agent.train_step_counter.assign(0)

    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    
    print("step 2 passed")
    returns = np.empty(num_iterations//eval_interval + 1)
    returns[0] = avg_return
    cnt = 1
    state_episodes = []
    action_episodes = []
    state, actions = state_action(eval_env, tf_agent.policy)
    state_episodes.append(state)
    action_episodes.append(actions)

    for _ in range(num_iterations):
    
      # Collect one step using collect_policy and save to the replay buffer.
      collect_step(train_env, collect_policy, RB)
      # Sample a batch of data from the buffer and update the agent's network.
      experience, unused_info = next(iterator)
      train_loss = tf_agent.train(experience)
      LOSS = train_loss.loss
    
      step = tf_agent.train_step_counter.numpy()
      
      

      if step % log_interval == 0:
           
        print('step = {0}: Average loss = {1}'.format(step, LOSS))
#        with open(log_file_name,"a") as log_file:
#          log_file.write('step = {0}: Average loss = {1}\n'.format(step, LOSS))
    
      if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
#        with open(log_file_name,"a") as log_file:
#          log_file.write('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns[cnt] = avg_return
        cnt += 1
        
      if (step) % save_interval == 0:
        state, actions = state_action(eval_env, tf_agent.policy )
        state_episodes.append(state)
        action_episodes.append(actions)
        scipy.io.savemat(file_name, dict(returns = returns,
                                          state_episodes = state_episodes,
                                          action_episodes = action_episodes,
                                          num_IoTD = num_IoTD, 
                                          E_max = E_max,
                                          E_min = E_min,
                                          T_max = T_max,
                                          T_min = T_min, 
                                          V_max = V_max,
                                          V_min = V_min,
                                          height_max = height_max,
                                          height_min = height_min,
                                          x_max = x_max,
                                          y_max = y_max,
                                          ))
        