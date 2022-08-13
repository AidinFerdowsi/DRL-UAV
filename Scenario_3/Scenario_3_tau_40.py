# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:30:04 2019

@author: Aidin
"""


import numpy as np
import tensorflow as tf
import scipy.io

from environment import UAVGridEnvTime as env

#from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import trajectory

from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent

from tf_agents.policies import random_tf_policy

from tf_agents.replay_buffers import tf_uniform_replay_buffer

from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()

# Global hyperparams

initial_collect_steps = 1000
collect_steps_per_iteration = 1
replay_buffer_capacity = 2000000

fc_layer_params = (200,)

learning_rate = 1e-3
batch_size = 200


num_iterations = 500000
num_eval_episodes = 100
eval_interval = 10000
log_interval = 1000
save_interval = 10000

# Agent
class Agent:
    def __init__(self,train_env,fc_layer_params,learning_rate):

        q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)
    
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        train_step_counter = tf.compat.v2.Variable(0)
    
        self.tf_agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=dqn_agent.element_wise_squared_loss,
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
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


def state_action(environment, policy):



    time_step = environment.reset()
    state = [time_step.observation.numpy()]
    actions = [] 

    while not time_step.is_last():
      action_step = policy.action(time_step)
      actions.append(action_step.action.numpy())
      time_step = environment.step(action_step.action)
      state.append(time_step.observation.numpy())

  
    return state, actions



# Data Collection

def collect_step(environment, policy):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  replay_buffer.add_batch(traj)
  

if __name__ =="__main__":
    # Environment
    num_IoTD = 3
    w =  [1/3] * 3
    IoTD_loc = [[5  ,   10],
                [0  ,   0],
                [10  ,  0]]
    AoI_max = [50] * 3
    E_max = [100] * 3
    boundary= [10,10]
    tau = 40
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
    file_name = './results/Scenario_3_tau_40.mat'
	
    train_py_env = env(num_IoTD, w, IoTD_loc, AoI_max, E_max, boundary , tau, start_loc, end_loc, sigma, C , beta, endtime_penalty, height, wall_hit_cost,dist_conv,energy_conv)
    eval_py_env = env(num_IoTD, w, IoTD_loc, AoI_max, E_max, boundary , tau, start_loc, end_loc, sigma, C , beta, endtime_penalty, height, wall_hit_cost,dist_conv, energy_conv)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    train_env.reset()
    eval_env.reset()
    
    Agent = Agent(train_env,fc_layer_params,learning_rate)
    tf_agent = Agent.tf_agent
    tf_agent.initialize()
    
    # Replay Buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=tf_agent.collect_data_spec,
                batch_size=train_env.batch_size,
                max_length=replay_buffer_capacity)
    
    
    collect_policy = tf_agent.collect_policy
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                        train_env.action_spec())
    
    for _ in range(initial_collect_steps):
      collect_step(train_env, random_policy)


    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)

    iterator = iter(dataset)


    # Training 
    
    tf_agent.train = common.function(tf_agent.train)

    # Reset agent
    tf_agent.train_step_counter.assign(0)

    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    returns = np.empty(num_iterations//eval_interval + 1)
    returns[0] = avg_return
    cnt = 1
    state_episodes = []
    action_episodes = []

    for _ in range(num_iterations):
    
      # Collect one step using collect_policy and save to the replay buffer.
      collect_step(train_env, tf_agent.collect_policy)
    
      # Sample a batch of data from the buffer and update the agent's network.
      experience, unused_info = next(iterator)
      train_loss = tf_agent.train(experience)
    
      step = tf_agent.train_step_counter.numpy()
    
      if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))
    
      if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns[cnt] = avg_return
        cnt += 1
        scipy.io.savemat(file_name, dict(returns = returns,
                                                      state_episodes = state_episodes,
                                                      action_episodes = action_episodes,
                                                      num_IoTD = num_IoTD, 
                                                      w = w, 
                                                      IoTD_loc = IoTD_loc,
                                                      AoI_max = AoI_max,
                                                      E_max = E_max,
                                                      boundary = boundary,
                                                      tau = tau,
                                                      start_loc = start_loc,
                                                      end_loc = end_loc,
                                                      sigma = sigma,
                                                      C = C,
                                                      beta = beta,
                                                      endtime_penalty = endtime_penalty,
                                                      height = height,
                                                      wall_hit_cost = wall_hit_cost,
                                                      energy_conv = energy_conv,
                                                      dist_conv = dist_conv,
                                                      ))
        
      #if (step + 1) % save_interval == 0:
      #  state, actions = state_action(eval_env, tf_agent.policy)
      #  state_episodes.append(state)
      #  action_episodes.append(actions)
    
#    scipy.io.savemat(file_name, dict(returns = returns,
#                                                      state_episodes = state_episodes,
#                                                      action_episodes = action_episodes,
#                                                      num_IoTD = num_IoTD, 
#                                                      w = w, 
#                                                      IoTD_loc = IoTD_loc,
#                                                      AoI_max = AoI_max,
#                                                      E_max = E_max,
#                                                      boundary = boundary,
#                                                      tau = tau,
#                                                      start_loc = start_loc,
#                                                      end_loc = end_loc,
#                                                      sigma = sigma,
#                                                      C = C,
#                                                      beta = beta,
#                                                      endtime_penalty = endtime_penalty,
#                                                      height = height,
#                                                      wall_hit_cost = wall_hit_cost,
#                                                      energy_conv = energy_conv,
#                                                      dist_conv = dist_conv,
#                                                      ))
    
    # Plots
    
#    steps = range(0, num_iterations + 1, eval_interval)
#    plt.plot(steps, returns)
#    plt.ylabel('Average Return')
#    plt.xlabel('Step')
#    plt.ylim(top=250)
