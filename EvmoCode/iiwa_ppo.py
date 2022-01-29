from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger
from torch.distributions import Categorical
import torch as t
import torch.nn as nn
import gym
import RobotDART as rd
import numpy as np
from math import cos, sin, degrees

import torch

import pandas as pd
from torch.distributions import Normal
from torch.nn.functional import softplus
import matplotlib.pyplot as plt
import math

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)
        self.mu_head = nn.Linear(16, action_dim)
        self.sigma_head = nn.Linear(16, action_dim)
        self.action_range = 1


    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        mu = self.mu_head(a)
        sigma = softplus(self.sigma_head(a))
        dist = Normal(mu, sigma)
        act = (action
               if action is not None
               else dist.sample())
        act_entropy = dist.entropy().sum(1, keepdim=True)
        act_log_prob = dist.log_prob(act)
        act_tanh = t.tanh(act)
        act = act_tanh * self.action_range

        # the distribution remapping process used in the original essay.
        act_log_prob -= t.log(self.action_range *
                              (1 - act_tanh.pow(2)) +
                              1e-6)
        act_log_prob = act_log_prob.sum(1, keepdim=True)

        return act, act_log_prob, act_entropy



class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state):
        v = t.relu(self.fc1(state))
        v = t.relu(self.fc2(v))
        v = self.fc3(v)
        return v



class Simulation():
    def __init__(self,e_graphics):
        timestep=0.004
        self.simu=rd.RobotDARTSimu(timestep)
        self.simu.set_collision_detector("fcl")
        if e_graphics==1:
            gconfig=rd.gui.GraphicsConfiguration(1024,768)
            self.graphics=rd.gui.Graphics(gconfig)


        packages= [("iiwa_description","iiwa/iiwa_description")]
        self.robot=rd.Robot("iiwa/iiwa.urdf",packages)
        self.robot.fix_to_world()
        self.robot.set_actuator_types("servo")

        positions=self.robot.positions()
        self.robot.set_positions(positions)

        self.robot_ghost=self.robot.clone_ghost()
        self.first_pos=[1,0,1,1,0,-1.2,1]
        self.robot.set_positions(self.first_pos)

        

        self.simu.add_robot(self.robot)
        self.simu.add_robot(self.robot_ghost)


        self.simu.add_floor()

        if e_graphics==1:

            self.simu.set_graphics(self.graphics)
            self.graphics.look_at([3., 1., 2.], [0., 0., 0.])


    def step(self,positions):
            
        self.robot.set_positions(positions)
        robot_ghost_tf=self.robot_ghost.body_pose("iiwa_link_ee")
        tf=self.robot.body_pose("iiwa_link_ee")
        
        for _ in range(80):
            if self.simu.step_world():
                break

    

        rot_error = np.linalg.norm(rd.math.logMap(robot_ghost_tf.rotation() @ tf.rotation().T))
        
        lin_error =np.linalg.norm(robot_ghost_tf.translation() - tf.translation())
        
        error_threshold=0.005

        dist_ee_goal=0.5*lin_error+0.5*rot_error
        

        if dist_ee_goal>2*error_threshold:
            reward=-dist_ee_goal

        else:
            reward=1

        next_state=(self.robot.positions())
        
        
        done=False
       # if self.robot.positions==self.robot_ghost.positions:
       #     done=True

        return next_state,reward,done

    def reset(self):
        starting_state=[1,0,1,1,0,-1.2,1]
        

        self.robot.set_positions(starting_state)
        self.robot.set_positions(starting_state)
        return starting_state



def ppo_sim(print_statues,max_episodes,max_steps,enable_plots=0):
    #graphics=input("Do you want graphics?(n or y)")
    #if graphics=='y':
    #    iiwa_simumation=Simulation(1)
    #elif graphics=='n':
    #    iiwa_simulation=Simulation(0)
    #else:
     #   print("Wrong choice.Please try again")
     #   exit(0)
    iiwa_sim=Simulation(enable_plots)
    action_dim = 7
    state_dim=7
    action_range = 1
    action_num=7
    solved_reward = 100
    solved_repeat = 15
    data = []
  



    actor = Actor(state_dim, action_num)
    critic = Critic(state_dim)

    
    ppo = PPO(actor, critic, t.optim.Adam, nn.MSELoss(reduction="sum"),batch__size=20)

    reward_fulfilled = 0
    smoothed_total_reward = 0.0



    for episode in range(1,max_episodes +1):
        episode_reward = 0.0
        terminal = False
        step = 0
        state = t.tensor(iiwa_sim.reset(), dtype=t.float32).view(1, state_dim)



        tmp_observations = []

        while not terminal and step < max_steps:
            step += 1
            with t.no_grad():
    
                if episode < 80:
                    action=((2.0*t.rand(1,7) - 1.0) * action_range)
                    torque=np.transpose(action)
                else:
                    action = ppo.act({"state": state})[0]
                    torque=np.transpose(action)


                next_state,reward,terminal=iiwa_sim.step(torque)
                next_state=t.tensor(next_state,dtype=t.float32).view(1,state_dim)
                episode_reward += reward


                tmp_observations.append(
                    {
                        "state": {"state": state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward,
                        "terminal": terminal or step == max_steps,
                    }
                )
                state=next_state
            
        current_data=[episode,episode_reward]
        data.append(current_data)
        print(f"Episode:[{episode:4d}/{max_episodes:4d}] Step: [{step:4d}/{max_steps:4d}] Reward: [{episode_reward:.2f}]",end="\r")
        print("",end="\n")
        smoothed_total_reward = smoothed_total_reward * 0.9 + episode_reward * 0.1

        ppo.store_episode(tmp_observations)

        if episode > 80:
            ppo.update()

        if smoothed_total_reward >= solved_reward:
            reward_fulfilled += 1

            if reward_fulfilled>=solved_repeat:
                print("Environment solved!")
                data_Dataframe=pd.DataFrame(data,columns=['Episode', 'Reward'])
                data_Dataframe.to_csv("iiwa_ppo.csv")
                data_Dataframe.plot(x="Episode",y="Reward",kind="line")
                plt.show()
                plt.boxplot(data_Dataframe["Reward"])
                plt.show()
                return data
        else:
            reward_fulfilled = 0

    data_Dataframe=pd.DataFrame(data,columns=['Episode', 'Reward'])
    data_Dataframe.to_csv("iiwa_ppo.csv")
    data_Dataframe.plot(x="Episode",y="Reward",kind="line")
    plt.show()
    plt.boxplot(data_Dataframe["Reward"])
    plt.show()
    
    return data




if __name__ =='__main__':
   ppo_sim(1,400,100,0)

