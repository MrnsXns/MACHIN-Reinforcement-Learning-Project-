from machin.frame.algorithms import TD3
from machin.utils.logging import default_logger as logger
from torch.distributions import Categorical
import torch as t
import torch.nn as nn
import gym
import RobotDART as rd
import numpy as np
from math import cos, sin, degrees
from machin.frame.algorithms import PPO
import torch
from retry import retry
import pandas as pd
from torch.distributions import Normal
from torch.nn.functional import softplus
from math import sqrt
import matplotlib.pyplot as plt





# model definition
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.tanh(self.fc3(a)) * self.action_range
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(state_action))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q


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

        #reward=4+(target - current_pos)

        rot_error = np.linalg.norm(rd.math.logMap(robot_ghost_tf.rotation() @ tf.rotation().T))
        #print("rot="+str(rot_error))
        lin_error =np.linalg.norm(robot_ghost_tf.translation() - tf.translation())
        #print("lin="+str(lin_error))
        error_threshold=0.005

        dist_ee_goal=0.5*lin_error+0.5*rot_error
        #print("dist"+str(dist_ee_goal))

        if dist_ee_goal>2*error_threshold:
            reward=-dist_ee_goal

        else:
            reward=1

        next_state=(self.robot.positions())
        #reward=0
        #for pos in next_state:
        #    reward += pos * pos
        #reward = -sqrt(reward)

        
        
        #print("Reward="+str(reward))
        
        #error=np.linalg.norm(rd.math.logMap(tf.inverse().multiply(robot_ghost_tf)))
        
        #reward=error
        #print(reward)
        

        #next_state=(self.robot.positions())
        #error=np.linalg.norm(rd.math.logMap(tf.inverse().multiply(robot_ghost_tf)))
        #reward=20-error
        #print(reward)
        '''
        reward=0
        for pos in next_state:
            reward += pos * pos
        reward = -sqrt(reward)
        '''

        
        done=False
        if self.robot.positions==self.robot_ghost.positions:
            done=True

        return next_state,reward,done

    def reset(self):
        starting_state=[1,0,1,1,0,-1.2,1]
        

        self.robot.set_positions(starting_state)
        self.robot.set_positions(starting_state)
        return starting_state


#@retry(Exception, tries=5,delay=0,backoff=0)
def td3_sim(print_status,max_episodes,max_steps,plots_enable=0):
    
    
    iiwa_simulation=Simulation(plots_enable)
    observe_dim = 7
    action_dim = 7
    state_dim=7
    action_range = 1
    #max_episodes = 1000
    #max_steps = 200
    solved_reward = 500
    solved_repeat = 500
    data = []
    noise_param=(0,0.2)
    noise_mode="normal"
    angle_l=[]



    actor = Actor(observe_dim, action_dim, action_range)
    actor_t = Actor(observe_dim, action_dim, action_range)
    critic = Critic(observe_dim, action_dim)
    critic_t = Critic(observe_dim, action_dim)
    critic2 = Critic(observe_dim, action_dim)
    critic2_t = Critic(observe_dim, action_dim)

    
    td3 = TD3(
        actor,
        actor_t,
        critic,
        critic_t,
        critic2,
        critic2_t,
        t.optim.Adam,
        nn.MSELoss(reduction="sum"),
        batch_size=40
    )
    reward_fulfilled = 0
    smoothed_total_reward = 0.0



    for episode in range(1,max_episodes +1):
        episode_reward = 0.0
        terminal = False
        step = 0
        state = t.tensor(iiwa_simulation.reset(), dtype=t.float32).view(1, observe_dim)



        tmp_observations = []

        while not terminal and step < max_steps:
            step += 1
            with t.no_grad():
                # agent model inference
                #action = ppo.act({"state": old_state})[0]
                #state, reward, terminal, _ = pendulum_simulation.step(action.item())
                #state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                #total_reward += reward

                if episode < 80:
                    action=((2.0*t.rand(1,7) - 1.0) * action_range)
                    torque=np.transpose(action)
                else:
                    action = td3.act_with_noise({"state": state},noise_param=noise_param,mode=noise_mode)
                    torque=np.transpose(action)


                next_state,reward,terminal=iiwa_simulation.step(torque)
                next_state=t.tensor(next_state,dtype=t.float32).view(1,state_dim)
                episode_reward+=reward


                tmp_observations.append(
                    {
                        "state": {"state": state},
                        "action": {"action": action},
                        "next_state": {"state": next_state},
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

        td3.store_episode(tmp_observations)

        if episode > 80:
            for _ in range(step):
                td3.update()

        if smoothed_total_reward >= solved_reward:
            reward_fulfilled += 1

            if reward_fulfilled>=solved_repeat:
                print("Environment solved!")
                if plots_enable==1:
                    data_Dataframe=pd.DataFrame(data,columns=['Episode', 'Reward'])
                    data_Dataframe.to_csv("iiwa_ppo.csv")
                    data_Dataframe.plot(x="Episode",y="Reward",kind="line")
                    plt.show()
                    plt.boxplot(data_Dataframe["Reward"])
                    plt.show()
                    return data
        else:
            reward_fulfilled = 0
    if plots_enable==1:
        data_Dataframe=pd.DataFrame(data,columns=['Episode', 'Reward'])
        data_Dataframe.to_csv("iiwa_ppo.csv")
        data_Dataframe.plot(x="Episode",y="Reward",kind="line")
        plt.show()
        plt.boxplot(data_Dataframe["Reward"])
        plt.show()
        
    return data

    





if __name__ =='__main__':
   td3_sim(1,400,100,0)