from machin.frame.algorithms import TD3
from machin.utils.logging import default_logger as logger
from torch.distributions import Categorical
import torch as t
import torch.nn as nn
import gym
import RobotDART as rd
import numpy as np
from math import cos, sin, degrees
import torch
from retry import retry
import pandas as pd
from torch.distributions import Normal
from torch.nn.functional import softplus
import matplotlib.pyplot as plt
import math


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




class Simulation:
	def __init__(self,e_graphics=0):
		timestep=0.004
		self.simu=rd.RobotDARTSimu(timestep)
		self.simu.set_collision_detector("fcl")
		if e_graphics==1:
			gconfig=rd.gui.GraphicsConfiguration(1024,768)
			self.graphics=rd.gui.Graphics(gconfig)



		self.pendulum=rd.Robot("pendulum.urdf")
		self.pendulum.fix_to_world()
		self.pendulum.set_actuator_types("torque")

		positions = self.pendulum.positions()
		positions[0]=np.pi
		self.pendulum.set_positions(positions)

		self.simu.add_robot(self.pendulum)

		self.pendulum.set_commands(np.array([3]))
		#
		if e_graphics==1:
			self.simu.set_graphics(self.graphics)
			self.graphics.look_at([1,1,1])



	def reset(self):
		positions=self.pendulum.positions()
		positions[0]=np.pi
		self.pendulum.set_positions(positions)

		initial_state=(positions[0],self.pendulum.velocities()[0])
		self.done=False
		return initial_state





	def step(self,velocity):
		#reward=0
		done=False
		#self.pendulum.set_commands(velocity)
		self.pendulum.set_external_torque(self.pendulum.body_name(1),[0.,velocity,0.])


		for _ in range(80):
			if self.simu.step_world():
				break

		theta= self.pendulum.positions()[0]
		
		velocities=self.pendulum.velocities()[0]
		

		current_angle=degrees(self.pendulum.positions()[0]) % 360
		

	

		
		reward = cos(theta)
		
		

		
		if current_angle>=270 and current_angle<355:
			reward=5 

		elif (current_angle>=355 and current_angle<=359.8):
			reward=10
			#done=True
		

		
		elif degrees(self.pendulum.positions()[0])>360 or degrees(self.pendulum.positions()[0])<0:
			reward=-10
			self.reset()
		if (current_angle==0) and (velocities==0.):
			
			done=True
		
			
		next_state=(theta,velocities)
		#print("reward="+str(reward))
		
		return next_state, reward, done

def td3_sim(print_status,max_episodes,max_steps,plots_enable=0):
    
    
	pendulum_sim=Simulation(plots_enable)
	observe_dim = 2
	action_dim = 1
	action_range=2
	solved_reward = 500
	solved_repeat =50
	data=[]
	angle_l=[]
	noise_param=(0,0.2)
	noise_mode="normal"
	



	actor = Actor(observe_dim, action_dim, action_range)
	actor_t=Actor(observe_dim, action_dim, action_range)
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
		nn.MSELoss(reduction="sum")

		
	)

	reward_fulfilled = 0
	smoothed_total_reward = 0
	angle_sum=0
	episode=0
	current_angle=degrees(pendulum_sim.pendulum.positions()[0])
	
	for episode in range(max_episodes + 1):
		#print("Episode="+str(episode))
		
#		print("EPISOD="+str(episode))
		total_reward = 0.0
		terminal = False
		step = 0
		state = t.tensor(pendulum_sim.reset(), dtype=t.float32).view(1, observe_dim)
		
		tmp_observations = []
		angle=0
		
		while (not terminal and step <= max_steps) :
			
			
	
			step+=1
				
			
			with t.no_grad():
				#old_state=state
				if episode<80:
					action=((2.0 * torch.rand(1,1) - 1.0) * action_range)
					torque=action
				else:
					action = td3.act_with_noise({"state": state},noise_param=noise_param,mode=noise_mode)
					torque=action[0]


				next_state, reward, terminal= pendulum_sim.step(torque)
				next_state = t.tensor(next_state, dtype=t.float32).view(1, observe_dim)
				total_reward +=reward
				#print("total_reward="+str(total_reward))

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

			angle=degrees(pendulum_sim.pendulum.positions()[0])%360
			
		
		current_data=[episode,total_reward,angle]
		data.append(current_data)
		
			
		
		print(f"Episode: [{episode:3d}/{max_episodes:3d}] Reward: {total_reward:.2f} Angle: {angle:.2f}",end="\r")
		print("",end="\n")


			

		
		
		# show reward
		smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
		# update
		td3.store_episode(tmp_observations)
		
	

		if episode>80:
			td3.update()



		if smoothed_total_reward >= solved_reward:
			reward_fulfilled += 1

			if reward_fulfilled > solved_repeat:
				print("Environment solved!")
				if plots_enable==1:
				#print("Average angle="+str(angle_sum/max_episodes))
					data_Dataframe=pd.DataFrame(data,columns=['Episode', 'Reward', 'Angle'])
				#print(m_angle)
					data_Dataframe.to_csv("td3_pendulum.csv")
					data_Dataframe.plot(x="Episode",y="Reward",kind="line")
					plt.show()
					plt.boxplot(data_Dataframe["Reward"])
					plt.show()
				return data

				#exit(0)
		else:
			reward_fulfilled = 0


	if plots_enable==1:
		data_Dataframe=pd.DataFrame(data,columns=['Episode', 'Reward', 'Angle'])
		data_Dataframe.to_csv("td3_pendulum.csv")
		data_Dataframe.plot(x="Episode",y="Reward",kind="line")
		plt.show()
		plt.boxplot(data_Dataframe["Reward"])
		plt.show()
	
	#return data_Dataframe
	return data
	


if __name__=='__main__':
	td3_sim(1,400,100,0)