from machin.frame.algorithms import SAC
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


def atanh(x):
	return 0.5 * t.log((1 + x) / (1 - x))



class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, action_range):
		super().__init__()

		self.fc1 = nn.Linear(state_dim, 16)
		self.fc2 = nn.Linear(16, 16)
		self.mu_head = nn.Linear(16, action_dim)
		self.sigma_head = nn.Linear(16, action_dim)
		self.action_range = action_range

	def forward(self, state, action=None):
		a = t.relu(self.fc1(state))
		a = t.relu(self.fc2(a))

		mu = self.mu_head(a)
		sigma = softplus(self.sigma_head(a))
		
		dist = Normal(mu, sigma)
		act = (
			atanh(action / self.action_range) if action is not None else dist.rsample()
		)
		act_entropy = dist.entropy()

		# the suggested way to confine your actions within a valid range
		# is not clamping, but remapping the distribution
		act_log_prob = dist.log_prob(act)
		act_tanh = t.tanh(act)
		act = act_tanh * self.action_range

		# the distribution remapping process used in the original essay.
		act_log_prob -= t.log(self.action_range * (1 - act_tanh.pow(2)) + 1e-6)
		act_log_prob = act_log_prob.sum(1, keepdim=True)

		return act, act_log_prob, act_entropy


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


		for _ in range(20):
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

def sac_sim(print_status,max_episodes,max_steps,plots_enable=0):
	#graphics=input("Do you want graphics?(n or y)")
	#if graphics=='y':
	#	pendulum_sim=Simulation(1)
	#elif graphics=='n':
	#	pendulum_sim=Simulation(0)
	#else:
	#	print("Wrong choice.Please try again")
	#	exit(0)
	pendulum_sim=Simulation(plots_enable)
	observe_dim = 2
	action_dim = 1
	action_range=2
	solved_reward = 500
	solved_repeat =15
	data=[]
	angle_l=[]
	



	actor = Actor(observe_dim, action_dim, action_range)
	critic = Critic(observe_dim, action_dim)
	critic_t = Critic(observe_dim, action_dim)
	critic2 = Critic(observe_dim, action_dim)
	critic2_t = Critic(observe_dim, action_dim)

	sac = SAC(
		actor,
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
				old_state=state
				if episode<80:
					action=(2.0 * torch.rand(1,1) - 1.0) * action_range
					torque=action
				else:
					action=sac.act({"state":old_state})[0]
					torque=action[0]


				next_state, reward, terminal= pendulum_sim.step(torque)
				next_state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
				total_reward +=reward
				#print("total_reward="+str(total_reward))

				tmp_observations.append(
					{
						"state": {"state": old_state},
						"action": {"action": action},
						"next_state": {"state": state},
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


			

		# update
		sac.store_episode(tmp_observations)
		
		# show reward
		smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
		
	
		
		if episode>79:
			sac.update()


		smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
		if smoothed_total_reward >= solved_reward:
			reward_fulfilled += 1

			if reward_fulfilled >= solved_repeat:
				print("Environment solved!")
				if plots_enable==1:
				#print("Average angle="+str(angle_sum/max_episodes))
					data_Dataframe=pd.DataFrame(data,columns=['Episode', 'Reward', 'Angle'])
				#print(m_angle)
					data_Dataframe.to_csv("sac_pendulum.csv")
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
		data_Dataframe.to_csv("sac_pendulum.csv")
		data_Dataframe.plot(x="Episode",y="Reward",kind="line")
		plt.show()
		plt.boxplot(data_Dataframe["Reward"])
		plt.show()
	
	#return data_Dataframe
	return data
	


if __name__=='__main__':
	sac_sim(1,400,100,0)