import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import pandas as pd
import pendulum_td3 #change according to filename
import pendulum_sac #change according to filename
import pendulum_ppo#change according to filename

import iiwa_sac
import iiwa_ppo
import iiwa_td3

import time 
import csv



max_episodes=200
max_steps=100
repeat=5
'''
def pendulum_boxplots():
	p_td3=[]
	p_ppo=[]
	p_sac=[]

	for i in range(repeat):

		#TD3 Algorithm

		initial_t=time.time()


		td3_p=pendulum_td3.td3_sim(0,max_episodes,max_steps,0)


		final_t=time.time()

		p_td3.append(final_t-initial_t)

		########################################

		#SAC Algorithm

		initial_t=time.time()


		sac_p=pendulum_sac.sac_sim(0,max_episodes,max_steps,0)


		final_t=time.time()

		p_sac.append(final_t-initial_t)

		########################################

		#PPO Algorithm

		initial_t=time.time()


		ppo_p=pendulum_ppo.ppo_sim(0,max_episodes,max_steps,0)


		final_t=time.time()

		p_ppo.append(final_t-initial_t)


	pendulum_time = [p_td3 ,p_ppo,p_sac]





	fig=plt.figure()
	fig=suptitle("Pendulum",fontsize=15,fontweight='bold')

	plt.boxplot(pendulum_time)
	plt.ylabel('Time')
	plt.xticks([1,2,3],['TD3','PPO,','SAC'])

	plt.savefig('PendulumBoxplotsTime.png')

	plt.show()


	with open("pendulum_time_data.csv","w") as file:
		w=csv.writer(file)
		w.writerows(pendulum_time)
'''
def iiwa_boxplots():
	l_iiwa_td3=[]
	l_iiwa_ppo=[]
	l_iiwa_sac=[]

	for i in range(repeat):

		#TD3 Algorithm

		initial_t=time.time()


		td3_p=iiwa_td3.td3_sim(0,max_episodes,max_steps,0)


		final_t=time.time()

		l_iiwa_td3.append(final_t-initial_t)

		########################################

		#SAC Algorithm

		initial_t=time.time()


		sac_p=iiwa_sac.sac_sim(0,max_episodes,max_steps,0)


		final_t=time.time()

		l_iiwa_sac.append(final_t-initial_t)

		########################################

		#PPO Algorithm

		initial_t=time.time()


		ppo_p=iiwa_ppo.ppo_sim(0,max_episodes,max_steps,0)


		final_t=time.time()

		l_iiwa_ppo.append(final_t-initial_t)


	iiwa_time = [l_iiwa_td3 ,l_iiwa_ppo,l_iiwa_sac]





	fig=plt.figure()
	fig=suptitle("Iiwa",fontsize=15,fontweight='bold')

	plt.boxplot(iiwa_time)
	plt.ylabel('Time')
	plt.xticks([1,2,3],['TD3','PPO,','SAC'])

	plt.savefig('IiwaBoxplotsTime.png')

	plt.show()


	with open("iiwa_time_data.csv","w") as file:
		w=csv.writer(file)
		w.writerows(iiwa_time)





#pendulum_boxplots()
iiwa_boxplots()