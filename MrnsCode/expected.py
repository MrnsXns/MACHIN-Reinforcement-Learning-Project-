import pandas as pd 
import pendulum_sac
import pendulum_ppo
import pendulum_td3


import iiwa_sac
import iiwa_ppo
import iiwa_td3

import matplotlib.pyplot as plt
import csv

max_episodes=200
max_steps=100
repeat=5

def pendulum_expected():
	p_ppo_frames=[]
	for i in range(repeat):
		ppo=pendulum_ppo.ppo_sim(1,max_episodes,max_steps,0)
		df2=pd.DataFrame(ppo,columns=['Episode', 'Reward', 'Angle'])
		p_ppo_frames.append(df2)

	res=pd.concat(p_ppo_frames)
	res.to_csv("pendulum_ppo_expected_return.csv")
#res.plot(x="Episode",y="Reward",kind="line")
#plt.show()

	p_td3_frames=[]
	for i in range(repeat):
		td3=pendulum_td3.td3_sim(1,max_episodes,max_steps,0)
		df2=pd.DataFrame(td3,columns=['Episode', 'Reward', 'Angle'])
		p_td3_frames.append(df2)

	res=pd.concat(p_td3_frames)
	res.to_csv("pendulum_td3_expected_return.csv")

	p_sac_frames=[]
	for i in range(repeat):
		sac=pendulum_sac.sac_sim(1,max_episodes,max_steps,0)
		df2=pd.DataFrame(sac,columns=['Episode', 'Reward', 'Angle'])
		p_sac_frames.append(df2)

	res=pd.concat(p_sac_frames)
	res.to_csv("pendulum_sac_expected_return.csv")





def iiwa_expected():
	'''i_ppo_frames=[]
	for i in range(repeat):
		ppo=iiwa_ppo.ppo_sim(1,max_episodes,max_steps,0)
		df2=pd.DataFrame(ppo,columns=['Episode', 'Reward'])
		i_ppo_frames.append(df2)

	res=pd.concat(i_ppo_frames)
	res.to_csv("iiwa_ppo_expected_return.csv")
	'''
#res.plot(x="Episode",y="Reward",kind="line")
#plt.show()

	i_td3_frames=[]
	for i in range(repeat):
		td3=iiwa_td3.td3_sim(1,max_episodes,max_steps,0)
		df2=pd.DataFrame(td3,columns=['Episode', 'Reward'])
		i_td3_frames.append(df2)

	res=pd.concat(i_td3_frames)
	res.to_csv("iiwa_td3_expected_return.csv")

	i_sac_frames=[]
	for i in range(repeat):
		sac=iiwa_sac.sac_sim(1,max_episodes,max_steps,0)
		df2=pd.DataFrame(sac,columns=['Episode', 'Reward'])
		i_sac_frames.append(df2)

	res=pd.concat(i_sac_frames)
	res.to_csv("iiwa_sac_expected_return.csv")


pendulum_expected()
iiwa_expected()
