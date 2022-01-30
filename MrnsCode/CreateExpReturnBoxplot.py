import pandas as pd
import matplotlib.pyplot as plt





def iiwa_exp_return():
	df1=pd.read_csv("iiwa_td3_expected_return.csv")
	df2=pd.read_csv("iiwa_sac_expected_return.csv")
	df3=pd.read_csv("iiwa_ppo_expected_return.csv")

	
	data=[df1["Reward"],df2["Reward"],df3["Reward"]]
	labels=["TD3","SAC","PPO"]
	plt.boxplot(data,labels=labels)
	plt.ylabel("Expected return")
	plt.title("IIWA Expected Return Boxplots")
	plt.savefig('IIWABoxplotsExpectedReturns.png')
	plt.show()



def pendulum_exp_return():
	df1=pd.read_csv("pendulum_td3_expected_return.csv")
	df2=pd.read_csv("pendulum_sac_expected_return.csv")
	df3=pd.read_csv("pendulum_ppo_expected_return.csv")


	data=[df1["Reward"],df2["Reward"],df3["Reward"]]
	labels=["TD3","SAC","PPO"]
	plt.boxplot(data,labels=labels)
	plt.ylabel("Expected return")
	plt.title("Pendulum Expected Return Boxplots")
	plt.savefig('PendulumBoxplotsExpectedReturns.png')
	plt.show()

pendulum_exp_return()