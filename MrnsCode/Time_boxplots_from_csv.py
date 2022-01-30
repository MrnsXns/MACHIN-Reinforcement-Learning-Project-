import csv
import matplotlib.pyplot as plt

file=open('pendulum_time_data.csv')

csvreader=csv.reader(file)

data=list(csvreader)


final_data=[list(map(float,sublist)) for sublist in data] #convert string elements of each sublist into float type





fig=plt.figure()
fig.suptitle('Pendulum',fontsize=15,fontweight='bold')

plt.boxplot(final_data)
plt.ylabel('Time')
plt.xticks([1,2,3],['TD3','PPO,','SAC'])

plt.savefig('PendulumBoxplotsTime.png')
plt.show()

