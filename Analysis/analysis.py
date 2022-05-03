import pandas as pd
import matplotlib.pyplot as plt




data = pd.read_csv('../Log/train_eposide_reward.txt')

data['flag'] = data.index % 100
data['reward_m'] = data['reward'].rolling(100).mean()
data = data[data['flag'] == 0]

plt.plot(range(len(data['reward_m'])), data['reward_m'])
plt.show()

print(data.tail())