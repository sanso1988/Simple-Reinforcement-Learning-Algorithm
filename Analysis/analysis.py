import pandas as pd
import matplotlib.pyplot as plt




simple_data = pd.read_csv('../Log/train_eposide_reward_simple.txt')
vanilla_data = pd.read_csv('../Log/train_eposide_reward_vanilla.txt')
ppo_data = pd.read_csv('../Log/train_eposide_reward_ppo.txt')

simple_data['reward_n'] = simple_data['reward'].str.split('.').str[0].astype('int')
vanilla_data['reward_n'] = vanilla_data['reward'].str.split('.').str[0].astype('int')

simple_data['reward_m'] = simple_data['reward_n'].rolling(100).mean()
vanilla_data['reward_m'] = vanilla_data['reward_n'].rolling(100).mean()
ppo_data['reward_m'] = ppo_data['reward'].rolling(100).mean()


data = pd.merge(simple_data, vanilla_data, on='eposide', how='left')
data['reward_m_y'].fillna(method='pad', inplace=True)
data = pd.merge(data, ppo_data, on='eposide', how='left')
data['reward_m'].fillna(method='pad', inplace=True)
print(data.tail())


x = range(len(data))
fig, ax = plt.subplots()
ax.plot(x, data['reward_m_x'], label='simple policy gradient')
ax.plot(x, data['reward_m_y'], label='vanilla policy gradient')
ax.plot(x, data['reward_m'],   label='ppo')
ax.set_ylabel('average rewards per 100 episodes ')
ax.set_xlabel('eposide')  # Add a y-label to the axes.
ax.set_title("Simple PG VS Vanilla PG vs PPO")  # Add a title to the axes.
ax.legend();  # Add a legend.
plt.show()


