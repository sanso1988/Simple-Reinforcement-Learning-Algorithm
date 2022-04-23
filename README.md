# Reinforcement-Learning-Algorithm-Script
This repository is only for studying Reinforcement Learning.
This repository are focusing on implementing classical Reinforcement Learning algorithms. The code is created as simple as possible.
If you find anything wrong, don't hesitate to comment. I am very happy to correct it.

## Requirements
- **gym**
- **pytorch**
- **atari-py**
- **ale-py**
- **numpy**
- **matplotlib**


## Usage
python3 main.py --train 1

## Already Implementation
### Deep Q-Learning
---
1. Algorithm refers to Playing Atari with Deep Reinforcement Learning [1].

2. Arguments:
- Enviroment: ALE/Pong-v5. You can try others.
- Experience Memory: 20000 steps
- Epsilon: decay from 1 to 0.1 in 50k to 1m steps with linear style.
- Stack Frames: 4
- Agent Update Period: 4 steps
- Target Agent(Netwrok) Update Period: 10k steps.
- Optimize: RMSProp(learning rate=0.00025, epsilon=0.01, gradient squared moving average=0.95, momentum=0.95)
- Network Architecture: 3 covolution layers, 2 linear layers.


3. Training Average Reward(100) vs Training Episode.
<img width="597" alt="image" src="https://user-images.githubusercontent.com/39761761/164689558-4453bb3c-15d6-4b89-9e43-5feecb69d2c8.png">

4. Skilled robot plays look like this. It seems the robot exploit some strang games bug. - -| Maybe you trained much more eposides it will be perfect.

https://user-images.githubusercontent.com/39761761/164703456-793f3ade-ff07-4a44-90de-943d21aacdfe.mov

PS: What's interesting is the estimated action value is larger than 1, it seems the robot overestimate a lot. Maybe robot think he will get a score on most states, he got one unit reward and next states(one more expected unit reward). it doesn't make sense for pong game. if we set the eposide end at the moment reward got, estimate action value will be in 0 to 1, But it's not the general for other games. so we don't do it. 

---


## References & Resource
[1] Playing Atari with Deep Reinforcement Learning, https://arxiv.org/pdf/1312.5602.pdf.

- **OpenAI Spinning Up**

https://spinningup.openai.com/en/latest/

- **Reinforcement Learning: A Introduction**

http://incompleteideas.net/book/RLbook2018.pdf
