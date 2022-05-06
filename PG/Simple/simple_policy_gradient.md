# Simple Policy Gradient Algorithm

Basic idea of Policy Gradient(PG) Algorithm is to improve policy directly, what's different from value-based algorithm which want to find the ture value of state and action. Actually OpenAI Spinning Up[1] give a clear demonstration about PG, you can see this document as a note of that.

## Basic Concept

**State**(s) represents the situation agent is in the environment like image dispalyed in a game. 

**Action**(a) represents what agent can take to interact with environment like pushing left/right button. 

**Reward**(r) is what agent want to get as much as possible like score in a game.

**Policy**($\pi$) is a map from state to action decide agent how to react to environment, what means $a\sim{\pi}(s)$.

**Information Generating process:** At any time(step) t, agent gets information from environment we call $s_t$ , take the action $a_t$, receive the reward $r_t$ and next state $s_{t+1}$.  Information collects from agent interacts with environment is basis for improving the policy.

**Trajectory**($\tau$) is a trace agent interacts with environment, $s_0,a_0,r_0,s_1,a_1,r_1,....$ until terminal. It's called episode which trajectory is in. $\tau$ related to policy $\pi$ and environment which decide next state and reward.

**Target:** It's obvious agent's goal is to maximize the cumulative rewards which is actually random varibale according to policy. We use the conventional denotation G[2] to represent the all rewards.
$$
G(\tau)=\sum_{t=0}^{\infty}{\gamma}^tr_t
$$
**Policy Determined by Parameters:** Set the parameter $\theta$ to determine the policy via MLP or other stochastic function (stochastic), policy ${\pi}_\theta$ gives the probability for taking certain action. Parameterized policy is convenient for solving the Reinforcement Learning(RL) problem.

**RL Problem:** Find the optimum policy $\pi^*$ base on the optimum $\theta^*$, which maximize expectation of cumulative rewards.
$$
\theta^*=argmax_\theta(E[G(\tau)])
$$


 ## Mathematical Derivation

According to optimization process, move $\theta$ with direction ${\nabla}_{\theta}E[G(\tau)]$ will lift the target value $E[G(\tau)]$.
$$
\theta \leftarrow \theta+\alpha{\nabla}_{\theta}E[G(\tau)]
$$
$\alpha$ is learning rate. Updating stops when $\theta$ near to $\theta^*$.

**How to calculate the gradient  $\nabla_{\theta}E[G(\tau)]$ is the key point.** 

**1. What $E[G({\tau})]$ looks like:**
$$
\begin{align}
E[G(\tau)]
&=\int_{\tau}G(\tau)P(\tau)d\tau\\
&=\int_{\tau}G(\tau){\prod}p(s_0)\pi_\theta(a_0|s_0)p(r_0,s_1|s_0,a_0)...\pi_\theta(a_t|s_t,a_t)p(r_t,s_{t+1}|s_t,a_t)...d\tau\\
\end{align}
$$
**2. Transform Production to Linear Summation **

It's a important trick in statistics what get the closed-form maxlikelihood estimator. Log transformation make probability production to linear summation.  The trick can be used here.
$$
\begin{align}
{\nabla}_{\theta}E[G(\tau)]
&={\nabla_\theta}\int_{\tau}G(\tau)P(\tau)d\tau\\
&=\int_{\tau}G(\tau){\nabla}_{\theta}P(\tau)d\tau\\
&=\int_{\tau}G(\tau)P(\tau){\nabla}_{\theta}logP(\tau)d\tau\\
&=\int_{\tau}G(\tau)P(\tau){\nabla}_{\theta}(logp(s_0) +{\sum}(log\pi_{\theta}(a_k|s_k)+logp(r_k,s_{k+1}|s_k,a_k)))
d\tau\\
&=\int_{\tau}G(\tau)P(\tau){\nabla}_{\theta}{\sum}log\pi_{\theta}(a_k|s_k)
d\tau\\
&=E[G(\tau){\nabla}_{\theta}{\sum}log\pi_{\theta}(a_k|s_k)]
\end{align}
$$
**$logp(r_k,s_{k+1}|s_k,a_k)$** **is a probability relate to environment not policy parameter $\theta$, its' gradient is zero, so as $logp(s_0)$**.

The derivation above shows the gradient we need is a expectation. **According to Large Number Law and Central Limit Theorem, average is a good estimator for expectation.**
$$
{\nabla}_{\theta}E[G(\tau)]=E[G(\tau){\nabla}_{\theta}{\sum}log\pi_{\theta}(a_k|s_k)]
\leftarrow
\frac{1}{N_\tau}{\sum_i}G(\tau_i){\nabla}_{\theta}{\sum_k}log\pi_{\theta}(a_i,_k|s_i,_k)
$$

## Conclusion

Policy gradient algorithms improve policy via parameter updating, target function gradient calculation is the key. According to mathmatical derivation, the gradient can be estimated by average function of agent-environment interacting information. It means agent keep interacting with environment , gather information to calculate the gradient estimator to update the parameter. This process goes on literally until we get the optimum policy (or good enough).







---

[1] OpenAI Spinning Up, https://spinningup.openai.com.

[2] Reinforcement Learing An Introduction, Second Edition, Sutton et al.
