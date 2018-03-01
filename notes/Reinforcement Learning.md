# Reinforcement Learning

**MDP $\rightarrow$ OFFLINE**

**RL $\rightarrow$ ONLINE**

- Assume a MDP
- Do not have T and R

## Model-Based Learning
### Learn approximate model (T & R) based on experience!
* $S \rightarrow A \rightarrow S'$
	* 즉 R과 L을 직접 경험을 통해 배움
* solve for values as if learned model is correct
	* 이 때문에 optimal 하지 않을 수 도 있음.

### How?
#### Step 1: Learn Empirical MDP Model

* Count outcomes $s'$ for each $s, a$
* Normalize to give estimate of $T(s, a, s')$
* Discover each $R(s, a, s')$ when we experience $(s, a, s')$

#### Step 2: Solve the learned MDP

* Use Value Iteration

## Model-Free Learning

### Passive Reinforcement Learning

#### Policy Evaluation
* Goal: Learn state values
* fixed policy $\pi(s)$
* Unknown $T(s,a,s')$
* Unknown $R(s,a,s')$

Policy가 정해져 있기 때문에 action의 choice가 없다!

Policy를 execute 하고 그 경험으로 부터 배운다!

Offline Planning (MDP)가 아니다! Since we takes actions in the world.

#### Direct Evaluation
* Goal: Compute values for each state under $\pi$
* Idea: Average together observed sample values
	1. Act according to $\pi$
	2. Every time you visit a state, write down what the sum of discounted rewards turned out to be
	3. Average those samples

* Advantages 
	* no need of knowledge about $T, R$
	* Computes the correct average values
* Disadvantages
	* waste information about state connections
	* each state must be learned seperately (= takes long time)

#### Sample-Based Policy Evaluation (TD Learning)

* Idea: 
	* Update $V(s)$ each time we experience a $s \rightarrow a \rightarrow s'$
	* 자주 등장하는 $s'$은 업데이트에 더 큰 contribution을 하게 됨.

* How?
	* Sample of $V(s)$: 

	$sample = R(s, \pi(s), s') + \gamma V^\pi(s)$

	* Update of $V(s)$:

	$V^\pi(s) \leftarrow (1-\alpha)V^\pi(s) + \alpha * sample$

	$V^\pi(s) \leftarrow V^\pi(s) + \alpha * (sample-V^\pi(s))$

* Problem
	* if we want to turn values into new policy, we are sunk

**여기까지 Recap!**

1. Direct Evaluation 의 경우 individual states 마다 계산을 해야 하고 서로 상호적이지 않기 때문에 state 마다 열심히 계산한 결과들이 따로 놀게 됨.

2. 즉 이런 상호 작용의 문제를 해결하기 위해 나온것이 TD Learning (Temporal Difference Learning) 임. TDL의 경우 기본적으로 Bellman's Equation을 이용해서 value iteration을 하는 방법. 문제가 있다면 policy를 바꿀 수가 없음. (애초부터 fixed policy를 전제로 시작했기에...)

3. 이러한 문제를 보완 하기 위해 우리는 Q-Learning 을 적용. Q-Learning의 경우 메인이 value iteration이 아닌 policy iteration임.

(어찌 됬건 $T, R$ 값이 없기 때문에 depth-limited value iteration을 통해 $V$값을 얻어내고 그 이후 Q값을 계산.)

### Active Reinforcment Learning 
#### Q-Learning (Off-Policy Learning)
* Goal: Learn Optimal Policy / Values
* Still don't know $T, R$
* suboptimal action still leads optimal policy
* Learn $Q(s, a)$ values as we go

	* $Q_{k+1} \leftarrow \sum_{s'}T(s, a, s')[R(s, a, s') +\gamma * max_{a^'}Q_k(s', a')]$

* 그러므로 Learner makes choices

* Update as we go

	* $sample = R(s, a, s') +\gamma * max_{a^'}Q_k(s', a')$

	* $Q_{k+1} \leftarrow (1-\alpha) * Q(s, a)+\alpha * sample$

* 여기서 $\alpha$ 값에 따라 과거값에 대한 weight가 달라진다.. 아마도?

* Exploration vs. Exploitation
	* $\epsilon-greedy$
		* Small $\epsilon$ = act randomly
		* Large $1-\epsilon$ = act on current policy
		* decrease as time goes on
	* Exploration Functions
		* $f(u, n) = u + \frac{k}{n}$
		* Regular Update: $R(s, a, s^')+\gamma * max_{a'}Q(s', a')$
		* Modified Update: $R(s, a, s^')+\gamma * max_{a'}f(Q(s', a'), N(s', a'))$

* Regret 

measure of total mistake cost