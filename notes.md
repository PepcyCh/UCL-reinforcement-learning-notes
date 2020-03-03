# Reinforcement Learning - UCL

## Lect 1 Introduction to RL

observation, reward, action

agent, environment

history:
$$
H_t = O_1, R_1, A_1, \dots, A_{t - 1}, O_t, R_t
$$
state:
$$
S_t = f(H_t)
$$
environment state, agent state

Markov state

* $P[S_{t + 1} \mid S_t] = P[S_{t + 1} \mid S_t, \dots, S_1]$
* The future is independent of the past given the present
* Once the state is known, the history may be thrown away

Markov decision process

* fully observability
* $O_t = S_t^e = S_t^a$

partially observable Markov decision process

* partial observability
* $S_t^a \neq S_t^e$
* agent must construct its own state representation $S_t^a$, e.g.
  * complete history: $S_t^a = H_t$
  * beliefs of environment state = $S_t^a = (P[S_t^e = s^1], \dots, P[S_t^e = s^n])$
  * recurrent neural network: $S_t^a = \sigma(S_{t - 1}^a W_s + O_t W_o)$

RL agent: policy, value function, model

policy

* agent’s behavior, a map from state to action
* $a = \pi(s)$
* stochastic policy: $\pi(a \mid s) = P[A_t = a \mid S_t = s]$

value function

* prediction of future reward
* $v_{\pi} (s) = E_{\pi} [R_{t + 1} + \gamma R_{t + 2} + \gamma^2 R_{t + 3} + \cdots \mid S_t = s]$

model

* internal model of environment, predict what the environment will do next
* $\mathcal{P}$ predicts the next state, $\mathcal{R}$ predicts the next immediate reward
* $\mathcal{P_{ss’}^a} = P[S_{t + 1} = s’ \mid S_t = s, A_t = a]$
* $\mathcal{R_{s}^a} = E[R_{t + 1} \mid S_t = s, A_t = a]$

categorizing of RL agents

* value based
  * value function
* policy based
  * policy
* actor critic
  * policy
  * value function
* model free
  * policy and/or value function
* model bases
  * poliy and/or value function
  * model

exploration, exploitation

predict, control

## Lect 2 Markov Decision Process

state transition matrix

* $\mathcal{P_{ss’} = P[S_{t + 1} = s’ \mid S_t = s]}$

Markov process (or Markov chain) $\langle \mathcal{S}, \mathcal{P} \rangle$

* $\mathcal{S}$ is a finite set of states
* $\mathcal{P}$ is the state transition matrix

Markov reward process $\langle \mathcal{S}, \mathcal{P}, \mathcal{R}, \gamma \rangle$

* a Markov chain with values
* $\mathcal{R}$ is a reward function, $\mathcal{R_s} = E[R_{t + 1} \mid S_t = s]$
* $\gamma$ is a discount factor, $\gamma \in [0, 1]$

return $G_t$

* total discounted reward from time-step $t$
* $G_t = R_{t + 1} + \gamma R_{t + 2} + \gamma^2 R_{t + 3} + \cdots$
* $\gamma$ close to $0$ - myopic
* $\gamma$ close to $1$ - far-sighted

why discount

* Mathematically convenient to discount rewards
* Avoids infinite returns in cyclic Markov processes
* Uncertainty about the future may not be fully represented
* If the reward is financial, immediate rewards may earn more interest than delayed rewards
* Animal/human behaviour shows preference for immediate reward
* It is sometimes possible to use undiscounted Markov reward processes (i.e. $\gamma = 1$), e.g. if all sequences terminate.

value function $v(s)$

* $v(s) = E[G_t \mid S_t = s]$

Bellman equation for MRPs

* value function can be decomposed into 2 parts: immediate reward $R_{t + 1}$ and $\gamma v(S_{t + 1})$
* $v(s) = E[R_{t + 1} + \gamma v(S_{t + 1}) \mid S_t = s] = \mathcal{R_s} + \gamma \sum_{s’ \in S} \mathcal{P_{ss’}} v(s')$
* matrix form: $v = \mathcal{R} + \gamma \mathcal{P} v$
* solution: $v = (I - \gamma \mathcal{P})^{-1} \mathcal{R}$
* direct solution only possible for small MRPs
* iterative methods
  * Dynamic programming
  * Monte-Carlo evaluation
  * Temporal-Differance learning

Markov decision process $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$

* MRP with decisions
* $\mathcal{A}$ is a finite set of actions

policy $\pi(a \mid s)$

* distribution over actions given states
* fully defines the behavior of an agent
* $\pi(a \mid s) = P[A_t = a \mid S_t = s]$
* stationary / time-independent
* $\mathcal{P_{ss’}^{\pi}} = \sum_{a \in \mathcal{A}} \pi(a \mid s) \mathcal{P_{ss’}^{a}}$
* $\mathcal{R_s^{\pi}} = \sum_{a \in A} \pi(a \mid s) \mathcal{R_s^a}$

state-value function $v_{\pi} (s)$

* $v_{\pi} (s) = E_{\pi} [G_t \mid S_t = s] = E_{\pi} [R_{t + 1} + \gamma v_{\pi} (S_{t + 1}) \mid S_t = s]$

action-value function $q_{\pi} (s, a)$

* $q_{\pi} (s, a) = E_{\pi} [G_t \mid S_t = s, A_t = a] = E_{\pi} [R_{t + 1} + \gamma q_{\pi} (S_{t + 1}, A_{t + 1}) \mid S_t = s, A_t = a]$

Bellman expectation equation

* $v_{\pi} (s) = \sum_{a \in \mathcal{A}} \pi (a \mid s) q_{\pi} (s, a)$
* $q_{\pi} (s, a) = \mathcal{R_s^a} + \gamma \sum_{s’ \in \mathcal{S}} \mathcal{P_{ss’}^{a}} v_{\pi} (s')$
* $v_{\pi} (s) = \sum_{a \in \mathcal{A}} \pi (a \mid s) (\mathcal{R_s^a} + \gamma \sum_{s’ \in \mathcal{S}} \mathcal{P_{ss’}^a} v_{\pi} (s'))$
* $q_{\pi} (s, a) = \mathcal{R_s^a} + \gamma \sum_{s’ \in S} \mathcal{P_{ss’}^a} \sum_{a \in \mathcal{A}} \pi (a’ \mid s’) q_{\pi} (s’, a')$
* matrix form: $v_{\pi} = \mathcal{R^{\pi}} + \gamma \mathcal{P^{\pi}} v_{\pi}$
* solution: $v_{\pi} = (I - \gamma \mathcal{P^{\pi}})^{-1} \mathcal{R^{\pi}}$

optimal value function

* best possible performance
* $v_{\ast}(s) = \max_{\pi} v_{\pi} (s)$
* $q_{\ast} (s, a) = \max_{\pi} q_{\pi}(s, a)$

optimal policy $\pi_{\ast}$

* all optimal policies achieve the optimal state/action-value function
* $\pi_{\ast} (a \mid s) = \begin{cases} 1 & \text{if } a = \mathrm{argmax}_{a \in \mathcal{A}} q_{\ast} (s, a) \\ 0 & \text{otherwise} \end{cases}$

Bellman optimality equation

* $v_{\ast} (s) = \max_{a} q_{\ast} (s, a)$
* $q_{\ast} (s, a) = \mathcal{R_s^a} + \gamma \sum_{s’ \in \mathcal{S}} \mathcal{P_{ss’}^{a}} v_{\ast} (s')$
* $v_{\ast} (s) = \max_a \mathcal{R_s^a} + \gamma \sum_{s’ \in \mathcal{S}} \mathcal{P_{ss’}^a} v_{\ast} (s')$
* $q_{\ast} (s, a) = \mathcal{R_s^a} + \gamma \sum_{s’ \in S} \mathcal{P_{ss’}^a} \max_{a’} q_{\ast} (s’, a')$
* non-linear

solving the Bellman optimality equation

* value iteration
* policy iteration
* Q-learning
* Sarsa

extensions to MDPs

* infinite and continuous MDPs
* partially observable MDPs (POMDPs)
* undiscounted, average reward MDPs

infinite MDPs

* countably infinte state and/or action spaces
  * straightforward
* continuous state and/or action spaces
  * closed form for linear quadratic model (LQR)
* continuous time
  * partially differential equation
  * Hamilton-Jacobi-Bellman (HJB) equation
  * limiting case of Bellman equation as time $\rightarrow 0$

POMDPs $\langle \mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{P}, \mathcal{R}, \mathcal{Z}, \gamma \rangle$

* $\mathcal{O}$ is a finite set of observations
* $\mathcal{Z}$ is an observation function, $\mathcal{Z_{s’o}^{a}} = P[O_{t + 1} = o \mid S_{t + 1} = s’, A_t = a]$

belief states $b(h)$

* history: $H_t = A_0, O_1, R_1, \dots, A_{t - 1}, O_t, R_t$
* probability distribution over states, conditioned on the history $h$
* $b(h) = (P[S_t = s^1 \mid H_t = h], \dots, P[S_t = s^n \mid H_t = h])$
* reductions: (infinite) history tree, (infinite) belief state tree

ergodic Markov process

* *recurrent*: each state is visited an infinite number of times
* *aperiodic*: each state is visited without any systematic period
* has limiting stationary distribution $d^{\pi}(s)$ with $d^{\pi}(s) = \sum_{s \in \mathcal{S}} \mathcal{P_{s's}} d^{\pi} (s')$

ergodic MDP

* an MDP is ergodic if the Markov chain induced by and policy is ergodic
* for any policy $\pi$, an ergodic MDP has an *average reward per time-step* $\rho^{\pi}$ that is independent of start state
* $\rho^{\pi} = \lim_{T \rightarrow \infty} \frac{1}{T} E[ \sum_{t = 1}^{T} R_{t} ]$ 

average reward value function

* value function of an undiscounted, ergodic MDP can be expressed in terms of average reward.
* extra reward due to start state $s$: $\tilde{v}_{\pi} (s) = E_{\pi}[\sum_{k = 1}^{\infty} (R_{t + k} - \rho^{\pi}) \mid S_{t} = s]$
* Bellman equation: $\tilde{v}_{\pi} (s)= E_{\pi}[(R_{t + 1} - \rho^{\pi}) + \tilde{v}_{\pi} (S_{t + 1}) \mid S_t = s]$

## Lect 3 Planning by Dynamic Programming

iterative policy evaluation

* problem: evaluate a given policy $\pi$
* solution: iterative application of Bellman expactation backup
* synchronous backups
* matrix form: $v^{k + 1} = \mathcal{R^\pi} + \gamma \mathcal{P^\pi} v^k$

improve policy

* evaluate $v_\pi (s)$
* acting greedily with respect to $v_\pi(s)$: $\pi’ = \text{greedy}(v_\pi)$

policy iteration

* policy evaluation: $\pi \to v$, iterative policy evaluation
* policy improvement: $v \to \pi$, greedy policy improvement
* always converges to $\pi_{\ast}, v_{\ast}$

modified policy iteration

* policy evaluation - $\varepsilon$-convergence / stop after $k$ iterations
  * $k = 1$: value iteration

generalized policy iteration

* policy evaluation: $\pi \to v$, any policy evaluation algorithm
* policy improvement: $v \to \pi$, any policy improvement algorithm

principle of optimality

* A policy $\pi(a \mid  s)$ achieves the optimal value from state $s$, $v_\pi(s) = v_\ast(s)$, iff for any state $s’$ reachable from $s$, $\pi$  achieves the optimal value from state $s’$, $v_\pi(s’) = v_\ast(s')$

deterministic value iteration

* if we know the solution to subproblem $v_\ast(s’)$, then the solution $v_\ast(s)$ can be found by one-step lookahead $v_\ast(s) \leftarrow \max_{a \in \mathcal{A}} \mathcal{R_{s}^{a}} + \gamma \sum_{s’ \in \mathcal{S}} \mathcal{P_{ss’}^a} v_\ast(s')$
* the idea of value iteration is to apply these updates iteratively
* intuition: start with final rewards and work backwords

value iteration

* problem: find optimal policy $\pi$
* solution: iterative application of Bellman expactation backup
* synchronous backups
* unlike policy iteration, there is no explicit policy; intermediate value functions may not correspond to any policy
* matrix form: $v^{k + 1} = \max_{a \in \mathcal{A}} ( \mathcal{R^a} + \gamma \mathcal{P^a} v^k)$

synchronous DP algorithms

| Problem    | Bellman equation                                         | Algorithm                   |
| ---------- | -------------------------------------------------------- | --------------------------- |
| prediction | Bellman expectation equation                             | iterative policy evaluation |
| control    | Bellman expectation equation + gready policy improvement | policy iteration            |
| control    | Bellman optimality equation                              | value iteration             |

* algorithms are based on state-value function $v_\pi(s)$ or $v_\ast(s)$
* complexity $O(mn^2)$ per iteration, for $m$ actions and $n$ states
* could also apply to action-value function $q_\pi(s, a)$ or $q_\ast(s, a)$
* complexity $O(m^2n^2)$ per iteration

asynchronous DP

* asynchronous DP backs up states individually, in any order
* for each selected state, apply the appropriate backup
* can significantly reduce computation
* guatanteed to converge if all states to be selected
* three simple ideas: in-place, prioritized sweeping, real-time

in-place DP

prioritized sweeping

* use magnitude of Bellman error to guide state selection, e.g. $\vert \max_{a \in \mathcal{A}}( \mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P_{ss’}^a} v(s’)) - v(s) \vert$
* backup the state with the largest remaining Bellman error
* can be implemented efficiently by a priority queue

real-time DP

* use agent’s experience to guide state selection
* after each time-step $S_t, A_t, R_{t + 1}$, backup $S_{t + 1}$

full-width backup

* DP uses full-width backup
* for each backup, every successor state and action is considered
* DP is effective for medium-sized problems (millions of states)

sample backup

* using sample reward and transitions $\lang S, A, R, S' \rang$ instead of reward function $\mathcal(R)$ and transition dynamics $\mathcal{P}$
* model-free: no advance knowledge of MDP required
* breaks the curse of dimensionality
* cost of backup is constant, independant of $n$

approximate DP

* use a function approximator $\hat{v}(s, w)$, apply DP to $\hat{v}(\cdot, w)$
* $w_{k + 1} = \hat{v}_k(s)$, where $\hat{v}_k(s)$ is backuped from $\hat{v}(s', w_k)$ and $s \in \tilde{\mathcal{S}} \subseteq \mathcal{S}$

contraction mapping theorem

* skipped...

