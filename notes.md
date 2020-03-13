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

## Lect 4 Model-Free Prediction

Monte-Carlo Reinforcement Learning

* MC methods learn directly from episodes of experience
* MC is model-free: no knowledge of MDP transitions / rewards
* MC learns from complete episodes: no bootstrapping
* MC uses the simplest possible idea: value = mean return
* Caveat: can only apply MC to episodic MDPs
  * All episodes must terminate

Monte-Carlo  Policy Evaluation (every/firts-visit)

* Every/the first time-step $t$ that state $s$ is visited in an episode
* Increment counter $N(s) \leftarrow N(s) + 1$
* Increment total return $S(s) \leftarrow S(s) + G_t$
* Value is estimated by mean return $V(s) = S(s) / N(s)$
* By law of large number, $V(s) \to v_\pi(s)$ as $N(s) \to \infty$

Incremental Mean

* $\mu_k = \mu_{k - 1} + (x_k - \mu_{k - 1}) / k$

Incremental MC Updates

* $N(s) \leftarrow N(s) + 1, V(s) \leftarrow V(s) + (G_t - V(s)) / N(s)$
* In non-stationary problems, it can be useful to track a running mean, i.e. forget old episodes: $V(s) \leftarrow V(s) + \alpha(G_t - V(s))$

Temporal-Difference Learning

* TD methods learn directly from episodes of experience
* TD is model-free: no knowledge of MDP transitions / rewards
* TD learns from incomplete episodes, by bootstrapping 
* TD updates a guess towards a guess

Simplest Temporal-Difference Learning Algorithm: TD(0)

* Use $R_{t + 1} + \gamma V(S_{t + 1})$ instead of $G_t$ in MC methods
* $R_{t + 1} + \gamma V(S_{t + 1})$ is called the TD target
* $\delta_{t + 1} = R_{t + 1} + \gamma V(S_{t + 1}) - V(S_t)$ is called the TD error

Bias / Variance

* Return $G_t = R_{t + 1} + \gamma R_{t + 2} + \cdots + \gamma^{T - t - 1} R_{T}$ is unbiased
* True TD target $R_{t + 1} + \gamma v_\pi(S_{t + 1})$ is unbiased
* TD target $R_{t + 1} + \gamma V(S_{t + 1})$ is biased
* TD target is much lower variance than the return

Batch MC and TD

* sample $K$ episodes without updates

Certainty Equivalence

* MC converges to solution with minimum mean-squared error
  * best fit to the observed returns
  * $\sum_{k = 1}^{K} \sum_{t = 1}^{T_k} (G_t^k - V(S_t^k))^2$
* TD(0) converges to solution of max likelihood Markov model
  * solution to the MDP $\lang \mathcal{S}, \mathcal{A}, \hat{\mathcal{P}}, \hat{\mathcal{R}}, \gamma \rang$ that best fits the data
  * $\hat{\mathcal{P}}_{ss’}^a = 1/N(s, a) \sum_{k = 1}^{K} \sum_{t = 1}^{T_k} [s_t^k, a_t^k, s_{t + 1}^k = s, a, s']$
  * $\hat{\mathcal{R}}_s^a = 1 / N(s, a) \sum_{k = 1}^{K} \sum_{t = 1}^{T_k} [s_t^k, a_t^k = s, a]r_t^k$

Advantages & Disadvantages of MC vs. TD

* TD can learn before knowing the final outcome
  * TD can learn online after every step
  * MC must wait until end of episode before return is known
* TD can learn without the final outcome
  * TD can learn from incomplete sequences
  * MC can only learn from complete sequences
  * TD works in continuing (non-terminating) environments
  * MC only works for episodic (terminating) environments
* MC has high variance, zero bias
  * Good convergence properties
  * (even with function approximation)
  * Not very sensitive to initial value
  * Very simple to understand and use
* TD has low variance, some bias
  * Usually more efficient than MC
  * TD(0) converges to $v_\pi(s)$
  * (but not always with function approximation)
  * More sensitive to initial value
* TD exploits Markov property
  * Usually more efficient in Markov environments
* MC does not exploit Markov property
  * Usually more effective in non-Markov environments

Bootstrapping and Sampling

* Bootstrapping: update involves an estimate
  * DP, TD bootstrap
  * MC does not bootstrap
* Sampling: update samples an expectation
  * MC, TD samples
  * DP does not sample

n-step Prediction

* $n = 1$: TD(0)
* $n = \infty$: MC

n-step Return

* $G_t^{(n)} = R_{t + 1} + \gamma R_{t + 2} + \cdots + \gamma^{n - 1}R_{t + n} + \gamma^n V(S_{t + n})$
* n-step TD: $V(S_t) \leftarrow V(S_t) + \alpha(G_t^{(n)} - V(S_t))$

Averaging n-step Returns

Forward-view TD($\lambda$)

* $\lambda$-return: $G_t^\lambda = (1 - \lambda)\sum_{n = 1}^{\infty}\lambda^{n - 1}G_t^{(n)}$
* Forward-view TD($\lambda$): $V(S_t) \leftarrow V(S_t) + \alpha(G_t^\lambda - V(S_t))$
* Can only be computed from complete episodes

Eligibility Traces (资格迹)

* $E_0(s) = 0$
* $E_t(s) = \gamma \lambda E_{t - 1}(s) + [S_t = s]$

Backward-view TD($\lambda$)

* Keep an eligibility trace for each state
* Update value $V(s)$ for every state $s$
* $\delta_t = R_{t + 1} + \gamma V(S_{t + 1}) - V(S_t)$
* $V(s) \leftarrow V(s) + \alpha \delta_t E_t(s)$

TD($\lambda$) and TD(0)

* $E_t(s) = [S_t = s]$
* $V(s) \leftarrow V(s) + \alpha \delta_t [S_t = s]$, which is just TD(0)

TD($\lambda$) and MC

* The sum of offline updates is identical for forward-view and backward-view TD($\lambda$)
* $\sum_{t = 1}^{T} \alpha \delta_t E_t(s) = \sum_{t = 1}^{T} \alpha (G_t^\lambda - V(S_t))[S_t = s]$
* Over the course of an episode, total update for TD(1) is the same as total update for MC

MC and TD(1)

* TD(1) is roughly equivalent to every-visit MC
* Error is accumelated online, step-by-step
* If value function is only updated offline at end of episode
* Then total update is exactly the same as MC

Telescoping in TD($\lambda$)

* $G_t^\lambda = V(S_t) + \sum_{i = 0}^{\infty} (\lambda \gamma)^i \delta_{t + i}$

Offline Equivalence of Forward and Backward TD

* Offline updates are accumelated within episode
* but applied in batch at the end of episodes

Online Equivalence of Forward and Backward TD

* applied online at each step within episode
* Forward and backward-view TD($\lambda$) are slightly different
* Exact online TD($\lambda$) achieves perfect equivalence
* By using a slightly different form of eligiblity trace

## Lect 5 Model-Free Control

Model-Free Control can solves:

* MDP model is unknown, but exprience can be sampled
* MDP model is known, but is too big to use, except by samples

Generalized Policy Iteration with MC Evaluation

* $V = v_\pi$ ?
* Greedy policy improvement ?

Generalized Policy Iteration with Action-Value Function

* Greedy policy improvement over $Q(s, a)$ is model-free
* $Q = q_\pi$ !
* Greedy policy improvement ? (we are sampling, max may not be the best)

$\varepsilon$-Greedy Exploration

* Simplest idea for ensuring continual exploration
* All $m$ actions are tried with non-zero probability
* With probability $1 - \varepsilon$ choose the greedy action
* With probability $\varepsilon$ choose an action at random (include the greedy one)

MC Policy Iteration

* MC policy evaluation, $Q = q_\pi$
* $\varepsilon$-greedy policy improvement

MC Control

* For every episode
* MC policy evaluation $Q \approx q_\pi$
* $\varepsilon$-greedy policy improvement

GLIE

* Greedy in the Limit with Infinite Exploration
* All action-state pairs are explored infinitely many times
* The policy converges on a greedy policy
* $\varepsilon$-greedy is GLIE if $\varepsilon$ reduces to zero at $\varepsilon_k = 1 / k$

GLIE MC Control

* For each state $S_t$ and action $A_t$ in the episode
* $N(S_t, A_t) \leftarrow N(S_t, A_t) + 1$
* $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + (G_t - Q(S_t, A_t)) / N(S_t, A_t)$
* Do $\varepsilon$-greedy with $\varepsilon = 1/k$
* GLIE MC control converges to the optimal action-value function

On-Policy Control with Sarsa

* Apply TD on Q: $Q(s, a) \leftarrow Q(s, a) + \alpha(R + \gamma Q(s’, a’) - Q(s, a))$
* Every time-step
* Sarsa, $Q \approx q_\pi$
* $\varepsilon$-greedy policy improvement

Convergence of Sarsa

* GLIE sequence of policies $\pi_t(a \mid s)$
* Robbins-Monro sequence of step-sizes $\alpha_t$:
* $\sum_{t = 1}^{\infty} \alpha_t = \infty, \sum_{t = 1}^{\infty} \alpha_t^2 < \infty$

n-step Sarsa

* $q_t^{(n)} = R_{t + 1} + \gamma R_{t + 2} + \cdots + \gamma^{n - 1} R_{t + n} + \gamma^n Q(S_{t + n}, A_{t + n})$
* $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (q_t^{(n)} - Q(S_t, A_t))$

Forward-view Sarsa($\lambda$)

* $q_t^\lambda = (1 - \lambda) \sum_{n = 1}^{\infty} \lambda^{n - 1} q_t^{(n)}$
* $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (q_t^\lambda - Q(S_t, A_t))$

Backward-view Sarsa($\lambda$)

* Eligibility trace: $E_0(s, a) = 0; E_t(s, a) = \gamma \lambda E_{t - 1}(s, a) + [S_t = a, A_t = a]$
* $\delta_t = R_{t + 1} + \gamma Q(S_{t + 1}, A_{t + 1}) - Q(S_t, A_t)$
* $Q(s, a) \leftarrow Q(s, a) + \alpha \delta_t E_t(s, a)$

Off-policy Learning

* behavior policy $\mu(a \mid s)$
* Learn from observing humans or other agents
* Re-use expeirence generated from old policies
* Learn about optimal policy while following exploratory policy
* Learn about multiple policies while following one policy

Importance Sampling for Off-policy MC

* Use returns generated from $\mu$ to evaluate $\pi$
* $G_t^{\pi / \mu} = \frac{\pi(A_t \mid  S_t) \cdots \pi(A_T \mid S_T)}{\mu(A_t \mid S_t) \cdots \mu(A_T \mid S_T)} G_t$
* $V(S_t) \leftarrow V(S_t) + \alpha (G_t^{\pi / \mu} - V(S_t))$
* Can dramatically increase variance

Importance Sampling for Off-policy TD

* $V(S_t) \leftarrow V(S_t) + \alpha(\frac{\pi(A_t \mid S_t)}{\mu(A_t \mid S_t)} (R_{t + 1} + \gamma V(S_{t + 1})) - V(S_t))$
* Much lower variance than MC importance sampling
* Policies only need to be similar over a single step

Q-Learning

* Off-policy learning of action-values
* No importance sampling is required
* $A_{t + 1} \sim \mu(\cdot \mid S_t)$, $A’ \sim \pi(\cdot \mid S_t)$
* $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha(R_{t + 1} + \gamma Q(S_{t + 1}, A’) - Q(S_t, A_t))$

Off-policy Control with Q-Learning

* Allow both behavior and target policies to improve
* Target policy $\pi$ is greedy; Behavior policy $\mu$ is $\varepsilon$-greedy
* Q-learning target can be simplified to $R_{t + 1} + \max_{a’} \gamma Q(S_{t + 1}, a')$
* Q-learning control converges to the optimal action-value function

Relationship between DP and TD

|                  | Full Backup (DP)                                             | Sample Backup (TD)                                           |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| BEE for $v_\pi$  | Iterative Policy Evaluation $V(s) \leftarrow E[R + \gamma V(S') \mid s]$ | TD Learning $V(S) \overset{\alpha}{\leftarrow} R + \gamma V(S')$ |
| BEE for $q_\pi$  | Q-Policy Iteration $Q(s, a) \leftarrow E[R + \gamma Q(S', A') \mid s, a]$ | Sarsa $Q(S, A) \overset{\alpha}{\leftarrow} R + \gamma Q(S’, A')$ |
| BOE for $q_\ast$ | Q-Valur Iteration $Q(s, a) \leftarrow E[R + \gamma \max_{a’} Q(S’, a’) \mid s, a]$ | Q-Learning $Q(S, A) \overset{\alpha}{\leftarrow} R + \gamma \max_{a'} Q(S’, a')$ |

* where $x \overset{\alpha}{\leftarrow} y$ means $x \overset{\alpha}{\leftarrow} x + \alpha(y - x)$

## Lect 6 Value Function Approximation

Problems with large MDPs

* There are too many states and/or actions to store in memory
* It is too slow to learn the value of each state individually

Solution for large MDPs

* Estimate value function with function approximation
* $\hat{v}(s, w) \approx v_\pi(s)$ or $\hat{q}(s, a, w) \approx q_\pi(s, a)$ ($w$ is a vector)
* Generalize from seen states to unseen states
* Update parameter $w$ using MC or TD learning

Which Function Approximator

* Differentiable: linear combinations of features, neural network
* Require a training method that is suitable for non-stationary, non-iid data

Value Function Approx. By Stochastic Gradiant Descent

* Goal: minimize $J(w) = E_\pi [(v_\pi(S) - \hat{v}(S, w))^2]$
* Gradient descent: $\Delta w = \alpha E_\pi [(v_\pi(S) - \hat{v}(S, w)) \nabla_w \hat{v}(S, w)]$
* Stochastic gradient descent samples the gradient

Linear Value Function Approximation

* $\hat{v}(S, w) = x(S)^T w$
* Stochastic gradient descent converges on the global optimum
* $\Delta w = \alpha (v_\pi(S) - \hat{v}(S, w)) x(S)$

Incremental Prediction Algorithm

* In practive, we substitude a target for $v_\pi(S)$
* MC: return $G_t$
* TD(0): TD target $R_{t + 1} + \gamma \hat{v}(S_{t + 1}, w)$
* TD($\lambda$): $\lambda$-return $G_t^\lambda$

MC with Value Function Approx.

* Return $G_t$ is unbiased
* Can apply supervised learning to $(S_1, G_1), (S_2, G_2), \dots, (S_T, G_T)$
* Converges to a local optimum. Even when using a non-linear value function approx.

TD(0) with Value Function Approx.

* TD target is biased
* Can still apply supervised learning to $(S_1, R_2 + \gamma \hat{v}(S_2, w)), \dots, (S_{T - 1}, R_T)$
* Linear TD(0) converges to global optimum

TD($\lambda$) with Value Function Approx.

* $\lambda$-return is biased
* Can still apply supervised learning to $(S_1, G_1^\lambda), \dots, (S_{T - 1}, R_{T - 1}^\lambda)$
* Forward-view linear TD($\lambda$): $\Delta w = \alpha(G_t^\lambda - \hat{v}(S_t, w)) x(S_t)$
* Backward-view linear TD($\lambda$): $E_t = \gamma \lambda E_{t - 1} + x(S_t), \Delta w = \alpha \delta_t E_t$
* Forward and backward are equivalent

Control with Value Function Approx.

* Approximate policy evaluation
* $\varepsilon$-greedy policy improvement

Action-Value Function Approx.

Linear Action-Value Function Approx.

Incremental Control Algorithm

* Substitude a target for $q_\pi(S, A)$
* MC: return $G_t$
* TD(0): TD target $R_{t + 1} + \gamma \hat{q}(S_{t + 1}, A_{t + 1})$
* Forward-view TD($\lambda$): action-value $\lambda$-return $q_t^\lambda$
* Backward-view TD($\lambda$): $E_t = \gamma \lambda E_{t - 1} + \nabla_w \hat{q}(S_t, A_t, w), \Delta w = \alpha \delta_t E_t$

Experience Replay in DQN

* Take action $a_t$ according to $\varepsilon$-greedy policy
* Store transition $(s_t, a_t, r_{t + 1}, s_{t + 1})$ in replay memory $\mathcal{D}$
* Sample random minibatch of transitions $(s, a, r, s')$ from $\mathcal{D}$
* Compute Q-learning targets with respect to old, fixed parameters $w^-$
* Optimize MSE between Q-network and Q-learning targets $\mathcal{L_i}(w_i) = E_{s, a, r, s’ \sim \mathcal{D}}[(r + \gamma \max_{a'} Q(s', a’, w^-) - Q(s, a, w_i)^2]$
* Using variant of stochastic gradient descent

Linear Least Squares Prediction

* We can solve the least squares solution directly

* $$
  \begin{align}
  E_{\mathcal{D}}[\Delta w] &= 0 \\
  \alpha \sum_{t = 1}^{T} x(s_t) (v_t^\pi - x(s_t)^T w) &= 0 \\
  \sum_{t = 1}^{T} x(s_t) v_t^\pi &= \sum_{t = 1}^{T} x(s_t) x(s_t)^Tw \\
  w &= \left( \sum_{t = 1}^T x(s_t)x(s_t)^T \right)^{-1} \sum_{t = 1}^T x(s_t) v_t^\pi
  \end{align}
  $$

* For $n$ features, direct solution is $O(n^3)$

* Incremental solution time is $O(n^2)$ using Shermann-Morrison

* LSMC: $v_t^\pi \approx G^t$

* LSTD: $v_t^\pi \approx R_{t + 1} + \gamma \hat{v}(S_{t + 1}, w)$

* LSTD($\lambda$): $v_t^\pi \approx G_t^\lambda$

LS Policy Iteration

* Policy evaluation by LS Q-learning
* Greedy policy improvement

LS Action-Value Function Approx.

LS Control

* The experience is generated from any policies
* So to evaluate $q_\pi(S, A)$ we must learn off-policy

