# WHY DPO IS A MISSPECIFIED ESTIMATOR AND HOW TO FIX IT

Aditya Gopalan Indian Institute of Science Bangalore, India aditya@iisc.ac.in

Sayak Ray Chowdhury Indian Institute of Technology Kanpur, India sayakrc@iitk.ac.in

Debangshu Banerjee HP AI Research Bangalore, India debangshu.banerjee@hp.com

## ABSTRACT

Direct alignment algorithms such as Direct Preference Optimization (DPO) finetune models based on preference data, using only supervised learning instead of two-stage reinforcement learning with human feedback (RLHF). We show that DPO encodes a statistical estimation problem over reward functions induced by a parametric policy class. When the true reward function that generates preferences cannot be realized via the policy class, DPO becomes misspecified, resulting in failure modes such as preference order reversal, worsening of policy reward, and high sensitivity to the input preference data distribution. On the other hand, we study the local behavior of two-stage RLHF for a parametric class and relate it to a natural gradient step in policy space. Our fine-grained geometric characterization allows us to propose AuxDPO, which introduces additional auxiliary variables in the DPO loss function to help move towards the RLHF solution in a principled manner and mitigate the misspecification in DPO. We empirically demonstrate the superior performance of AuxDPO on didactic bandit settings as well as LLM alignment tasks.

## 1 INTRODUCTION

Preference-based alignment is a key part of the training process of large language models (LLMs). It aims to steer a pretrained model’s conditional distribution toward outputs that humans (or carefully calibrated annotator models) prefer. Formally, given comparison data $( s , a _ { w } , a _ { l } )$ , the goal is to shape a policy π whose induced responses align with a latent reward model that generated those preferences.

Two-stage RLHF is the standard way of carrying out preference-based alignment (Ziegler et al., 2019). However, it is computationally demanding (it requires training a separate reward model) and complex due to a two-stage pipeline (supervised learning for the reward model followed by RL policy optimization based on the learned reward model). Concretely, the reward model $r _ { \phi } ( s , a )$ is trained on preference pairs via a Bradley–Terry/Logistic objective (Bradley and Terry, 1952), maximizing log $\bar { \sigma } \big ( r _ { \phi } ( s , a _ { w } \bar { ) } - r _ { \phi } ( s , a _ { l } ) \big )$  over $( s , a _ { w } , a _ { l } )$ . The second stage then optimizes a KL-regularized objective of the form max<sub>π</sub> $\begin{array} { r l r } { { \mathbb E } _ { s \sim \rho , a \sim \pi ( \cdot | s ) } \big [ r _ { \phi } ( s , a ) \big ] } & { { } - } & { \beta D _ { \mathrm { K L } } \big ( \pi ( \cdot  { | } s ) \big | \big | \pi _ { \mathrm { r e f } } \big ( \cdot  { | } s \big ) \big ) } \end{array}$ , typically implemented with PPO-style updates. This stage is on-policy and rollout-heavy: the model must repeatedly generate samples to estimate advantages under $r _ { \phi } ,$ , maintain a stable KL to the reference policy $\pi _ { \mathrm { r e f } }$ (often the SFT model), and tune sensitive hyperparameters $\left( \mathrm { e . g . , } \beta , \right.$ , clip ranges, learning rates). In practice, this entails nontrivial engineering (reward hacking mitigation, variance reduction, response-length control) and significant compute for both reward-model training and RL updates, which motivates interest in lighter-weight alternatives.

The introduction of direct alignment algorithms such as Direct Preference Optimization (DPO) (Rafailov et al., 2023) was a landmark step that paved the way for lightweight alignment of a base model using preference data and only a single supervised training phase. DPO operates by explicitly solving the second, KL-regularized, policy optimization phase of RLHF and using it to reparameterize the first phase of reward learning in terms of the optimized policy, in effect achieving a one-step equivalent to the original two-step pipeline. This has been instrumental in enabling both industrial players and the open-source AI community to carry out fast alignment of models without the burden of additional resources. Many variants of DPO have since been developed catering to various aspects of direct alignment.

Despite its widespread appeal, however, the design of DPO rests on the idealized assumption that the policy class is tabular, i.e., it includes every possible input-output conditional probability distribution $\bar { ( } \pi ( \boldsymbol { a } \ \bar { } \ | \ s ) ) _ { s , a } .$ , where s and a denote prompt and response strings, respectively. This assumption enables the KL-regularized policy optimization problem to be solved in closed form and used explicitly to derive the equivalent supervised DPO loss (Rafailov et al., 2023, Appendix A.1).

In contrast, real-world LLMs are far from tabular and are, in fact, parametric policy classes, resulting naturally from the use of neural architectures (e.g., Transformers) with only a finite number of parameters. One may then ask: does minimizing the DPO loss over a non-tabular policy class still preserve the claimed equivalence with full two-stage RLHF? If not, then how does it differ from the ideal RLHF-optimal policy? Does it enjoy any guarantees with respect to the performance of the latter, and, if not, is there a principled fix?

We address these questions by introducing a systematic framework to uncover the geometry of direct preference optimization in parametric policy classes. Our study helps show how DPO essentially solves a misspecified statistical estimation problem in the space of reward functions that are implicitly parameterized by the underlying policy class. Misspecified estimation problems have been analyzed, in the statistics literature, for exhibiting undesirable phenomena such as inconsistent and arbitrary estimates that are sensitive to the input data distribution (White, 1982); we show that such phenomena also manifest in the DPO setting. Our analysis framework also allows us to modify DPO in a principled manner, to yield a new algorithm (AuxDPO) towards achieving the performance of two-stage RLHF

![](images/0c47017f61c145706b49072e3101bb0379bb3200b7dddfbf9c72b34ed4ea55a1.jpg)  
Figure 1: The geometry of DPO for parametric policies. (Left) DPO essentially performs a projection of the true preference-generating reward function $( r ^ { * }$ in black) onto the manifold of reward functions implicitly expressed by the policy class. $\operatorname { I f } r ^ { * }$ is in the manifold, then DPO finds the correct KL-regularized RLHF policy, but otherwise, the policy found (any orange point) is unreliable. (Right, zoomed inset) Locally linearizing the manifold around the base policy’s implicit reward function $( r _ { \theta _ { 0 } } )$ uncovers geometric insights. To reliably drive the solution to the reward function corresponding to the ideal RLHF solution $( r _ { \theta _ { \mathrm { R L H F } } }$ in blue), AuxDPO introduces additional controlled degrees of freedom, along the null space of a base-policy dependent matrix to sidestep misspecification.

in parametric models. More specifically, we make the following contributions:

1. We show that for general parametric policy classes, there is a misspecified statistical estimation problem at the core of the DPO algorithm by design: DPO loss minimization is equivalent to a weighted KL-projection of the true reward function r<sup>∗</sup> onto the (parametric, lower-dimensional) manifold of reward functions induced by the policy class. The weights of the projection are governed by the preference data collection frequencies (Fig. 1, left).

2. We show that DPO, in the misspecified setting, can suffer from various failure modes such as order reversal of preferences, overall reward reduction, sensitivity to preference data frequencies, etc. These failure modes occur even with ‘clean data’, i.e., infinite preference data generated using a BTL model based on an underlying true reward function r<sup>∗</sup> and fed to DPO. Our analysis is based on taking a local, linearized view of DPO’s implicit reward function manifold, which is accurate in the large-β regime.

3. On the other hand, studying the local geometry of two-stage RLHF for general parametric policy classes yields new insights about linear equivalence classes of reward functions. We use these insights to design AuxDPO, a new direct preference optimization algorithm that effectively mitigates the misspecification issue by introducing auxiliary controlled degress of freedom in reward space (Fig. 1, right). We demonstrate the effectiveness of AuxDPO in experiments. On real-world LLM preference tuning tasks, AuxDPO consistently outperforms DPO in aligning to held-out human preferences, confirming its practical value.

Related work. A recent line of work focuses on studying the insufficiency and implications of the tabular policy class assumption. Gao et al. (2024) and Swamy et al. (2025) call into question the tabular policy class assumption in the context of original two-stage RLHF. Tajwar et al. (2024) carry out an empirical investigation and note that the standard DPO loss can inadvertently reduce the model’s absolute likelihood of chosen responses as long as the relative probability between chosen and rejected responses increases. Meng et al. (2024) and Xu et al. (2024a) propose fixes based on considerations of margin and length normalization, and elimination of the reference policy.

Xu et al. (2024b) and Song et al. (2024) study the shortcomings of DPO arising from a lack of coverage, arguing that DPO can fail if a strong coverage condition is not met. The latter provide a counterexample to this end, showing the existence of an implicitly expressible reward function that is ε-approximately close to the true reward function but corresponds to a policy not in the KL neighborhood of the base policy. It is, however, unclear if such an implicit reward function can actually be output by DPO. Our fine-grained analysis in this paper shows that even with perfect coverage (uniform base policy), the policy returned by DPO can suffer from pathologies such as preference reordering and a decrease of overall expected reward (Proposition 3).

A separate line of work focuses on the gradient dynamics of DPO loss optimization and its impact on policy probabilities (Pal et al., 2024; Razin et al., 2024; Jian et al., 2025). It is shown that an individual gradient step on the standard DPO loss can result in likelihood displacement, where the probability of preferred responses can drop relative to the base policy for a gradient step. Our approach eschews assuming any specific optimization algorithm such as gradient descent and considering individual gradient steps, and instead focuses on showing failure modes such as likelihood displacement, preference reversal, reward reduction, etc. by studying the minimizer of the DPO loss.

Shi et al. (2025), perhaps the closest in spirit to our study, demonstrate a multi-armed bandit example that, when subjected to DPO with a log-linear policy class, does not cause any movement from the base policy. However, this example relies on a degenerate and symmetric reward function; we are able to demonstrate, in a fine-grained manner, that the policy can strictly worsen from the base policy in a manner that is highly sensitive to the preference data distribution (Proposition 3).

## 2 PRELIMINARIES

Let $\mathcal { D } = \left\{ \left( s ^ { ( i ) } , a _ { w } ^ { ( i ) } , a _ { l } ^ { ( i ) } \right) : i \in [ n ] \right\}$ be a dataset of n samples, where each sample has a prompt $s \in S$ , two responses $a _ { w } , a _ { l } \in \mathcal { A }$ such that $a _ { w } \succ a _ { l } , \mathrm { i . e . , } a _ { w }$ is a preferred response over $a _ { l }$ . We assume both S and A are finite sets, with $| { \cal S } | \cdot | { \cal A } | = m$ . The prompt s is sampled from a distribution ρ over ${ \mathcal { S } } .$ . The pair of responses $( a _ { w } , a _ { l } )$ is sampled from some base (reference) policy $\pi _ { \mathrm { r e f } }$ conditioned on $s , \mathrm { i . e . , } a _ { w } , a _ { l } \sim \pi _ { \mathrm { r e f } } ( \cdot | s )$ . The preference ordering between a pair of responses is assumed to be sampled according to a Bradley-Terry-Luce (BTL) model: the probability of a being preferred to $a ^ { \prime }$ is given by $p _ { s , a , a ^ { \prime } } ^ { \mathrm { B T L } } ( \check { r ^ { * } } ) = \sigma ( r ^ { * } ( \check { s } , a ) - \check { r } ^ { * } ( s , a ^ { \prime } ) ;$ ), where $r ^ { * } : \mathcal { S } \stackrel { \cdot } { \times } \mathcal { A } $ R is a (latent) reward function and $\begin{array} { r } { \sigma ( z ) : = \frac { 1 } { 1 + e ^ { - z } } } \end{array}$ is the sigmoid function.

Let $\pi _ { \theta } : { \mathcal { S } }  \Delta ( { \mathcal { A } } )$ be a policy (e.g., a language model) smoothly parameterized by a d-dimensional vector $\theta \in \mathbb { R } ^ { d } ( \mathrm { { \dot { e } } . g . }$ , the weights of a transformer), where $\Delta ( \mathcal { \bar { A } } )$ denotes the probability simplex over A. Let $\theta _ { 0 } \in \bar { \mathbb { R } } ^ { d }$ be the parameter for the base policy $\pi _ { \mathrm { r e f } }$ so that $\pi _ { \mathrm { r e f } } = \pi _ { \theta _ { 0 } }$ . A special case is the tabular policy class, where $d = m = | S | \cdot | A |$ and $\pi _ { \theta } ( a | s ) = \theta _ { s , a }$ (assuming without loss of generality that $\dot { \sum } _ { s , a } \theta _ { s , a } = 1 )$ . However, LLM policy classes are structured and non-tabular with parameter dimension $d \ll m , { \mathrm { e . g } }$ ., the neural softmax policy $\begin{array} { r } { \pi _ { \theta } ( a \mid s ) = \frac { \exp ( f _ { \theta } ( s , a ) ) } { \sum _ { a ^ { \prime } \in \mathcal { A } } \exp ( f _ { \theta } ( s , a ^ { \prime } ) ) } } \end{array}$ , where $f _ { \theta }$ is, say, a neural network.

For a given reward function $r ^ { * }$ , the optimal policy in a KL-regularized sense is obtained by maximiz ing the following objective:

$$
J (\theta ; r ^ {*}) = \mathbb {E} _ {\rho , \pi_ {\theta}} \left[ r ^ {*} (s, a) - \beta \log \frac {\pi_ {\theta} (a | s)}{\pi_ {\theta_ {0}} (a | s)} \right] = \mathbb {E} _ {\rho , \pi_ {\theta}} \left[ r ^ {*} (s, a) - \beta D _ {\mathrm{KL}} (\pi_ {\theta} (\cdot | s) | | \pi_ {\theta_ {0}} (\cdot | s)) \right].\tag{1}
$$

where $\mathbb { E } _ { \rho , \pi _ { \theta } }$ denotes expectation taken over $s \sim \rho ( \cdot )$ and $a \sim \pi _ { \theta } ( \cdot \mid s )$ , and $\beta > 0$ is a parameter that controls the amount of deviation from the base policy. We assume that $\theta ^ { * } \in \mathbb { R } ^ { d }$ is the unique minimizer of (1). For our analytical results, we will focus on the ‘local’ case $\beta \gg 1$ , meaning that the policy is not allowed to move beyond a local neighborhood of $\pi _ { \theta _ { 0 } }$ . When the policy class is tabular, it follows that the optimal policy $\pi _ { \theta ^ { \ast } }$ and the latent reward $r ^ { * }$ satisfy

$$
\pi_ {\theta^ {*}} (a | s) = \frac {1}{Z ^ {*} (s)} \pi_ {\theta_ {0}} (a | s) \exp (r ^ {*} (s, a) / \beta) \iff r ^ {*} (s, a) = \beta \log \frac {\pi_ {\theta^ {*}} (a | s)}{\pi_ {\theta_ {0}} (a | s)} + \beta \log Z ^ {*} (s)\tag{2}
$$

where $\begin{array} { r } { Z ^ { * } ( s ) = \sum _ { a \in \mathcal { A } } \pi _ { \theta _ { 0 } } ( a | s ) \exp ( r ^ { * } ( s , a ) / \beta ) } \end{array}$ is the normalizing or partition function (Rafailov et al., 2023). Under this reward-policy equivalence, the preference probabilities under the BTL model can be expressed using the optimal policy $\pi _ { \theta ^ { * } }$ ∗ and the base policy $\pi _ { \theta _ { 0 } }$ as follows.

$$
p _ {s, a, a ^ {\prime}} ^ {\mathrm{BTL}} \left(r ^ {*}\right) = \sigma \left(\beta \log \frac {\pi_ {\theta^ {*}} (a \mid s)}{\pi_ {\theta_ {0}} (a \mid s)} - \beta \log \frac {\pi_ {\theta^ {*}} \left(a ^ {\prime} \mid s\right)}{\pi_ {\theta_ {0}} \left(a ^ {\prime} \mid s\right)}\right) = \sigma \left(r _ {\theta^ {*}} ^ {\beta} (s, a) - r _ {\theta^ {*}} ^ {\beta} (s, a ^ {\prime})\right),
$$

where, for any $\theta \in \Theta , r _ { \theta } ^ { \beta } : \mathcal { S } \times \mathcal { A } $ R defined via $\begin{array} { r } { r _ { \theta } ^ { \beta } ( s , a ) : = \beta \log \frac { \pi _ { \theta } ( a \vert s ) } { \pi _ { \theta _ { 0 } } ( a \vert s ) } } \end{array}$ denotes the implicit reward function corresponding to the policy $\pi _ { \theta }$ at deviation level $\beta$ (note that $r _ { \theta _ { 0 } } ^ { \beta } \equiv 0$ by definition). Let $\mathcal { R } ^ { \beta } = \left\{ r _ { \theta } ^ { \beta } : \theta \in \mathbb { R } ^ { d } \right\} \subsetneq \mathbb { R } ^ { m }$ be the set of all implicit reward functions induced by the policy parameters $\theta$ at deviation level $\beta$ . Given the dataset D, DPO (Rafailov et al., 2023) finds the minimizer of the empirical DPO loss

$$
\mathcal {L} _ {\mathcal {D}} (\theta) = - \sum_ {i = 1} ^ {n} \log \sigma \left(r _ {\theta} ^ {\beta} \left(s ^ {(i)}, a _ {w} ^ {(i)}\right) - r _ {\theta} ^ {\beta} \left(s ^ {(i)}, a _ {l} ^ {(i)}\right)\right) = - \sum_ {s, a _ {w}, a _ {l}} N _ {s, a _ {w}, a _ {l}} \log \sigma \left(r _ {\theta} ^ {\beta} \left(s, a _ {w}\right) - r _ {\theta} ^ {\beta} \left(s, a _ {l}\right)\right),
$$

where $N _ { s , a _ { w } , a _ { l } }$ is the total number of pairwise preferences for which $a _ { w } \succ a _ { l }$ at s. If $n _ { s , a , a ^ { \prime } }$ denotes the total number of pairwise preferences for the triplet $( s , a , a ^ { \prime } )$ (assumed non-random and fixed in advance), then $N _ { s , a _ { w } , a _ { l } } \sim \mathtt { B }$ inomial $\big ( n _ { s , a _ { w } , a _ { l } } , p _ { s , a _ { w } , a _ { l } } ^ { \mathrm { B T L } } ( r ^ { * } ) \big )$ , yielding the population DPO loss

$$
\mathcal {L} (\theta) = - \sum_ {s, a, a ^ {\prime}} n _ {s, a, a ^ {\prime}} \left[ p _ {s, a, a ^ {\prime}} ^ {\mathrm{BTL}} (r ^ {*}) \log p _ {s, a, a ^ {\prime}} ^ {\mathrm{BTL}} (r _ {\theta} ^ {\beta}) + \left(1 - p _ {s, a, a ^ {\prime}} ^ {\mathrm{BTL}} (r ^ {*}) \log \left(1 - p _ {s, a, a ^ {\prime}} ^ {\mathrm{BTL}} (r _ {\theta} ^ {\beta})\right)\right) \right]\tag{3}
$$

We take up this loss for DPO as our main object of study in the sequel.

## 3 REWARD MISSPECIFICATION IN DPO

Let $r ^ { * }$ denote the m-dimensional vector of latent rewards $( r ^ { * } ( s , a ) ) _ { s , a }$ , by slightly abusing notation. Proposition 1 (DPO is weighted KL-projection). Assume that the pairwise preference data are drawn from $p _ { s , a _ { w } , a _ { l } } ^ { \mathrm { B T L } } ( r ^ { * } )$ ) for some $r ^ { \ast } \in \mathbb { R } ^ { m }$ , with $n _ { s , a , a ^ { \prime } }$ preference pairs drawn for each triplet $( s , a , a ^ { \prime } ) . \ : I f \theta _ { D P O }$ minimizes the DPO loss (3), then its corresponding implicit rewardfunction satisfies

$$
r _ {\theta_ {D P O}} ^ {\beta} = \arg \min _ {r \in \mathcal {R} ^ {\beta}} \sum_ {s, a, a ^ {\prime}} n _ {s, a, a ^ {\prime}} d _ {K L} \big (p _ {s, a, a ^ {\prime}} ^ {\mathrm{BTL}} (r ^ {*}) | | p _ {s, a, a ^ {\prime}} ^ {\mathrm{BTL}} (r) \big),\tag{4}
$$

where $d _ { K L } ( p | | q )$ is the Kullback-Leibler divergence between two Bernoulli random variables with parameters $p , q .$

The result establishes that DPO projects (according to reverse-KL divergence weighted by pairwise preference counts $n _ { s , a , a ^ { \prime } } )$ the true reward function $r ^ { * }$ onto the set of implicit reward functions $\mathcal { R } ^ { \beta }$ (with the corresponding policy being returned after fine-tuning). It implies that if $r ^ { * }$ is realizable, $\mathrm { i . e . , }$ $r ^ { * } = r _ { \theta } ^ { \beta }$ for some $\theta ,$ then this projection (trivially) finds $r _ { \theta } ^ { \beta }$ and hence the policy $\pi _ { \theta }$ , which coincides with the RLHF policy, $\mathrm { i } . \mathrm { e } . , \theta = \theta ^ { \ast }$

However, i $\displaystyle { \mathrm {  ~ f ~ } r ^ { * } \notin \mathcal { R } ^ { \beta } }$ (which is typically the case since $\mathcal { R } ^ { \beta }$ is a lower-dimensional (d-dimensional) manifold of $\mathbb { R } ^ { m } )$ , then we are in the misspecified estimation setting. In this case, the result of the KL-projection will, in general, be dependent on the exact weighted projection which is determined by the preference data frequencies $\left( n _ { s , a , a ^ { \prime } } \right) _ { s , a , a ^ { \prime } }$ . We demonstrate, in the next section, that the policies resulting from DPO’s misspecified estimation enjoy no guarantees: they could suffer from preference reversal, or even worse, yield a lower average reward than even the base policy (contrary to two-stage RLHF, where the average reward can never decrease).

## 3.1 LOCAL GEOMETRY OF DPO

Let us locally approximate the implicit reward $r _ { \theta } ^ { \beta } ( s , a )$ via its first-order Taylor expansion around $\theta _ { 0 } { \mathrm { : } }$

$$
r _ {\theta} ^ {\beta} (s, a) \approx r _ {\theta_ {0}} ^ {\beta} (s, a) + \left\langle \nabla r _ {\theta_ {0}} (s, a), \theta - \theta_ {0} \right\rangle = \beta \left\langle \frac {\nabla \pi_ {\theta_ {0}} (a | s)}{\pi_ {\theta_ {0}} (a | s)}, \theta - \theta_ {0} \right\rangle = \beta \nabla \log \pi_ {\theta_ {0}} (a | s) ^ {\top} (\theta - \theta_ {0})
$$

Define the $d \times m$ Jacobian matrix $A _ { \theta _ { 0 } }$ as the matrix containing $\nabla$ log $\pi _ { \theta _ { 0 } } ( a | s )$ in its $( s , a )$ -th column. With this, the local linear approximation of $r _ { \theta } ^ { \beta }$ takes the form $\overline { r } _ { \theta } ^ { \beta } \ : = \ : \beta A _ { \theta _ { 0 } } ^ { \top } ( \theta \ : - \ : \theta _ { 0 } )$ . Since $\left\{ \overline { r } _ { \theta } ^ { \beta } : \theta \in \mathbb { R } ^ { d } \right\}$ is the column space of $A _ { \theta } ^ { \top }$ , we arrive at the linear manifold approximation $\mathcal { R } ^ { \beta } \approx \dot { \mathcal { C } } ( A _ { \theta _ { \mathrm { { n } } } } ^ { \top } )$ , for θ in the local neighborhood of $\theta _ { 0 }$ . Note that this linear approximation is independent of the choice of $\beta$ and is a function only of the policy class and base policy $\pi _ { \theta _ { 0 } }$

Remark 2. The error in the linear approximation of the manifold $\mathcal { R } ^ { \beta } b y \mathcal { C } ( A _ { \theta _ { 0 } } ^ { \top } )$ can be controlled to within any arbitrary tolerance by taking the policy deviation parameter $\beta$ to be sufficiently large. Proposition 8formally controls the approximation error. In the sequel, we only work with the linear approximation to develop our results.

Armed with this local linearization of the implicit reward manifold, we now show an example of a single-prompt, 3-response setting, with a 1-dimensional policy parameter, in which DPO exhibits counterintuitive and unexpected behavior, including preference reversal, reward reduction, and high sensitivity to the pairwise preference data counts $n _ { s , a , a ^ { \prime } }$

Proposition 3 (Example of DPO with preference reversal and reward decrease). There exists a promptless policy optimization problem with three responses and linear softmax policy class parameterized with a 1-dimensional parameter θ and a true reward vector $r ^ { * } \in \dot { \mathbb { R } } ^ { 3 }$ such that $D P O ,$ , carried out with pairwise preferences generated according to $B T L ( r ^ { * } )$ , sufficiently large $\beta$ and pairwise counts $\{ n _ { i , j } \}$ , yields a policy $\pi _ { \theta }$ such that $( i ) \pi _ { \theta }$ favors the response with second highest reward, (ii) $\pi _ { \theta }$ decreases (resp. increases) the probability of the action with the highest reward (resp. second highest reward) with respect to the base policy $\pi _ { \boldsymbol { \theta } _ { 0 } } .$ , and (iii) $\pi _ { \theta } ^ { \top } r ^ { * } < \pi _ { \theta _ { 0 } } ^ { \top } r ^ { * }$

Proof. Consider the following example with 3 responses $a _ { 1 } , a _ { 2 } , a _ { 3 }$ with latent rewards $r ^ { * } = [ 2 , 3 , 1 ]$ , which yields the preference order $a _ { 2 } \succ a _ { 1 } \succ a _ { 3 }$ . The pairwise preference counts in the dataset are highly imbalanced in a way that $n _ { 3 , 1 } \gg \operatorname* { m a x } \{ n _ { 1 , 2 } , n _ { 2 , 3 } \}$ . The policy is given by $\begin{array} { r } { \pi _ { \theta } = \frac { 1 } { Z } [ e ^ { \theta } , e ^ { - \theta } , 1 ] } \end{array}$ , where $Z = { \overset { \cdot } { 1 } } + e ^ { \theta } + e ^ { - \theta }$ . We take a uniform base policy $( \mathrm { i . e . , } \theta _ { 0 } = 0 )$ . Under the BTL model, $r ^ { * } \equiv [ 1 , 2 , 0 ]$ since both induce the same preference distribution. Hence, we will work with this equivalent $r ^ { * }$ . The setting is depicted in Figure 2.

In this setting, the policy gradient matrix takes the form $\begin{array} { r } { A _ { \theta } = \frac { 1 } { Z } \overline { { \left[ 1 + 2 e ^ { - \theta } , - ( 1 + 2 e ^ { \theta } ) , e ^ { - \theta } - e ^ { \theta } \right] } } } \end{array}$ . Hence $A _ { \theta _ { 0 } } = [ 1 , - 1 , 0 ]$ and thus $\mathcal { R } ^ { \beta }$ ≈ span $( [ 1 , - 1 , 0 ] )$ in the local neighborhood of $\theta _ { 0 } = 0$ . Now, since $^ { n _ { 1 , 3 } }$ dominates the other two comparison counts, the solution to the optimization problem in $( 4 ) , \mathrm { e . g }$ • ,

$$
\arg \min _ {r \in \mathcal {R} ^ {\beta}} \sum_ {i \neq j} n _ {i, j} d _ {\mathrm{KL}} \big (p _ {i, j} ^ {\mathrm{BTL}} (r ^ {*}) | | p _ {i, j} ^ {\mathrm{BTL}} (r) \big)
$$

will push the probability that $a _ { 1 }$ is preferred over $a _ { 3 } , \mathrm { i . e . }$ $p _ { 1 , 3 } ^ { \mathrm { B T L } } ( r _ { \theta } ^ { \beta } )$ close to $p _ { 1 , 3 } ^ { \mathrm { B T L } } ( r ^ { * } )$ , yielding $r _ { \theta } ^ { \beta } ( a _ { 1 } ) - r _ { \theta } ^ { \beta } ( a _ { 3 } ) =$ $O ( \alpha )$ for some $\alpha > 0$ . Moreover, since $r _ { \theta } ^ { \beta }$ should lie in span $( [ 1 , - 1 , 0 ] )$ , DPO will end up with the reward function $r _ { \theta } ^ { \beta } \approx [ \alpha , - \alpha , 0 ]$ . This, in turn, will make $a _ { 1 }$ to be preferred over $a _ { 2 }$ by the learned reward, indicating a preference reversal from that given by $r ^ { * }$ . Furthermore, from the equivalence relation (5), we get the post-optimized policy parameter $\theta = O ( \alpha )$ . This yields $\mathrm { ( i ) } \pi _ { \theta } ( a _ { 1 } ) > \pi _ { \theta } ( a _ { 2 } )$ (the policy favors a sub optimal response) (ii) $\pi _ { \theta } ( a _ { 2 } ) < \pi _ { \theta _ { 0 } } ( a _ { 2 } )$ and $\pi _ { \theta } ( a _ { 1 } ) > \pi _ { \theta _ { 0 } } ( a _ { 1 } )$ (likelihood of the optimal response decreases and that of a suboptimal response increases) and (iii) $\begin{array} { r } { \pi _ { \theta } ^ { \top } r ^ { * } = \frac { e ^ { \alpha } + 2 e ^ { - \alpha } } { 1 + e ^ { \alpha } + e ^ { - \alpha } } < 1 = \pi _ { \theta _ { 0 } } ^ { \top } r ^ { * } } \end{array}$ (average reward decreases relative to the base policy).

![](images/9561ad59906f3098dc4d8a59f8fef3be6c93e17b5709baa7c3bf6eb81b04bb11.jpg)  
Figure 2: An example with 3 responses and 1-d policy parameter showing failure modes of $\mathrm { D P O } . \hat { r } ^ { * }$ is the latent reward. The red line denotes the linear approximation $\mathcal { C } ( A _ { \theta _ { 0 } } ^ { \top } )$ of the implicit reward manifold $\mathcal { R } ^ { \beta }$ The region shaded in orange represents all possible implicit reward functions that DPO can possibly project onto, depending on the relative proportion of pairwise preference counts $n _ { 1 , 2 } , n _ { 2 , 3 } , n _ { 3 , 1 }$ . If $^ { n _ { 3 , 1 } }$ dominates the rest, then the projection $r _ { \theta } ^ { \beta }$ induces a postoptimized policy parameter θ $> 0 .$ , leading to preference reversal and reduction of expected reward, causing DPO to fail.

The example exhibits the following aspects:

1. There are pairs $( i , j ) \ : ( \mathrm { e . g . , } i = 2 , j = 1 )$ where $a _ { i } \succ a _ { j }$ with respect to $r ^ { * } \left( \mathrm { i . e . , } r ^ { * } ( a _ { i } ) > r ^ { * } ( a _ { j } ) \right)$ but $\pi _ { \boldsymbol { \theta } } ( a _ { i } )$ decreases and $\pi _ { \theta } ( a _ { j } )$ increases with respect to the base policy. This is more extreme than likelihood displacement, where $\pi _ { \boldsymbol { \theta } } ( a _ { i } )$ and $\pi _ { \theta } ( a _ { j } )$ both decrease or increase together but their difference $\pi _ { \theta } ( a _ { i } ) - \pi _ { \theta } ( a _ { j } )$ is presumed to increase (Razin et al., 2024; Pal et al., 2024).

2. The expected reward with respect to $r ^ { * }$ decreases from $\pi _ { \theta _ { 0 } }$ , whereas two-stage RLHF would have increased the expected reward on any policy class (assuming $r ^ { * }$ is learnt accurately in the first stage).

3. Our example is based on the DPO population loss, which is effectively DPO operating in the data-rich regime where unlimited pairwise preference data from a BTL model are used as input. This circumvents the issue of failure modes of DPO known to occur due to scarce data sampling (Tajwar et al., 2024). We show that failure modes arise due to the inherent misspecified geometry induced by the lack of model capacity, interacting with the frequencies of pairwise sampling.

4. Our example applies in the strongest possible ‘oracle’ optimization model where we assume that the (population) DPO loss can be optimized exactly. Our results are not dependent on the idiosyncrasies or specifics of what algorithm is used to optimize the DPO loss (e.g., gradient descent and variants as explicitly considered in other works (Razin et al., 2024; Pal et al., 2024)), as long as it is (near) optimal.

5. Sensitivity of DPO with respect to the preference data distribution: In the example, if the pairwise preference counts are such that $n _ { 1 , 2 } \gg \{ n _ { 2 , 3 } , n _ { 3 , 1 } \}$ , then DPO would learn the desired reward function $r _ { \theta } ^ { \beta } \approx [ - \alpha , \alpha , 0 ]$ , which would, in turn, induce a policy parameter $\theta < 0 ,$ , escaping the failure modes effectively. Therefore, depending on the relative proportions of pairwise preference counts $\{ n _ { i , j } \} _ { i , j } ,$ , one could either get the desired result or the opposite one, as in the example, or perhaps no movement at all, making DPO sensitive to preference data sampling distribution. This sensitivity to the exact preference distribution also represents a failure mode of DPO.

Remark 4 (Global coverage is not sufficient for optimality). Song et al. (2024) argue that global coverage, i.e., max $\overline { { \cdot s , a } } \ \frac { \pi _ { \theta } ( a | s ) } { \pi _ { \theta _ { 0 } } ( a | s ) } \leqslant C \ \forall \theta \in \mathbb { R } ^ { d }$ , is necessary for DPO to converge to the optimal policy $\theta ^ { * }$ . In our example, $\pi _ { \theta _ { 0 } }$ satisfies this with $C = 3 .$ . However, this condition is not sufficientfor DPO to perform optimally, since, as we have seen, DPO can learn an unaligned reward model depending on the relativefrequency ofpreference counts.

## 4 TOWARDS MITIGATING DPO’S PITFALLS

We propose to address the misspecification issue of DPO by first studying the nature of the RLHF policy optimization step in a suitably local sense (assuming ideal reward learning), and then using the insights gained to encourage movement towards this solution.

## 4.1 LOCAL GEOMETRY OF RLHF OPTIMIZATION

We locally approximate the objective $J ( \theta ; r ^ { * } )$ in (1) around the base policy $\pi _ { \theta _ { 0 } }$ . To do so, we approximate the expected reward using a first-order Taylor series expansion and the KL penalty using a second-order Taylor series expansion:

$$
\mathbb {E} _ {\rho , \pi_ {\theta}} [ r ^ {*} (s, a) ] \approx \mathbb {E} _ {\rho , \pi_ {\theta_ {0}}} [ r ^ {*} (s, a) ] + (\theta - \theta_ {0}) ^ {\top} A _ {\theta_ {0}} D _ {\rho , \theta_ {0}} r ^ {*}, D _ {\mathrm{KL}} (\pi_ {\theta} | | \pi_ {\theta_ {0}}) \approx \frac {1}{2} (\theta - \theta_ {0}) ^ {\top} F _ {\rho , \theta_ {0}} (\theta - \theta_ {0})
$$

where $D _ { \rho , \theta _ { 0 } }$ is a diagonal matrix with scaled base policy-probabilities $\rho ( s ) \pi _ { \theta _ { 0 } } ( a | s )$ in the diagonal entries, and $F _ { \rho , \theta _ { 0 } } = \mathbb { E } _ { \rho , \pi _ { \theta _ { 0 } } } \left[ \nabla \log \pi _ { \theta _ { 0 } } ( a | s ) \nabla \right.$ log $\pi _ { \theta _ { 0 } } ( a | s ) ^ { \top } ]$ denotes the Fisher information matrix at π<sub>θ</sub> (Amari, 2016). Introducing the d×m matrix $A _ { \rho , \theta _ { 0 } } = \bar { A } _ { \theta _ { 0 } } D _ { \rho , \theta _ { 0 } }$ containing the scaled gradients $\rho ( s ) \nabla \pi _ { \theta _ { 0 } } ( a | s )$ in its columns, we arrive at a local quadratic approximation of the objective

$$
J (\theta ; r ^ {*}) \approx \mathbb {E} _ {\rho , \pi_ {\theta_ {0}}} [ r ^ {*} (s, a) ] + (\theta - \theta_ {0}) ^ {\top} A _ {\rho , \theta_ {0}} r ^ {*} - \frac {\beta}{2} (\theta - \theta_ {0}) ^ {\top} F _ {\rho , \theta_ {0}} (\theta - \theta_ {0}).
$$

Remark 5. Just as described in Remark 2, the error in this local quadratic approximation of $J ( \theta ; r ^ { * } )$ can be controlled to a desired level ofaccuracy by taking β to be sufficiently large; see Proposition 8.

Based on this local quadratic approximation of the RLHF objective function, we can deduce (via first-order optimality conditions) a relation that suitably generalizes (2), between the optimal policy $\pi _ { \theta ^ { \ast } }$ ∗ and the latent reward $r ^ { * }$ , to parametric policy classes:

$$
A _ {\rho , \theta_ {0}} r ^ {*} = \beta F _ {\rho , \theta_ {0}} (\theta^ {*} - \theta_ {0}) \iff \theta^ {*} = \theta_ {0} + \frac {1}{\beta} F _ {\rho , \theta_ {0}} ^ {\dagger} A _ {\rho , \theta_ {0}} r ^ {*}.\tag{5}
$$

This has the form of a natural policy gradient update (Kakade, 2001). Moreover, it partitions the set of all reward functions into equivalence classes as follows. For each policy parameter θ, define

$$
\mathcal {R} _ {\mathrm{eq}} ^ {\beta} (\theta) = \left\{r \in \mathbb {R} ^ {m}: A _ {\rho , \theta_ {0}} r = \beta F _ {\rho , \theta_ {0}} \left(\theta - \theta_ {0}\right) \right\}.\tag{6}
$$

Lemma 6 (Equivalence classes induced by RLHF). For a fixed $\theta \in \mathbb { R } ^ { d } ,$ , two reward vectors $r _ { 1 } , r _ { 2 } \in \mathbb { R } ^ { m }$ belong to $\mathcal { R } _ { e q } ^ { \beta } ( \theta )$ if and only if they differ by a vector $\delta \in \mathcal { N } ( A _ { \rho , \theta _ { 0 } } )$ , i.e., a null space element of $A _ { \rho , \theta _ { 0 } }$

Note that for tabular policies, the class $\mathcal { R } _ { \mathrm { e q } } ^ { \beta } ( \theta )$ essentially reduces to the singleton $r _ { \theta } ^ { \beta }$ , the “implicit reward” of DPO, i.e., $\begin{array} { r } { r _ { \theta } ^ { \beta } ( s , a ) = \beta \log \frac { \pi _ { \theta } ( a \vert s ) } { \pi _ { \theta _ { 0 } } ( a \vert s ) } . } \end{array}$ 1

Interestingly, we can show that DPO’s linearized reward functions $\overline { r } _ { \theta } ^ { \beta }$ are in bijective correspondence with RLHF’s equivalence classes, with each linearized reward function being the minimum-norm representative of its equivalence class:

Proposition 7 (Relationship between RLHF equivalence classes and DPO linearization). For a base $p o l i c y \pi _ { \theta _ { 0 } }$ and KL penalty $\beta > 0$ , the reward vector $r \in \mathcal { R } _ { e q } ^ { \beta } ( \theta )$ which has the minimum Mahalonobis norm $\| r \| _ { D _ { \rho , \theta _ { 0 } } } : = \sqrt { r ^ { \top } D _ { \rho , \theta _ { 0 } } r } ,$ , i.e., argmin $\mathsf { l } _ { \boldsymbol { r } \in \mathbb { R } ^ { m } } \left\| \boldsymbol { r } \right\| _ { D _ { \rho , \theta _ { 0 } } }$ such that $A _ { \rho , \theta _ { 0 } } r = \beta F _ { \rho , \theta _ { 0 } } ( \theta - \theta _ { 0 } )$ is given by $\overline { r } _ { \theta } ^ { \beta } = \beta A _ { \theta _ { 0 } } ^ { \top } ( \theta - \theta _ { 0 } )$ , the local linearization $\begin{array} { r } { \jmath f r _ { \theta } ^ { \beta } ( s , a ) = \beta \log \frac { \pi _ { \theta } ( a | s ) } { \pi _ { \theta _ { 0 } } ( a | s ) } } \end{array}$

The following result helps to justify the local linear approximations made to (i) the DPO implicit reward function manifold and (ii) the RLHF policy optimization objective, by showing that error between the local approximations and the original functions can be controlled to an arbitrarily prescribed level by taking the deviation parameter $\beta$ to be suitably large (so that DPO and RLHF essentially reduce to optimization over policies in a neighborhood of $\pi _ { \theta _ { 0 } } )$

Proposition 8 (Approximation errors). $F i x \ \varepsilon \ > \ 0 .$ There exists a bounded neighborhood $\mathcal { E } \ \in \ \mathbb { R } ^ { d }$ containing $\theta _ { 0 } ,$ a bounded set $\mathcal { R } \subset \mathbb { R } ^ { m }$ , and $\beta _ { \mathrm { m i n } } ~ > ~ 0$ such that for every deviation parameter $\beta { \mathrm { ~  ~ { ~ > ~ } ~ } } \beta _ { \mathrm { m i n } }$ , we have $( i ) \ r _ { \theta } ^ { \beta } \ \in \ \mathcal { R }$ for each $\theta ~ \in ~ { \mathcal E } , ~ ( i i ) ~ r _ { \mathrm { D P O } } ^ { \beta } ~ = ~ r _ { \theta } ^ { \beta }$ for some $\theta \in \mathcal { E } ,$ , (iii) $r _ { \theta } ^ { \beta } ( s , a ) \ : - \ : \beta \nabla$ log $\pi _ { \theta _ { 0 } } ( a | s ) ^ { \top } ( \theta - \theta _ { 0 } ) \ \leq \ \varepsilon \ f o r$ each $\theta \in \mathcal { E } ,$ , and (iv) $\begin{array} { r } { J ( \theta ; r ^ { * } ) - \left( \mathbb { E } _ { \rho , \pi \theta _ { 0 } } [ r ^ { * } ( s , a ) ] + ( \theta - \theta _ { 0 } ) ^ { \top } A _ { \rho , \theta _ { 0 } } r ^ { * } - \frac { \beta } { 2 } ( \theta - \theta _ { 0 } ) ^ { \top } F _ { \rho , \theta _ { 0 } } ( \theta - \theta _ { 0 } ) \right) \le \varepsilon f o r } \end{array}$ each $\theta \in { \mathcal { E } }$

## 4.2 THE AUXDPO ALGORITHM

We introduce a new direct alignment algorithm, AuxDPO, which leverages our insights from the analysis of the local geometry of DPO and RLHF policy optimization to mitigate the failure modes of DPO in a principled manner.

Recall that, in the general setting where the true reward function $r ^ { * }$ is misspecified (outside the implicit reward manifold $\mathcal { R } ^ { \beta } )$ , then DPO finds the optimal policy $\theta ^ { * }$ only if the reverse-KL projection of $r ^ { * }$ on $\mathcal { R } ^ { \beta }$ fortuitously lands on $r _ { \theta ^ { * } } ^ { \beta }$ (Proposition 1). This depends crucially on relative proportions of pairwise preference counts $\{ n _ { i , j } \} _ { i , j }$ in the dataset and, as such, is beyond the learner’s control.

Instead, we take the following approach to encourage the optimization to move towards $r _ { \theta ^ { * } } ^ { \beta }$ . Note that by our local analysis of the RLHF optimization step (Sec 4.1), the true (misspecified) reward function $r ^ { * }$ and $r _ { \theta ^ { * } } ^ { \beta }$ differ by an element of the nullspace of $A _ { \rho , \theta _ { 0 } } , \mathrm { i . e . , } r ^ { * } = r _ { \theta ^ { * } } ^ { \beta } + \delta .$ , where $\delta \in \mathcal { N } ( A _ { \rho , \theta _ { 0 } } ) \subsetneq \mathbb { R } ^ { m }$ . Therefore, if we allow the search in the reward space $\mathbb { R } ^ { m }$ to utilize additional degrees of freedom δ along this nullspace (in addition to the usual degrees of freedom via $\theta )$ , then $r ^ { * }$ is no longer misspecified in this augmented representation. This should ideally result in the variables $\theta \in \mathbb { R } ^ { d }$ and $\delta \in \mathcal N ( A _ { \rho , \theta _ { 0 } } )$ settling in a manner that achieves $r ^ { * } = r _ { \theta ^ { * } } ^ { \beta } + \delta ^ { * }$

Observe that since $A _ { \rho , \theta _ { 0 } } = A _ { \theta _ { 0 } } D _ { \rho , \theta _ { 0 } }$ , both $\mathcal { N } ( A _ { \theta _ { 0 } } )$ and $\mathcal { N } ( A _ { \rho , \theta _ { 0 } } )$ have the same dimension. Moreover, $\mathcal { C } ( A _ { \theta _ { 0 } } ^ { \top } )$ and $\mathcal { C } ( A _ { \theta _ { 0 } } )$ ) also have same dimension. Thus, by the ranknullity theorem, by varying both θ and δ, we can search over the entire space of the rewards $\mathbb { R } ^ { m }$ , contrary to DPO which searches only over $\mathcal { C } ( A _ { \theta _ { 0 } } ^ { \top } )$ (under linear approximation), a manifold in $\mathbb { R } ^ { m }$ with dimension at most d.

To this end, we introduce auxiliary variables $\delta \in \mathbb { R } ^ { m }$ into the population loss of DPO (3), and minimize it jointly over $\bar { \theta \in \mathbb { R } ^ { d } } , \delta$ while enforcing the nullspace constraint $\delta \in \mathcal { N } ( A _ { \rho , \theta _ { 0 } } )$ . This gives us the AuxDPO procedure<sup>2</sup>:

![](images/a31ce13016b29f82f963773f1f1d5f9061b2582cb1c76f1e87a551884357b610.jpg)

$$
\begin{array}{c} \underset {\theta \in \mathbb {R} ^ {d}, \delta \in \mathcal {N} (A _ {\rho , \theta_ {0}})} {\text {minimize}} \mathcal {L} (\theta , \delta), \quad \text {where} \\ \mathcal {L} (\theta , \delta) = - \sum_ {s, a, a ^ {\prime}} n _ {s, a, a ^ {\prime}} \Big [ p _ {s, a, a ^ {\prime}} ^ {\mathrm{BTL}} (r ^ {*}) \log p _ {s, a, a ^ {\prime}} ^ {\mathrm{BTL}} (r _ {\theta , \delta} ^ {\beta}) \\ \quad + \left(1 - p _ {s, a, a ^ {\prime}} ^ {\mathrm{BTL}} (r ^ {*}) \log \left(1 - p _ {s, a, a ^ {\prime}} ^ {\mathrm{BTL}} (r _ {\theta , \delta} ^ {\beta})\right) \right]. \end{array}\tag{7}
$$

Proposition 9 (Auxiliary variables bypass misspecification). Let the hypothesis of Proposition 1 hold. Fix a tolerance $\varepsilon > 0$ . Then, for sufficiently large $\beta > 0 ,$ , the optimization (7) is minimized at $\theta = \theta ^ { * }$ up to error $O ( \varepsilon )$

We develop AuxDPO’s corresponding empirical loss version, which can be implemented with a finite preference dataset. We convert the constrained optimization over the set $\mathcal { N } ( A _ { \rho , \theta _ { 0 } } ) \subset \mathbb { R } ^ { m }$ to an unconstrained one over $\mathbb { R } ^ { m }$ by

Figure 3: AuxDPO fixes DPO’s misspecification. $r ^ { * }$ is the latent reward. The blue line denotes the equivalence class $\mathcal { R } _ { \mathrm { e q } } ^ { \beta } ( \theta ^ { * } )$ of all reward functions that yield the RLHFoptimal policy $\pi _ { \theta ^ { \ast } }$ . The red line denotes the linear approximation $\mathcal { C } ( A _ { \theta _ { 0 } } ^ { \top } )$ of the implicit reward manifold $\mathcal { R } ^ { \beta }$ . The region shaded in orange represents all possible implicit reward functions that DPO can possibly project onto. The green line depicts the domain of optimization over AuxDPO’s auxiliary variables $\delta \in \mathcal { N } ( A _ { \rho , \theta _ { 0 } } )$ for a fixed θ (the line shifts in parallel for other $\theta )$ . δ introduces additional degrees of freedom, which help push the KL projection of $r ^ { * }$ to lie in the equivalence class ${ \mathcal { R } } _ { \theta ^ { * } }$ . The projection induces the optimal policy π<sub>θ</sub>∗ .

adding the penalty term $\big \| A _ { \rho , \theta _ { 0 } } \delta \big \| _ { 2 } ^ { 2 }$ to the log-loss. Note that $A _ { \rho , \theta _ { 0 } } \delta = \mathbb { E } _ { \rho , \pi _ { \theta _ { 0 } } } \left[ \delta ( s , a ) \nabla \log \pi _ { \theta _ { 0 } } ( a | s ) \right]$ We approximate it with a given dataset $\mathcal { D } = ( s ^ { ( i ) } , a _ { w } ^ { ( i ) } , a _ { l } ^ { ( i ) } ) _ { i = 1 } ^ { n }$ in Monte-Carlo fashion. This leads to the empirical AuxDPO loss $\mathcal { L } _ { \mathcal { D } }$ over variables $\theta \in \mathbb { R } ^ { d }$ and $\delta \in \mathbb { R } ^ { 2 n }$

$$
\begin{array}{l} \mathcal {L} _ {\mathcal {D}} (\theta , \delta) = - \frac {1}{n} \sum_ {i = 1} ^ {n} \log \sigma \left(r _ {\theta} ^ {\beta} (s ^ {(i)}, a _ {w} ^ {(i)}) - r _ {\theta} ^ {\beta} (s ^ {(i)}, a _ {l} ^ {(i)}) + \delta (s ^ {(i)}, a _ {w} ^ {(i)}) - \delta (s ^ {(i)}, a _ {l} ^ {(i)})\right) \\ \qquad + \lambda \left\| \frac {1}{2 n} \sum_ {i = 1} ^ {n} \left(\delta (s ^ {(i)}, a _ {w} ^ {(i)}) \nabla \log \pi_ {\theta_ {0}} (a _ {w} ^ {(i)} \mid s ^ {(i)}) + \delta (s ^ {(i)}, a _ {l} ^ {(i)}) \nabla \log \pi_ {\theta_ {0}} (a _ {l} ^ {(i)} \mid s ^ {(i)})\right) \right\| ^ {2}. \end{array}
$$

where $\delta = \left\{ \delta ( s ^ { ( i ) } , a _ { w } ^ { ( i ) } ) , \delta ( s ^ { ( i ) } , a _ { l } ^ { ( i ) } ) \right\} _ { i = 1 } ^ { n } \in \mathbb { R } ^ { 2 n }$ denotes the vector of auxiliary variables (typically 2n ≪ m) and $\lambda > 0$ is a hyper-parameter that is responsible for enforcing the nullspace constraint on δ. Note that the total number of trainable parameters is $d + 2 n = O ( d )$ , since typically, $n \ll d .$

## 5 EXPERIMENTS

Datasets. We conduct evaluations on two benchmark datasets: REWARDBENCH V2 and MMLU-PRO. REWARDBENCH ${ \tt V } 2$ (Malik et al., 2025) contains 1.87K prompts covering categories like factuality, precise instruction following, and focus, with each prompt containing a chosen and a rejected response. MMLU-PRO (Wang et al., 2024b) is a multi-task understanding dataset containing 12K complex questions across various disciplines. Each question has 10 possible answers and a

<table><tr><td>Model</td><td>Dataset</td><td>Method</td><td>DPO</td><td>AuxDPO</td><td>IPO</td><td>DPOP</td></tr><tr><td rowspan="4">Llama3.1-8B</td><td>MMLU-PRO</td><td>ID</td><td>57.14</td><td>63.26</td><td>59.18</td><td>61.22</td></tr><tr><td>MMLU-PRO</td><td>OOD</td><td>8.16</td><td>14.28</td><td>10.20</td><td>6.12</td></tr><tr><td>REWARDBENCH V2</td><td>ID</td><td>56.01</td><td>66.72</td><td>61.34</td><td>62.27</td></tr><tr><td>REWARDBENCH V2</td><td>OOD</td><td>14.31</td><td>32.44</td><td>20.17</td><td>19.87</td></tr><tr><td rowspan="4">Llama3.2-1B</td><td>MMLU-PRO</td><td>ID</td><td>39.58</td><td>45.83</td><td>43.75</td><td>44.21</td></tr><tr><td>MMLU-PRO</td><td>OOD</td><td>6.25</td><td>12.52</td><td>14.58</td><td>4.16</td></tr><tr><td>REWARDBENCH V2</td><td>ID</td><td>77.21</td><td>86.37</td><td>69.72</td><td>71.21</td></tr><tr><td>REWARDBENCH V2</td><td>OOD</td><td>14.11</td><td>43.27</td><td>20.42</td><td>18.76</td></tr><tr><td rowspan="4">Qwen3-0.6B</td><td>MMLU-PRO</td><td>ID</td><td>53.12</td><td>61.78</td><td>47.48</td><td>56.67</td></tr><tr><td>MMLU-PRO</td><td>OOD</td><td>11.34</td><td>22.22</td><td>15.56</td><td>17.78</td></tr><tr><td>REWARDBENCH V2</td><td>ID</td><td>55.10</td><td>65.31</td><td>53.06</td><td>51.02</td></tr><tr><td>REWARDBENCH V2</td><td>OOD</td><td>-8.16</td><td>18.36</td><td>-8.23</td><td>-6.25</td></tr></table>

Table 1: Algorithm comparison. Values show percentage change in mean accuracy relative to the base policy, across MMLU-PRO and REWARDBENCH V2 under in-domain (ID) and out-of-domain (OOD) settings. Best gains are in bold, second-best are underlined. Accuracies which degrade from the base policy are marked in red.

correct answer. We use ULTRAFEEDBACK (Cui et al., 2024) as our training dataset. Specifically, its pre-processed and binarized version (Dong et al., 2024), which generates higher-quality reward models (Wang et al., 2024a; Xiong et al., 2024; Banerjee and Gopalan, 2024).

<table><tr><td rowspan="2">Subject</td><td rowspan="2">Base</td><td colspan="2">DPO</td><td colspan="2">AuxDPO</td><td colspan="2">IPO</td><td colspan="2">DPOP</td></tr><tr><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td></tr><tr><td>Overall</td><td>25.37</td><td>27.06</td><td>46.60</td><td>39.26</td><td>51.95</td><td>31.73</td><td>47.17</td><td>32.64</td><td>48.25</td></tr><tr><td>Biology</td><td>55.93</td><td>52.44</td><td>76.50</td><td>75.07</td><td>86.87</td><td>60.67</td><td>79.12</td><td>60.39</td><td>81.50</td></tr><tr><td>Business</td><td>13.18</td><td>20.03</td><td>35.67</td><td>29.96</td><td>43.49</td><td>24.21</td><td>32.56</td><td>24.08</td><td>37.96</td></tr><tr><td>Chemistry</td><td>10.42</td><td>16.25</td><td>37.12</td><td>26.78</td><td>40.86</td><td>21.64</td><td>34.56</td><td>21.73</td><td>35.86</td></tr><tr><td>Comp. Sc.</td><td>24.39</td><td>26.59</td><td>42.78</td><td>38.93</td><td>48.35</td><td>31.46</td><td>43.57</td><td>31.22</td><td>47.08</td></tr><tr><td>Economics</td><td>38.98</td><td>37.80</td><td>59.24</td><td>55.12</td><td>68.30</td><td>44.55</td><td>62.58</td><td>43.60</td><td>64.50</td></tr><tr><td>Engineering</td><td>10.22</td><td>17.75</td><td>46.06</td><td>32.18</td><td>48.60</td><td>26.01</td><td>41.46</td><td>28.48</td><td>43.82</td></tr><tr><td>Health</td><td>41.81</td><td>41.32</td><td>62.41</td><td>55.67</td><td>68.58</td><td>44.99</td><td>64.08</td><td>44.13</td><td>66.23</td></tr><tr><td>History</td><td>38.06</td><td>32.55</td><td>54.96</td><td>47.41</td><td>64.58</td><td>38.32</td><td>59.56</td><td>40.42</td><td>57.29</td></tr><tr><td>Law</td><td>27.25</td><td>23.61</td><td>42.20</td><td>34.50</td><td>48.01</td><td>27.88</td><td>45.25</td><td>30.43</td><td>42.31</td></tr><tr><td>Math</td><td>11.77</td><td>16.51</td><td>28.97</td><td>22.26</td><td>30.80</td><td>17.99</td><td>29.34</td><td>20.28</td><td>30.24</td></tr></table>

Table 2: Per-subject accuracies (top 10 subjects alphabetically) and overall win-rates across baseline (Llama3.1- 8B) and preference optimization methods. For each method, two settings are shown: OOD (cross-domain transfer) and ID (in-domain learning), along with the reported results. In each row, the best accuracy is shown in bold, and the second-best is underlined. Accuracies which degrade from the base policy are marked in red.

Evaluation and Methodology. We compare AuxDPO with DPO, IPO (Azar et al., 2023), and DPOP (Pal et al., 2024). Table 1 reports accuracies in terms of whether the logits of the chosen response were higher than those of the rejected response. While REWARDBENCH V2 provides chosen and rejected responses directly, we make MMLU-PRO into a preference dataset by filtering the correct answer as the chosen response and any incorrect response as the rejected response. We consider both in-distribution (ID) and out-of-distribution (OOD) evaluation settings. In the ID setup, each dataset is split 80/20 into training and evaluation subsets, ensuring IID comparisons. In the OOD setup, models are trained on cleaned ULTRAFEEDBACK and evaluated on the preference datasets. We report full finetuning results on the models. Table 2 presents overall and subject-wise accuracies on MMLU-PRO. Accuracy is measured by comparing the finetuned model’s generated answer with the correct answer provided in the dataset. We compare all methods on Llama3.1-8B under both OOD and ID settings. Across all three models, we see that AuxDPO outperforms other finetuning methods. Ablation studies, implementation, and dataset details are presented in Appendix B.2.

## ACKNOWLEDGMENTS

SRC would like to thank an early-career research grant from ANRF, India. DB would like to thank HP AI Labs for providing the necessary computational infrastructure to conduct the experiments.

## REFERENCES

Shun-ichi Amari. Information geometry and its applications, volume 194. Springer, 2016.

Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, and Rémi Munos. A general theoretical paradigm to understand learning from human preferences. arXiv preprint arXiv:2310.12036, 2023.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Debangshu Banerjee and Aditya Gopalan. Towards Reliable Alignment: Uncertainty-aware RLHF, 2024. URL https://arxiv.org/abs/2410.23726.

Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, Zhiyuan Liu, and Maosong Sun. UltraFeedback: Boosting Language Models with Scaled AI Feedback, 2024. URL https://arxiv.org/abs/2310.01377.

Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming Xiong, and Tong Zhang. RLHF Workflow: From Reward Modeling to Online RLHF, 2024. URL https://arxiv.org/abs/2405.07863.

Zhaolin Gao, Jonathan Chang, Wenhao Zhan, Owen Oertell, Gokul Swamy, Kianté Brantley, Thorsten Joachims, Drew Bagnell, Jason D Lee, and Wen Sun. Rebel: Reinforcement learning via regressing relative rewards. Advances in Neural Information Processing Systems, 37:52354–52400, 2024.

Chengtao Jian, Kai Yang, Ye Ouyang, and Xiaozhou Ye. Stable preference optimization for LLMs: A bilevel approach beyond direct preference optimization. arXiv preprint arXiv:2507.07723, 2025.

Sham M Kakade. A natural policy gradient. Advances in neural information processing systems, 14, 2001.

Saumya Malik, Valentina Pyatkin, Sander Land, Jacob Morrison, Noah A Smith, Hannaneh Hajishirzi, and Nathan Lambert. Rewardbench 2: Advancing reward model evaluation. arXiv preprint arXiv:2506.01937, 2025.

Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a reference free reward. Advances in Neural Information Processing Systems, 37:124198–124235, 2024.

Arka Pal, Deep Karkhanis, Samuel Dooley, Manley Roberts, Siddartha Naidu, and Colin White. Smaug: Fixing failure modes of preference optimisation with dpo-positive. arXiv preprint arXiv:2402.13228, 2024.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290, 2023.

Noam Razin, Sadhika Malladi, Adithya Bhaskar, Danqi Chen, Sanjeev Arora, and Boris Hanin. Unintentional unalignment: Likelihood displacement in direct preference optimization. arXiv preprint arXiv:2410.08847, 2024.

Ruizhe Shi, Minhak Song, Runlong Zhou, Zihan Zhang, Maryam Fazel, and Simon S Du. Understanding the performance gap in preference learning: A dichotomy of RLHF and DPO. arXiv preprint arXiv:2505.19770, 2025.

Yuda Song, Gokul Swamy, Aarti Singh, J Bagnell, and Wen Sun. The importance of online data: Understanding preference fine-tuning via coverage. Advances in Neural Information Processing Systems, 37:12243–12270, 2024.

Gokul Swamy, Sanjiban Choudhury, Wen Sun, Zhiwei Steven Wu, and J Andrew Bagnell. All roads lead to likelihood: The value of reinforcement learning in fine-tuning. arXiv preprint arXiv:2503.01067, 2025.

Fahim Tajwar, Anikait Singh, Archit Sharma, Rafael Rafailov, Jeff Schneider, Tengyang Xie, Stefano Ermon, Chelsea Finn, and Aviral Kumar. Preference fine-tuning of llms should leverage suboptimal, on-policy data. arXiv preprint arXiv:2404.14367, 2024.

Haoxiang Wang, Wei Xiong, Tengyang Xie, Han Zhao, and Tong Zhang. Interpretable preferences via multi-objective reward modeling and mixture-of-experts. In The 2024 Conference on Empirical Methods in Natural Language Processing, 2024a.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multitask language understanding benchmark. Advances in Neural Information Processing Systems, 37: 95266–95290, 2024b.

Halbert White. Maximum likelihood estimation of misspecified models. Econometrica: Journal of the econometric society, pages 1–25, 1982.

Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, and Tong Zhang. Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint. ICML, 2024.

Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan, Lingfeng Shen, Benjamin Van Durme, Kenton Murray, and Young Jin Kim. Contrastive preference optimization: Pushing the boundaries of llm performance in machine translation. arXiv preprint arXiv:2401.08417, 2024a.

Shusheng Xu, Wei Fu, Jiaxuan Gao, Wenjie Ye, Weilin Liu, Zhiyu Mei, Guangju Wang, Chao Yu, and Yi Wu. Is dpo superior to ppo for llm alignment? a comprehensive study. arXiv preprint arXiv:2404.10719, 2024b.

Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. In arXiv preprint arXiv:1909.08593, 2019.