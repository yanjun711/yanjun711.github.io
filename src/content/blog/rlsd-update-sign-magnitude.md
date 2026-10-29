---
title: "Separating Update Sign and Magnitude in RLSD"
description: "A verifier chooses whether a response is reinforced; privileged context redistributes credit among its tokens."
date: 2026-05-09
tags: [LLM, Reinforcement learning]
draft: false
toc: true
---

A verifier can tell us that a generated answer is wrong without telling us which token led it astray. Giving every token the same negative advantage is simple, but it also penalizes useful steps that happened to appear in an unsuccessful response.

[Self-Distilled RLVR](https://arxiv.org/abs/2604.03128v1) proposes RLSD, which uses privileged information to redistribute token-level credit: **the group-relative reward signal determines each token advantage's sign; the privileged context changes its magnitude**. Following that coefficient through the update shows how RLSD reinforces useful tokens and concentrates negative credit on less supported ones.

## One response, two probability estimates

Let $x$ be a question and $y=(y_1,\ldots,y_T)$ a response sampled without privileged information. At training time, a reference $r$ provides additional context. The same model evaluates each sampled token in two ways:

$$
P_S(y_t)=\pi_\theta(y_t\mid x,y_{<t}),
$$

$$
P_T(y_t)=\pi_\theta(y_t\mid x,r,y_{<t}).
$$

The labels student and teacher describe the two contexts, rather than two independently trained networks. These formulas follow the paper's shared-parameter description. Its implementation instead freezes a teacher snapshot between synchronizations every ten training steps. Both contexts are evaluated on the **same student-sampled prefix**. The teacher does not replace the rollout with a reference-generated response.

RLSD measures how much privileged context changes the probability of the token that was actually sampled:

$$
\Delta_t=\operatorname{sg}\left(
\log P_T(y_t)-\log P_S(y_t)
\right).
$$

The stop-gradient operator $\operatorname{sg}$ makes this a fixed weighting signal within the policy update. A positive value means the reference makes the sampled token more likely. A negative value means it makes the token less likely.

Reference support measures a token's likelihood under additional context. The verifier supplies a separate sequence-level judgment, allowing RLSD to retain useful steps within an otherwise failed response.

## Reverse the evidence ratio for a failed response

Following the group-relative construction in [DeepSeekMath](https://arxiv.org/abs/2402.03300v1), let the verifier rewards for a group of sampled responses have mean $\mu_G$ and nonzero standard deviation $\sigma_G$. A response receives the group-relative advantage

$$
A=\frac{R-\mu_G}{\sigma_G}.
$$

A positive advantage means the response performed better than its group baseline; a negative advantage means worse. It is this **advantage**, rather than the raw reward in isolation, that determines the sign in the mechanism.

RLSD defines the evidence weight

$$
w_t=\exp\bigl(\operatorname{sign}(A)\Delta_t\bigr)
=\left(\frac{P_T(y_t)}{P_S(y_t)}\right)^{\operatorname{sign}(A)}.
$$

When $A>0$, a teacher-supported token receives more positive credit. When $A<0$, the ratio is inverted: a teacher-disfavored token receives stronger negative credit, while a teacher-supported token receives less punishment.

Using the same ratio in both cases would miss this distinction. Multiplying a negative advantage by a large teacher-support ratio would punish the teacher-supported token more strongly. The exponent changes which tokens receive the largest magnitudes when the verifier says the response should be discouraged.

Algorithm 1 combines clipping with a mixing coefficient. The paper motivates bounded reweighting through the clipping design in [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347v1), although the evidence ratio here compares two contexts rather than a current and old policy:

$$
c_t=\operatorname{clip}(w_t,1-\epsilon_w,1+\epsilon_w),
$$

$$
\hat A_t=A\bigl[(1-\lambda)+\lambda c_t\bigr],
\qquad 0\leq\lambda\leq1.
$$

For $0<\epsilon_w<1$, the multiplier is strictly positive. More precisely,

$$
1-\lambda\epsilon_w
\leq (1-\lambda)+\lambda c_t
\leq1+\lambda\epsilon_w.
$$

Every nonzero $\hat A_t$ therefore retains the sign of $A$. If $A=0$, all token advantages are zero. Setting $\lambda=0$ recovers uniform sequence-level advantages; increasing it permits more redistribution.


## Three tokens are enough to see the effect

Take a group with rewards $[1,0]$. Using the population standard deviation gives advantages $+1$ and $-1$. Suppose three sampled token positions have evidence ratios

$$
\frac{P_T}{P_S}=[2,\;0.5,\;1].
$$

These can arise from student probabilities $[0.2,0.4,0.3]$ and teacher probabilities $[0.4,0.2,0.3]$. Each entry refers to a different conditional distribution at a different position, so these three numbers need not sum to one.

Choose $\epsilon_w=0.2$ and $\lambda=0.5$. For the successful response, the raw weights are $[2,0.5,1]$. Clipping gives $[1.2,0.8,1]$ and mixing with one gives $[1.1,0.9,1]$. Hence

$$
\hat A^{(+)}=[1.1,\;0.9,\;1].
$$

For the failed response, invert the ratios first. The weights become $[0.5,2,1]$, the clipped weights become $[0.8,1.2,1]$, and

$$
\hat A^{(-)}=[-0.9,\;-1.1,\;-1].
$$

The first token receives stronger reinforcement when the response succeeds and weaker punishment when it fails. The second receives the opposite treatment. The third receives the unchanged sequence advantage because the reference did not alter its probability.

Clipping means a ratio of twenty would produce the same coefficient as a ratio of two in this example. It bounds influence by discarding some distinctions among extreme evidence ratios. The probabilities and rewards here are constructed to demonstrate that tradeoff.

## Where the gradient still comes from

The paper's equation 17 describes the policy-gradient interpretation:

$$
\Delta\theta\propto
\mathbb E_{y\sim P_S}
\left[\sum_t\hat A_t\nabla_\theta\log P_S(y_t)\right].
$$

The detached advantage multiplies a differentiable policy score. For a minimal local illustration, hold the sampled tokens and advantages fixed and minimize

$$
L_{\mathrm{local}}(\theta)
=-\sum_t\operatorname{sg}(\hat A_t)\log\pi_\theta(y_t\mid x,y_{<t}).
$$

This local score-function surrogate isolates the detached coefficient. In GRPO, the current-versus-old-policy ratio additionally accounts for policy changes since rollout. RLSD's teacher-versus-student **evidence ratio** instead measures the effect of privileged context on the sampled token.

For one categorical distribution with logits $z$, probabilities $p$, and sampled class $k$, the negative loss gradient is

$$
-\nabla_zL_{\mathrm{local}}
=\hat A_t(e_k-p).
$$

The coefficient controls whether that sampled token is locally encouraged or discouraged. The parameter update sums these contributions across positions. Reweighting their magnitudes changes the direction of that sum even when every coefficient retains its sign.

[The runnable NumPy example](/code/rlsd-update-sign-magnitude.py) computes both coefficient lists, checks their bounds, and verifies the detached local loss gradient by finite differences. It keeps the weights fixed during that check; recomputing them at every perturbation would test a different objective.

The implementation above uses the score-function gradient in equation 17 and the mixed coefficient in Algorithm 1. The differentiable log-policy factor carries the learning signal; the detached advantage sets its weight. This makes the gradient path explicit where equation 16 and the algorithm's final update line leave it implicit.

## Reference support and reward quality

RLSD uses reference-conditioned likelihood as a credit-assignment heuristic. That likelihood can reflect a useful reasoning step, a familiar phrase, style, or direct reference overlap. The resulting weights express contextual support for the sampled tokens.

The verifier determines the sequence-level signal. A spurious positive reward reinforces the response, with RLSD redistributing that reinforcement according to reference support. Reward quality therefore shapes the update before token reweighting begins.

Groups with identical rewards have zero variance. An implementation can skip those groups or guard the standardization denominator; the worked example uses rewards $[1,0]$ to expose the sign-and-magnitude calculation directly.

The positive, bounded multipliers give RLSD its defining property: privileged context redistributes token-advantage magnitudes while the group-relative reward signal controls their signs.

## References

1. [Self-Distilled RLVR](https://arxiv.org/abs/2604.03128v1). 2026.
2. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300v1). 2024.
3. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347v1). 2017.
