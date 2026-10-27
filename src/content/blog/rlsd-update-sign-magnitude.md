---
title: "Separating Update Sign and Magnitude in RLSD"
description: "A verifier chooses whether a response is reinforced; privileged context redistributes credit among its tokens."
date: 2026-05-09
tags: [LLM, Reinforcement learning]
draft: false
toc: true
---

A verifier can tell us that a generated answer is wrong without telling us which token led it astray. Giving every token the same negative advantage is simple, but it also penalizes useful steps that happened to appear in an unsuccessful response.

[Self-Distilled RLVR](https://arxiv.org/abs/2604.03128v1) proposes RLSD, which uses privileged information to redistribute this token-level credit. Its central constraint is small enough to study directly: **the reward determines the sign of each token's advantage; the privileged context changes its magnitude**. This note follows that coefficient through the update and works out what the sign constraint actually guarantees.

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

That observation alone is not a correctness label. A reference may favor a token inside an otherwise failed response, and a plausible token can still belong to a bad overall strategy. The verifier supplies the sequence-level judgment.

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

<figure>
  <picture>
    <source media="(max-width: 620px)" srcset="/images/blog/rlsd-update-sign-magnitude/credit-mobile.svg" width="346" height="451" />
    <img src="/images/blog/rlsd-update-sign-magnitude/credit.svg" alt="Clipped token advantages versus teacher-student evidence ratio for positive and negative response advantages. Magnitudes vary between 0.9 and 1.1 while signs stay fixed." loading="lazy" width="595" height="307" />
  </picture>
  <figcaption>Figure 1. The coefficient as a function of evidence ratio, with clipping width 0.2 and mixing coefficient 0.5. Markers are the ratios used in the example. The horizontal reference is the unmodified response advantage.</figcaption>
</figure>

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

This is a score-function surrogate for the example, not a complete GRPO training objective. A production GRPO update also needs its current-versus-old-policy treatment. That policy ratio and RLSD's teacher-versus-student **evidence ratio** have different roles and should not share an implementation variable merely because both are probability ratios.

For one categorical distribution with logits $z$, probabilities $p$, and sampled class $k$, the negative loss gradient is

$$
-\nabla_zL_{\mathrm{local}}
=\hat A_t(e_k-p).
$$

The coefficient controls whether that sampled token is locally encouraged or discouraged. In a shared neural network, however, gradients from different positions interact. Preserving all coefficient signs does **not** preserve the direction of their summed parameter-gradient vector.

[The runnable NumPy example](/code/rlsd-update-sign-magnitude.py) computes both coefficient lists, checks their bounds, and verifies the detached local loss gradient by finite differences. It keeps the weights fixed during that check; recomputing them at every perturbation would test a different objective.

A source-reading detail matters here: equation 16 and Algorithm 1's final update line do not explicitly provide the differentiable policy factor needed for a detached-weight implementation. The calculation above follows the explicit score-function gradient in equation 17 and the mixed coefficient in Algorithm 1. Taking a gradient of a batch sum of detached advantages alone would produce no policy-learning signal.

## What the constraint leaves unresolved

The sign constraint gives a useful separation of responsibilities within each sampled response. It does not prove that the teacher identified the causally important reasoning step. Token likelihood can reflect style, a familiar phrase, or direct reference overlap. Reweighting such a token remains a heuristic credit assignment even when its algebra is well behaved.

The verifier can also be wrong or incomplete. A response with a spurious positive reward still receives positive token advantages. Positive multipliers preserve the supplied judgment; they cannot repair it.

If every reward in a group is identical, the displayed standardization has zero variance. An implementation needs an explicit numerical convention for that case. The example uses a nondegenerate group so that this issue does not disappear inside an arbitrary epsilon.

Finally, bounded coefficients do not establish general immunity to reference leakage, convergence, or evaluation shortcuts. They establish the narrower property proved above: privileged evidence redistributes the magnitudes of the token advantages while their signs remain controlled by the group-relative reward signal.

## References

1. [Self-Distilled RLVR](https://arxiv.org/abs/2604.03128v1). 2026.
2. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300v1). 2024.
3. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347v1). 2017.
