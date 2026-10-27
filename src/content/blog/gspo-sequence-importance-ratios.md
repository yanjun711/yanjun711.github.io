---
title: "Sequence-Level Importance Ratios in GSPO"
description: "Deriving GSPO's length-normalized ratio, its shared response-level weight, and the limits of sequence-level clipping."
date: 2025-08-09
tags: [LLM, Reinforcement learning]
draft: false
toc: true
---

A language model receives one reward for a complete answer, but an optimizer can change the probabilities of hundreds of tokens that produced it. If one token becomes much more likely and another becomes less likely, should clipping treat them independently or judge the answer as one sample?

[Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071v1), or GSPO, makes that decision at the response level. Its central quantity is the geometric mean of the token probability ratios. Understanding that mean explains both why the update has one shared weight per answer and what the clipping rule can miss.

## From token probabilities to one response ratio

Fix a prompt $x$. A frozen rollout policy $\pi_{\mathrm{old}}$ samples a response $y_i=(y_{i,1},\ldots,y_{i,L_i})$ of length $L_i$. During optimization, $\pi_\theta$ evaluates the same sampled tokens and prefixes. Define

$$
w_{i,t}
=\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}
{\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t})}.
$$

Autoregressive factorization gives the full response likelihood ratio:

$$
R_i
=\frac{\pi_\theta(y_i\mid x)}
{\pi_{\mathrm{old}}(y_i\mid x)}
=\prod_{t=1}^{L_i}w_{i,t}.
$$

$R_i$ is the ordinary joint importance-sampling ratio, assuming the old policy has support wherever the new-policy expectation needs it. It can grow or shrink sharply with length. If every token ratio is $1.02$, the joint ratio is about $1.49$ for 20 tokens and $7.24$ for 100 tokens, despite the identical per-token change.

GSPO therefore uses

$$
s_i=R_i^{1/L_i}
=\exp\!\left[
\frac{1}{L_i}\sum_{t=1}^{L_i}
\left(\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})
-\log\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t})\right)
\right].
$$

This is the geometric mean of the token ratios. In the repeated-$1.02$ example, $s_i=1.02$ for either length. Compute it from mean log-probability differences rather than multiplying a long list of probabilities.

The normalization is a consequential change of objective. $s_i$ is a length-normalized surrogate, **not** the exact density ratio $R_i$. Substituting it into an expectation does not preserve the usual unbiased importance-sampling identity. Its purpose is to provide a practical optimization weight on a more comparable scale across responses.

## Clipping one answer at a time

For a group of $G$ answers to the same prompt, let $r_i$ be the reward. As in GRPO, introduced in [DeepSeekMath](https://arxiv.org/abs/2402.03300v1), GSPO normalizes rewards within the group:

$$
A_i=\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)}.
$$

Rewards, advantages, and rollout-policy log probabilities are held fixed during the policy update. If all rewards are equal, the denominator needs a numerical guard or the group needs explicit handling; the equation alone is undefined in that case.

The clipping construction follows [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347v1), with GSPO applying it to the length-normalized response ratio. The sampled objective to maximize is

$$
J(\theta)=\frac{1}{G}\sum_{i=1}^{G}
\min\!\left(
s_i A_i,
\operatorname{clip}(s_i,1-\epsilon,1+\epsilon)A_i
\right).
$$

The sign of $A_i$ matters. For a positive advantage, increasing $s_i$ beyond $1+\epsilon$ reaches a flat branch. Decreasing it below $1-\epsilon$ does not: the minimum still selects $s_iA_i$. For a negative advantage, the flat region is instead below $1-\epsilon$.

Thus an out-of-range response is not automatically removed from the gradient. Clipping suppresses an update that would move too far in the reward-favored direction. The opposite direction remains penalized.

<figure>
  <picture>
    <source media="(max-width: 620px)" srcset="/images/blog/gspo-sequence-importance-ratios/clipping-mobile.svg" width="259" height="259" />
    <img src="/images/blog/gspo-sequence-importance-ratios/clipping.svg" alt="GSPO's clipped objective is flat above the upper ratio threshold for positive advantage, and below the lower threshold for negative advantage." width="446" height="180" />
  </picture>
  <figcaption>Figure 1. Clipping with advantages +1 and −1 and illustrative ε = 0.1. Dashed curves show the unclipped objective; the flat branches stop rewarding ratio changes beyond the relevant threshold.</figcaption>
</figure>

Inside an active, unclipped branch, differentiation gives

$$
\nabla_\theta(s_iA_i)
=\frac{s_iA_i}{L_i}
\sum_{t=1}^{L_i}
\nabla_\theta\log\pi_\theta(y_{i,t}\mid x,y_{i,<t}).
$$

Every token score gradient in one answer receives the same scalar multiplier $s_iA_i/L_i$. Their vector contributions can still differ, because the token gradients themselves differ. Shared weighting does not mean that every word has equal causal responsibility for the reward.

## A two-token calculation

Consider a response with two token ratios

$$
w_1=1.44,\qquad w_2=0.81.
$$

These are valid probability changes, for example from $0.4$ to $0.576$ for the first sampled token and from $0.5$ to $0.405$ for the second. Probabilities are conditional on the sampled prefixes; the remaining vocabulary probabilities adjust to keep each distribution normalized.

The joint ratio and GSPO ratio are

$$
R=1.44\times0.81=1.1664,
\qquad
s=\sqrt{1.1664}=1.08.
$$

Let $A=1$ and use the toy clipping interval $[0.9,1.1]$. GSPO is still on its active branch, because $1.08<1.1$. A token-wise clipped surrogate would flatten the first token's positive contribution, since $1.44>1.1$, while leaving the second active. The two rules therefore produce different gradients even though they use the same answer and reward.

To check the GSPO derivative directly, introduce coordinates $\delta_t=\log w_t$. Then

$$
s=\exp\!\left(\frac{\delta_1+\delta_2}{2}\right),
\qquad
\frac{\partial(sA)}{\partial\delta_1}
=\frac{\partial(sA)}{\partial\delta_2}
=\frac{sA}{2}=0.54.
$$

Here the $\delta_t$ are independent mathematical coordinates for checking the surrogate. A real model maps shared parameters to normalized token distributions, so its parameter gradient additionally contains the log-probability Jacobians.

The essential calculation takes only a few lines:

```python
from math import exp

def contribution(log_ratios, advantage, epsilon):
    s = exp(sum(log_ratios) / len(log_ratios))
    clipped_s = max(1 - epsilon, min(1 + epsilon, s))
    return min(s * advantage, clipped_s * advantage)
```

[Download the numerical check](/code/gspo-sequence-importance-ratios.py). It verifies the $0.54$ derivatives with finite differences and checks the flat branches for positive and negative advantages. The actual training loss usually negates this maximization objective before calling a gradient-descent optimizer.

## What the aggregate ratio hides

An answer can have $s=1$ even when individual token changes are large. With $w_1=4$ and $w_2=0.25$,

$$
s=\sqrt{4\times0.25}=1.
$$

The opposing log changes cancel. Response-level clipping sees an unchanged average log likelihood, so it does not bound either individual ratio. Nor does it impose a hard bound on the policy's KL divergence over all possible answers.

Length normalization has a related tradeoff. It makes a repeated per-token change comparable across lengths, but it also dilutes an isolated token's log-ratio change in a long response. Whether that is useful depends on the behavior being rewarded and the optimization regime.

The group advantage adds another limitation: the same answer can receive different normalized weights depending on the other answers sampled for its prompt. GSPO's ratio does not improve a poor reward function, supply token-level credit, or create a learning signal in a group with no reward variation.

These boundaries do not make the response-level choice arbitrary. A scalar completion reward and a shared response update are a coherent pairing. The precise mechanism is that GSPO averages **log probability changes**, exponentiates that average, and clips the resulting response contribution. Its empirical behavior should be assessed with that surrogate in mind, rather than treating it as exact joint importance sampling.

## References

1. Chujie Zheng et al. [Group Sequence Policy Optimization](https://arxiv.org/html/2507.18071v1). 2025.
2. Zhihong Shao et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300v1). 2024.
3. John Schulman et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347v1). 2017.
