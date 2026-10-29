---
title: "Self Forcing: Training on Generated Video Prefixes"
description: "Why generated prefixes change a video model's training problem, and what remains when gradients stop at the KV cache."
date: 2025-07-12
tags: [Video generation, Optimization]
draft: false
toc: true
---

An autoregressive video generator produces a frame, adds it to its context, and uses that context to produce the next frame. A small error can therefore become an input to every later prediction. Training on clean video prefixes does not expose the model to this particular source of error: the preceding frames are supplied by the dataset, even when the model would have produced something different.

[Self Forcing](https://arxiv.org/abs/2506.08009v1) changes where those prefixes come from. During training, the model conditions on frames that it generated itself. This post focuses on that choice and the gradient truncation that makes it affordable. The two changes address different questions: **which states the model sees**, and **which dependencies receive gradients**.

## The problem with a correct previous frame

Write a video as $x^{1:N}$, where $x^i$ is a frame or a generated chunk, as in [CausVid](https://arxiv.org/abs/2412.07772v2). An autoregressive model factorizes its distribution as

$$
p_\theta(x^{1:N})
=\prod_{i=1}^{N}p_\theta(x^i\mid x^{<i}).
$$

The conditional distribution alone does not specify a training procedure. We can train it with a prefix from a real video, or with a prefix sampled from the model. These produce different distributions of conditioning inputs.

For example, suppose a generated hand is slightly displaced from the object it is holding. Given the correct previous frame, the model never needs to respond to that displacement. Given its own previous frame, it does. The mismatch concerns the joint trajectory, even if each individual denoising prediction is reasonable under clean context.

Adding noise to real prefixes can expose a model to corruptions, but those corruptions need not resemble its own errors. Self Forcing instead rolls out the autoregressive generator inside training. The resulting frames are both outputs to be evaluated and context for later frames.

This makes a video-level distribution objective useful. If a loss evaluates a sample produced through the same causal dependencies used at inference, it can detect problems that only appear when predictions are fed back. It does not need paired ground-truth continuations for each particular generated prefix.

## A generated prefix, with a stopped gradient

Each conditional in Self Forcing is a few-step diffusion generator. Let

$$
0=t_0<t_1<\cdots<t_T
$$

be its noise levels, ordered from clean to noisy. Starting with Gaussian noise at $t_T$, the model predicts a clean frame. That prediction is re-noised to the next lower noise level and passed to another denoising call.

Unrolling this process for many frames would retain a graph through both the denoising steps and the temporal context. Self Forcing shortens that graph in two places.

First, Algorithm 1 samples an index $s$ uniformly from $\{1,\ldots,T\}$ for a training sequence. For each frame, it runs denoising calls at $t_T,t_{T-1},\ldots,t_s$. Calls before $t_s$ run without gradients. At $t_s$, the model's predicted **clean** output becomes that frame's output, and this last call is differentiated. The frame does not continue through the remaining lower-noise calls in that update.

Second, the model computes the frame's cached keys and values without gradients. Later frames can read those values, but their losses cannot differentiate through the cache into the earlier frame's computation.

<figure class="figure-mechanism">
  <picture>
    <source media="(max-width: 620px)" srcset="/images/blog/self-forcing-generated-prefixes/prefix-drift-mobile.svg" width="360" height="305" />
    <img src="/images/blog/self-forcing-generated-prefixes/prefix-drift.svg" alt="Previous generated frames are encoded into a detached KV prefix. The selected denoising call conditions on that prefix to produce the current frame, with gradients stopped before the cached context." width="620" height="260" />
  </picture>
  <figcaption>Figure 1. For frame i, earlier generated frames supply a detached KV prefix. The selected denoising call receives gradients; the conditioning cache does not. Earlier outputs retain their own loss paths.</figcaption>
</figure>

A schematic version is:

```python
s = sample_exit_index()  # One index for this sequence.
cache = []
outputs = []

for frame in range(num_frames):
    state = sample_noise()
    for j in range(T, s - 1, -1):
        if j > s:
            with no_grad():
                clean = denoise(state, t[j], cache)
                state = renoise(clean, t[j - 1])
        else:
            clean = denoise(state, t[j], cache)
    outputs.append(clean)  # Retain this call's gradient.
    with no_grad():
        cache.append(encode_clean_frame(clean, cache))

distribution_loss(outputs).backward()
```

In this pseudocode, `cache` represents the model's causal context, and `encode_clean_frame` computes the frame's KV representations.

Detachment preserves the generated frame values as forward context while cutting their backward paths. Randomizing the exit distributes supervision across denoising positions over successive updates. Each update differentiates the selected call with its generated prefix held fixed.

Early exit also shortens the denoising trajectory relative to full $T$-step inference. Training aligns the source of causal prefixes with generation while using shorter denoising paths and detached caches to control memory.

### What evaluates the generated video?

Self Forcing combines its rollout with distribution-matching objectives. For the version based on [Distribution Matching Distillation (DMD)](https://arxiv.org/abs/2311.18828v1), a useful schematic expression is

$$
\mathbb{E}_{\tau}
\operatorname{KL}\!\left(
p_{\theta,\tau}\,\|\,p_{\mathrm{data},\tau}
\right).
$$

Here $\tau$ is a noise level used by the distribution loss, and $p_{\theta,\tau}$ and $p_{\mathrm{data},\tau}$ are generated and real **video** distributions after applying that noising process. DMD estimates their score functions to construct the training update.

The paper also considers SiD and adversarial objectives. Generated-prefix training is therefore not tied to a single choice of divergence. Its role is to supply video samples whose temporal dependencies include the model's own predictions.

## A three-step example

A scalar recurrence makes the prefix issue visible without a diffusion model. Let the desired state be $1$ at every step and define

$$
h_i=ah_{i-1}+b,
\qquad h_0=1,
\qquad a=0.9,\quad b=0.15.
$$

If training always supplies the correct previous state, every prediction is $0.9\times1+0.15=1.05$. During a generated rollout, the states are instead

$$
h_1=1.05,\qquad
h_2=1.095,\qquad
h_3=1.1355.
$$

Now use $L=\frac12\sum_{i=1}^{3}(h_i-1)^2$. Full differentiation with respect to $b$ follows the recurrence

$$
\frac{\partial h_i}{\partial b}
=a\frac{\partial h_{i-1}}{\partial b}+1.
$$

The sensitivities are $1$, $1.9$, and $2.71$, so

$$
\frac{dL}{db}
=0.05+0.095(1.9)+0.1355(2.71)
=0.597705.
$$

If each preceding state is detached, the local derivative of $h_i$ with respect to $b$ is just $1$. The same forward states then produce the gradient

$$
\widetilde{\frac{dL}{db}}
=0.05+0.095+0.1355
=0.2805.
$$

Both computations see the accumulated error. Only the first assigns credit through earlier transitions. The scalar example separates the forward rollout from backward credit assignment in a setting where both gradients can be calculated by hand.

[Download the checked example](/code/self-forcing-generated-prefixes.py). It verifies the full derivative with finite differences, then verifies the detached derivative while holding the generated prefixes fixed. Re-running the complete rollout for the second check would incorrectly measure the full derivative again.

## Generated context and temporal credit

Generated-prefix training addresses models whose outputs become later conditioning inputs. Rolling out those dependencies during training exposes the distribution-matching objective to the errors the model feeds back into itself.

Detached caches reduce memory by omitting gradients through earlier context computations. The video-level loss evaluates the whole sequence, while the update uses the retained local parameter derivatives. Forward exposure to accumulated errors and backward temporal credit are separate parts of the training design.

The paper reports deterioration far beyond the training horizon, connecting long-video quality to rollout length, teacher quality, and retained context. Within training, generated prefixes let each new prediction encounter the consequences of the model's own earlier outputs.

## References

1. Xun Huang et al. [Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/html/2506.08009v1). 2025.
2. Tianwei Yin et al. [From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](https://arxiv.org/abs/2412.07772v2). CVPR, 2025.
3. Tianwei Yin et al. [One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828v1). CVPR, 2024; first preprint, 2023.
