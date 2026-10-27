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

This is structural pseudocode, not a runnable diffusion implementation. In particular, `cache` represents the model's causal context, and `encode_clean_frame` includes the operations needed to obtain its KV representations.

The forward context remains generated context after detachment. Stopping a derivative does not replace a frame with ground truth. However, this is also not full backpropagation through the generated trajectory. Randomizing the exit allows different denoising positions to receive supervision across updates; it does not restore the omitted temporal derivatives.

There is a further qualification: an early-exit training sample need not have exactly the same distribution as a sample from all $T$ inference steps. The central alignment is the use of generated causal prefixes, with a deliberate approximation in denoising depth and gradient flow.

### What evaluates the generated video?

Self Forcing combines its rollout with distribution-matching objectives. For the version based on [Distribution Matching Distillation (DMD)](https://arxiv.org/abs/2311.18828v1), a useful schematic expression is

$$
\mathbb{E}_{\tau}
\operatorname{KL}\!\left(
p_{\theta,\tau}\,\|\,p_{\mathrm{data},\tau}
\right).
$$

Here $\tau$ is a noise level used by the distribution loss, and $p_{\theta,\tau}$ and $p_{\mathrm{data},\tau}$ are generated and real **video** distributions after applying that noising process. They are distributions, not single paired videos. In practice, DMD uses score estimates rather than directly evaluating these densities.

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

<figure>
  <picture>
    <source media="(max-width: 620px)" srcset="/images/blog/self-forcing-generated-prefixes/prefix-drift-mobile.svg" width="346" height="307" />
    <img src="/images/blog/self-forcing-generated-prefixes/prefix-drift.svg" alt="A scalar toy recurrence stays at 1.05 with clean prefixes, but drifts to 1.1355 when conditioned on its own outputs." width="595" height="307" />
  </picture>
  <figcaption>Figure 1. A toy recurrence, not a video-model measurement. The one-step error is small under clean prefixes; feeding predictions back changes the later inputs.</figcaption>
</figure>

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

Both computations see the accumulated error. Only the first assigns credit through earlier transitions. This example is not an assertion that Self Forcing uses squared scalar errors; it isolates the difference between forward rollout and backward credit assignment.

[Download the checked example](/code/self-forcing-generated-prefixes.py). It verifies the full derivative with finite differences, then verifies the detached derivative while holding the generated prefixes fixed. Re-running the complete rollout for the second check would incorrectly measure the full derivative again.

## Where this helps, and what it leaves open

The mechanism is relevant when a generator consumes its own outputs and its errors change later conditioning inputs. A model that generates an entire video jointly does not have this same prefix mismatch in this form. Likewise, replacing a real prefix with a generated one is useful only if the training objective can provide meaningful feedback on the resulting sample.

Detached caches reduce memory, but remove a path for learning how an early decision affects a much later frame. They do not make the truncated gradient an unbiased estimate of full temporal backpropagation. A video-level loss can still evaluate the whole sequence while receiving only these restricted parameter derivatives.

Finally, training on generated prefixes is not a guarantee of unlimited-horizon stability. The paper reports deterioration far beyond the training horizon. The length of training rollouts, the quality of the distribution-matching teacher, and the retained context still constrain what the model can learn. The narrow result is that the training inputs now include the consequences of the generator's own earlier predictions.

## References

1. Xun Huang et al. [Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/html/2506.08009v1). 2025.
2. Tianwei Yin et al. [From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](https://arxiv.org/abs/2412.07772v2). CVPR, 2025.
3. Tianwei Yin et al. [One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828v1). CVPR, 2024; first preprint, 2023.
