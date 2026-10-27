---
title: "Dimension-Dependent Noise Scheduling in RAE"
description: "Why a high-dimensional representation can need more corruption, and how RAE shifts the noise-to-signal ratio."
date: 2025-11-08
tags: [Image generation, Diffusion, Representation learning]
draft: false
toc: true
---

Suppose a diffusion transformer trains successfully on a compact image latent. We replace the encoder with a pretrained visual representation, keeping the same number of spatial tokens but increasing the channels per token. Can we keep sampling noise levels in exactly the same way?

The RAE paper, *Diffusion Transformers with Representation Autoencoders* [\[1\]](https://arxiv.org/abs/2510.11690v1), identifies a problem with this substitution: a noise level that usefully corrupts a small latent can leave too much recoverable information in a larger representation. Section 4.2 adjusts the schedule using the **total latent dimension**, including channels. The mechanism is a one-dimensional change of variables. Understanding it does not require treating the whole autoencoder or transformer architecture as a single intervention.

## What does a timestep mean?

Use the paper's interpolation convention:

$$
x_t=(1-t)x+t\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I),
\qquad 0\leq t\leq1.
$$

Here $x$ is a clean latent, $\varepsilon$ is independent Gaussian noise, and $x_t$ is the corrupted input. The endpoint $t=0$ is clean; $t=1$ is pure noise. This direction matters when interpreting a schedule shift. Increasing $t$ increases corruption.

The coefficient ratio of noise to signal is

$$
r(t)=\frac{t}{1-t}.
$$

If signal and noise have the same variance per coordinate, their variance ratio is $r(t)^2$. The ratio of coefficients and the ratio of variances are different quantities. The schedule below scales the former.

Why could dimension matter? Gaussian noise is sampled independently across coordinates, while useful structure can be redundant across spatial locations and channels. A denoiser can combine several noisy clues about the same content. More coordinates can therefore make information easier to recover at the same nominal $t$. Counting spatial tokens alone misses the extra clues carried by a wider representation. Earlier studies of noise scheduling [\[2\]](https://arxiv.org/abs/2301.10972v1) and high-resolution diffusion [\[3\]](https://arxiv.org/abs/2301.11093v1) discuss the related need to adjust corruption as image resolution changes.

This is an argument about representation structure. Ambient dimension by itself does not universally improve signal-to-noise ratio: appending unrelated, equally scaled signal and noise coordinates need not make any particular coordinate easier to recover.

## A shift in noise-to-signal odds

RAE adopts the schedule transformation used by Esser et al. [\[4\]](https://arxiv.org/abs/2403.03206v1) and extends the dimension count to include latent channels. Let $n$ be a reference dimension and $m$ the target dimension. RAE uses $n=4096$ and sets $m$ to the number of tokens multiplied by the number of channels in each token. Define

$$
\alpha=\sqrt{\frac{m}{n}}.
$$

A sampled reference timestep $t_n$ becomes

$$
\boxed{\quad
t_m=\frac{\alpha t_n}{1+(\alpha-1)t_n}.
\quad}
$$

The form is easier to interpret after taking the ratio to its complement:

$$
\frac{t_m}{1-t_m}
=\alpha\frac{t_n}{1-t_n}.
$$

The transformation multiplies noise-to-signal odds by $\alpha$. Equivalently, for interior timesteps,

$$
\operatorname{logit}(t_m)
=\operatorname{logit}(t_n)+\log\alpha.
$$

This is a translation in logit time, not a constant addition in ordinary time. Adding a fixed amount to every $t$ would require clipping near one and would collapse some distinct samples onto the same endpoint. The odds transformation preserves both endpoints and remains strictly increasing:

$$
\frac{d t_m}{d t_n}
=\frac{\alpha}{[1+(\alpha-1)t_n]^2}>0.
$$

For $m>n$, every interior timestep moves toward noise. Applying this map to sampled training times changes how frequently the model encounters different corruption levels. It does not require changing the interpolation endpoints.

<figure>
  <picture>
    <source media="(max-width: 620px)" srcset="/images/blog/rae-dimension-noise-schedule/schedule-mobile.svg" width="346" height="307" />
    <img src="/images/blog/rae-dimension-noise-schedule/schedule.svg" alt="Analytical noise schedule curves for dimension ratios one, four and sixteen. Larger ratios shift interior timesteps toward more noise." loading="lazy" width="595" height="307" />
  </picture>
  <figcaption>Figure 1. The exact schedule transformation for three dimension ratios. Increasing latent dimension moves an interior timestep toward stronger corruption. These curves do not measure generation quality.</figcaption>
</figure>

## A minimal model of redundant information

Consider a deliberately simple latent: a scalar value $z$, with unit variance, repeated in $D$ coordinates. Each coordinate receives independent unit-variance noise:

$$
y_i=(1-t)z+t\varepsilon_i,
\qquad i=1,\ldots,D.
$$

Average the coordinates. The shared signal survives the average, while independent noise has variance $t^2/D$:

$$
\bar y=(1-t)z+t\bar\varepsilon,
\qquad
\operatorname{Var}(\bar\varepsilon)=\frac1D.
$$

The signal-to-noise variance ratio available to this estimator is therefore

$$
\mathrm{SNR}_{\mathrm{average}}(D,t)
=D\left(\frac{1-t}{t}\right)^2.
$$

If $D$ grows from $n$ to $m$, preserving this particular SNR requires

$$
m\left(\frac{1-t_m}{t_m}\right)^2
=n\left(\frac{1-t_n}{t_n}\right)^2.
$$

Taking positive square roots gives exactly the odds scaling above. This toy model explains why a square root of the dimension ratio is plausible when independent noise obscures redundant signal. It is not a claim that real RAE channels are copies, or a proof that their effective information dimension equals the number of coordinates.

Take $n=4096$, $m=16384$, and $t_n=1/2$. Then $\alpha=2$ and

$$
t_m=\frac{2(1/2)}{1+(2-1)(1/2)}=\frac23.
$$

In the repeated-signal example, both cases have SNR $4096$. Without the shift, the larger latent at $t=1/2$ would have SNR $16384$. The larger number here means the toy estimator has an easier recovery problem.

For a wider illustrative latent with $256$ tokens and $768$ channels, $m=196608$, so $\alpha=\sqrt{48}$. The same reference midpoint moves to approximately $0.8739$. This is an arithmetic example, not a recommended setting independent of the representation's normalization and statistics.

## Checking the transformation

The [runnable NumPy example](/code/rae-dimension-noise-schedule.py) verifies the endpoints, odds ratio, inverse map, and the toy SNR calculation. Its essential calculation is short:

~~~python
import numpy as np

def shift(t, source_dim, target_dim):
    alpha = np.sqrt(target_dim / source_dim)
    return alpha * t / (1 + (alpha - 1) * t)

t = np.array([0.1, 0.5, 0.9])
shifted = shift(t, 4096, 16384)
assert np.allclose(shifted / (1 - shifted), 2 * t / (1 - t))
~~~

The inverse is useful when checking sampling code:

$$
t_n=\frac{t_m}{\alpha-(\alpha-1)t_m}.
$$

One common implementation error is to apply the correct expression to the opposite time convention. If a codebase uses $t=0$ for noise and $t=1$ for data, its numerical direction must be converted first. Another is to count only spatial positions despite using a much wider latent. The dimension convention and the interpolation convention must agree with the schedule.

## Where the argument stops

The shift is useful when changing the amount of recoverable structure in a latent changes the difficulty of denoising. It is a low-cost way to move training examples toward a more useful corruption regime. It cannot fix an encoder that discarded needed details, a decoder that cannot reconstruct them, or a denoiser whose capacity is inadequate for the representation.

Dimension also does not specify the latent's covariance, semantic redundancy, or coordinate scale. Correlated noise would change the averaging calculation. Strong channel anisotropy would make a single scalar schedule an incomplete description of corruption. These are reasons to inspect representation statistics rather than interpret the dimension formula as a universal law.

RAE's contribution here is precise: extend the dimension-dependent shift to account for channel width as well as token count, and evaluate that choice in its latent-generation setting. The mechanism can be understood as multiplying noise-to-signal odds, with a simple redundancy model explaining the square root.

## References

1. Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. [Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2510.11690v1). 2025.
2. Ting Chen. [On the Importance of Noise Scheduling for Diffusion Models](https://arxiv.org/abs/2301.10972v1). 2023.
3. Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. [Simple diffusion: End-to-end diffusion for high resolution images](https://arxiv.org/abs/2301.11093v1). 2023.
4. Patrick Esser et al. [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206v1). 2024.
