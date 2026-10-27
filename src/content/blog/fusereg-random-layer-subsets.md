---
title: "Random Layer Subsets as Regularization in FuseReg"
description: "Why averaging random nonempty subsets preserves a representation's mean while perturbing exactly the directions where encoder layers disagree."
date: 2026-10-10
tags: [Image generation, Representation learning]
draft: false
toc: true
---

Several frozen encoder layers can describe the same image at different levels of detail. Averaging them gives a convenient latent, but a decoder trained on only that average may depend on a particular cancellation between layers. Change the layer composition and its prediction can change sharply.

[FuseReg](https://arxiv.org/abs/2609.31620v1) trains downstream models on averages of random **nonempty layer subsets**. Its useful property is unusually precise: the operation preserves the full-layer mean in expectation, while injecting variation only where layers disagree. A three-layer example makes that property visible and explains the resulting regularizer under a linear squared-loss model.

## Normalize by the subset you actually kept

For one image, let $h_1,\ldots,h_K$ be the layer-normalized features from $K$ chosen frozen encoder layers. Each can be a token tensor; vectorize it for the analysis. The deployment representation is

$$
\bar h=\frac{1}{K}\sum_{k=1}^{K}h_k.
$$

Independently keep each layer with probability $1-p$, but condition on keeping at least one. If $S$ is the retained subset and $R=|S|$, the training latent is

$$
z_S=\frac{1}{R}\sum_{k\in S}h_k.
$$

The denominator is the actual number retained. This is not the usual inverted-dropout scaling by a fixed keep probability, and an empty subset cannot be assigned an arbitrary zero latent without changing the distribution.

For $0<p<1$, every nonempty subset is possible. A particular subset of size $r$ has probability

$$
\Pr(S)=\frac{(1-p)^r p^{K-r}}{1-p^K}.
$$

The distribution is exchangeable across layer indices. Consequently, each layer has the same expected normalized weight; since the weights always sum to one, each expected weight is $1/K$. Therefore

$$
\mathbb E_S[z_S]=\bar h.
$$

This identity holds for the latent itself, before making any assumption about a decoder. It does not imply that a nonlinear decoder's average prediction equals its prediction at the average latent.

## The noise follows layer disagreement

Define the layer deviations and their covariance:

$$
e_k=h_k-\bar h,\qquad
V=\frac{1}{K}\sum_{k=1}^{K}e_ke_k^{\mathsf T}.
$$

Conditioned on subset size $R=r$, exchangeability makes $S$ a uniform size-$r$ sample without replacement. Its mean has covariance

$$
\operatorname{Cov}(z_S\mid R=r)
=\frac{K-r}{r(K-1)}V.
$$

The finite-population correction matters: keeping all $K$ layers gives zero variance, not the variance of sampling with replacement. Averaging over the nonempty-conditioned binomial distribution of $R$ gives

$$
\operatorname{Cov}(z_S)=c_K(p)V,\qquad
c_K(p)=\mathbb E\!\left[\frac{K-R}{R(K-1)}\right].
$$

The scalar $c_K(p)$ controls strength; $V$ determines direction. If all layers agree on a component, every subset mean has that same component. Increasing the drop rate cannot perturb information that is identical across the selected layers.

At $p=0$, all layers are retained and $c_K(0)=0$. As $p$ approaches 1, conditioning on nonempty increasingly selects singletons, and $c_K(p)$ approaches 1. The coefficient does not diverge, although rejection sampling becomes inefficient near that limit.

## Three layers, seven possible means

Use the deliberately simple features

$$
h_1=(1,0),\quad h_2=(1,1),\quad h_3=(1,-1).
$$

They agree completely on the first coordinate and disagree on the second. Their full mean is $(1,0)$. Let $p=1/2$. All eight raw binary masks are equally likely; conditioning away the empty mask makes the seven retained subsets equiprobable.

| Retained layers | Subset mean |
| --- | --- |
| 1 | $(1,0)$ |
| 2 | $(1,1)$ |
| 3 | $(1,-1)$ |
| 1, 2 | $(1,1/2)$ |
| 1, 3 | $(1,-1/2)$ |
| 2, 3 | $(1,0)$ |
| 1, 2, 3 | $(1,0)$ |

The second coordinate has mean zero and variance

$$
\frac{0^2+1^2+(-1)^2+(1/2)^2+(-1/2)^2+0^2+0^2}{7}
=\frac{5}{14}.
$$

For these features, $V=\operatorname{diag}(0,2/3)$ and $c_3(1/2)=15/28$, so $c_3(1/2)V=\operatorname{diag}(0,5/14)$, exactly as enumeration predicts.

<figure>
  <img src="/images/blog/fusereg-random-layer-subsets/subsets.svg" alt="Seven subset means on the disagreement coordinate: minus one, minus one-half, three coincident zeros, one-half, and one. The shared coordinate remains one for every subset." width="480" height="332" />
  <figcaption>Figure 1. Exact subset means for the three-layer example. Each dot represents one equally likely subset; the three dots at zero are stacked to show multiplicity. Only the layer-disagreement coordinate changes.</figcaption>
</figure>

Now give a linear decoder $D=(0,2)$ the target $y=0$. It produces zero from the full mean, but its expected squared error under subset sampling is

$$
\mathbb E\|Dz_S-y\|^2=4\cdot\frac{5}{14}=\frac{10}{7}.
$$

In contrast, $D=(1,0)$ with target $y=1$ is correct for every subset. Random fusion penalizes sensitivity to the disagreeing component while leaving this shared component untouched. The [exact-enumeration script](/code/fusereg-subset-variance.py) verifies these numbers and the covariance identity for several other layer counts, rates, and feature matrices.

## Why that becomes a regularizer

For a fixed input, a linear decoder $D$, target $y$, and squared loss, expand $z_S=\bar h+\delta_S$, where $\mathbb E\delta_S=0$. The cross term vanishes, leaving the exact identity

$$
\begin{aligned}
\mathbb E_S\|Dz_S-y\|^2
&=\|D\bar h-y\|^2\\
&\quad+c_K(p)\operatorname{tr}(DVD^{\mathsf T}).
\end{aligned}
$$

The extra term is large when the decoder maps disagreement directions to large output changes. It is zero for a decoder insensitive to those directions. This gives a concrete interpretation of the stochastic training operation without treating its perturbation as isotropic Gaussian noise.

The paper applies the same subset operation at two interfaces. The pixel decoder reconstructs an image from a subset latent. The diffusion transformer receives a noisy subset latent but predicts the **full-layer mean**, not the sampled subset:

$$
z_t=t z_S+(1-t)\epsilon,
\qquad\text{target}=\bar h.
$$

Here time runs from noise to data. For a linear predictor and the paper's time-weighted squared loss, the disagreement penalty acquires the additional factor $\rho(t)t^2$, where $\rho(t)=\max(1-t,0.05)^{-2}$. The two stages therefore need not use the same drop rate: they see different inputs and predict different targets even though their subset sampler is shared.

## The limit of the linear explanation

The mean and covariance identities are exact properties of the sampling distribution. The squared-loss decomposition is exact under its stated linear assumptions. Actual representation-autoencoder decoders and diffusion transformers are nonlinear, and reconstruction training also includes perceptual and adversarial terms. The linear identity does not prove an identical explicit penalty for their full objectives.

A local Jacobian can help interpret small perturbations, but large subset changes can move a network across nonlinear regions. Strong disagreement may also carry useful detail rather than a nuisance shortcut. Regularizing sensitivity to it can trade fidelity for robustness, so a larger drop rate is not automatically better.

Finally, equal per-layer keep probabilities are essential to the simple mean-preservation argument. Biased layer sampling or a different normalization can shift the expected latent. Applying the mechanism requires preserving the sampling convention as carefully as the tensor dimensions.

## Source

[FuseReg, v1](https://arxiv.org/abs/2609.31620v1), September 2026. Sections 3.2–3.3 define normalized nonempty subsets and the two training targets; Section 4 and Appendix E derive the mean, covariance, and linear-loss identities. The seven-subset figure and decoder calculation above are exact toy examples, not reported generation results.
