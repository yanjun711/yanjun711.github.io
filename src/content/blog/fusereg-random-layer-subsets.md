---
title: "Random Layer Subsets as Regularization in FuseReg"
description: "Why averaging random nonempty subsets preserves a representation's mean while perturbing exactly the directions where encoder layers disagree."
date: 2026-10-10
tags: [Image generation, Representation learning]
draft: false
toc: true
---

In a [representation autoencoder](https://arxiv.org/abs/2510.11690v1), several frozen encoder layers can describe the same image at different levels of detail. Averaging them gives a convenient latent, but a decoder trained on only that average may depend on a particular cancellation between layers. Change the layer composition and its prediction can change sharply.

[FuseReg](https://arxiv.org/abs/2609.31620v1) trains downstream models on averages of random **nonempty layer subsets**. The operation preserves the full-layer mean in expectation while injecting variation where layers disagree. A three-layer example makes that property visible and explains the resulting regularizer under a linear squared-loss model.

## Normalize by the subset you actually kept

For one image, let $h_1,\ldots,h_K$ be the layer-normalized features from $K$ chosen frozen encoder layers. Each can be a token tensor; vectorize it for the analysis. The deployment representation is

$$
\bar h=\frac{1}{K}\sum_{k=1}^{K}h_k.
$$

Independently keep each layer with probability $1-p$, but condition on keeping at least one. If $S$ is the retained subset and $R=|S|$, the training latent is

$$
z_S=\frac{1}{R}\sum_{k\in S}h_k.
$$

The denominator is the actual number retained. This differs from the fixed keep-probability scaling used in [Dropout](https://jmlr.org/papers/v15/srivastava14a.html), and an empty subset cannot be assigned an arbitrary zero latent without changing the distribution.

For $0<p<1$, every nonempty subset is possible. A particular subset of size $r$ has probability

$$
\Pr(S)=\frac{(1-p)^r p^{K-r}}{1-p^K}.
$$

The distribution is exchangeable across layer indices. Consequently, each layer has the same expected normalized weight; since the weights always sum to one, each expected weight is $1/K$. Therefore

$$
\mathbb E_S[z_S]=\bar h.
$$

The sampler therefore preserves the latent mean. The decoder's response to the variation depends on its mapping, which we can analyze exactly in the linear case below.

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

Use three two-dimensional layer features:

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

## From linear decoders to nonlinear models

The sampler's mean and covariance identities hold for any downstream model. For a linear decoder with squared loss, they give the explicit disagreement penalty derived above. In a nonlinear decoder or diffusion transformer, a local Jacobian describes sensitivity to small subset perturbations. Perceptual and adversarial reconstruction losses contribute their own responses to those perturbations.

Larger subset changes can cross nonlinear regions, and disagreement between layers can carry useful detail. The drop rate controls how strongly training discourages dependence on those differences, trading sensitivity to layer composition against reconstruction fidelity.

Equal per-layer keep probabilities and normalization by the retained count produce the mean-preserving sampler. Biased sampling or a different denominator changes the expected latent and therefore the representation seen during training.

## References

1. [FuseReg: Regularizing Layer Fusion Mitigates the Reconstruction-Generation Gap in Representation Autoencoders](https://arxiv.org/abs/2609.31620v1). 2026.
2. Boyang Zheng et al. [Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2510.11690v1). 2025.
3. Nitish Srivastava et al. [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html). 2014.
