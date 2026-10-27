---
title: "Latent-to-4D: Aligning Video Latents with Geometry Tokens"
description: "Why matching a latent tensor's shape is insufficient, and how resampling and learned local projection bridge a video VAE and a 4D reconstructor."
date: 2026-09-12
tags: [3D vision, Video generation]
draft: false
toc: true
---

A video generator ends with a denoised latent tensor. A 4D reconstructor begins with spatiotemporal tokens. It is tempting to reshape one into the other and bypass video decoding. But equal tensor sizes do not imply equal meanings: a channel learned for appearance reconstruction need not be a channel a geometry model knows how to use.

[Beyond Pixels](https://arxiv.org/abs/2608.10744v1) introduces **Latent-to-4D**, which connects the video VAE's representation directly to a pretrained 4D reconstruction hierarchy. This note focuses on the alignment interface: fixed grid resampling followed by a learned local conversion. It explains why both are needed, and why compatibility is a stronger condition than matching dimensions.

## The mismatch at the interface

Let a video latent have shape

$$
z\in\mathbb R^{T_v\times H_v\times W_v\times C_v}.
$$

The geometry hierarchy expects $T$ frames, $M$ spatial tokens per frame, and a feature dimension $d$. Its initial token tensor has shape $T\times M\times d$. Batch dimensions are omitted for now, and this channels-last notation is for explanation rather than an implementation prescription.

Three mismatches appear independently. Temporal compression changes how many latent slices represent a clip. Spatial compression changes the sampling grid. Feature learning changes what the channels encode. Reshaping can reorder existing entries, but it cannot solve these three problems by itself.

The conventional route decodes the latent into RGB frames, then feeds those frames through a reconstruction model's image encoder. That route uses a familiar image interface, but introduces decoding and re-encoding work and allows RGB decoding artifacts to influence geometry. Latent-to-4D instead learns the representation conversion before RGB exists.

## Resample the grid, then learn the local conversion

The paper's alignment map is

$$
Q^{(0)}=\mathcal A_\phi(z)
=\operatorname{Flatten}\!\left(
\mathcal S_\phi\bigl(\mathcal R(z)\bigr)
\right).
$$

$\mathcal R$ is fixed trilinear resampling. It matches the temporal and spatial grid required by the geometry hierarchy. Within each interpolation cell, the resampled value is a weighted combination of its eight neighboring grid values, with weights determined by relative coordinates.

$\mathcal S_\phi$ is a learned 3D convolution. It converts the channel dimension and aggregates local spatiotemporal neighborhoods. Finally, Flatten groups the spatial grid into $M$ tokens while retaining the frame axis.

The order separates geometry of the grid from meaning of the features. Interpolation determines where samples are placed; the learned convolution determines how local appearance and motion evidence should be represented for the downstream hierarchy. Interpolation alone does not assign metric depth to a channel or establish a 3D correspondence.


The refinement hierarchy and prediction heads are initialized from [4RC](https://arxiv.org/abs/2602.10094v1). Alignment is followed by refinement rather than immediate geometric decoding. For a batch of tokens shaped $(B,T,M,d)$, frame-wise attention treats frames as separate batch elements, yielding $(BT,M,d)$. Global attention instead uses $(B,TM,d)$ to exchange evidence across frames. Alternating these operations supplies long-range information beyond the alignment convolution's local receptive field.

This distinction matters when describing the mechanism: the adapter makes features consumable by the geometry hierarchy; the adapter alone is not the complete reconstructor.

## A one-dimensional example you can calculate

Reduce the problem to time only, two input channels, and a pointwise channel projection. This is an analogue of the interface, not a reproduction of the learned 3D convolution.

Take two latent samples at times 0 and 1:

$$
z(0)=(1,0),\qquad z(1)=(0,1).
$$

Resample at $\tau\in\{0,1/3,2/3,1\}$. Linear interpolation gives

$$
\widetilde z(\tau)=(1-\tau,\tau).
$$

Use a supplied projection matrix

$$
W=\begin{bmatrix}1&0&-1\\0&1&1\end{bmatrix},
\qquad q(\tau)=\widetilde z(\tau)W.
$$

The aligned tokens are therefore

$$
Q=\begin{bmatrix}
1&0&-1\\
2/3&1/3&-1/3\\
1/3&2/3&1/3\\
0&1&1
\end{bmatrix}.
$$

<figure>
  <picture>
    <source media="(max-width: 620px)" srcset="/images/blog/latent-to-4d-alignment/interface-mobile.svg" width="346" height="451" />
    <img src="/images/blog/latent-to-4d-alignment/interface.svg" alt="Linear interpolation of two latent channels at four times, followed by a supplied projection into three output channels." loading="lazy" width="595" height="307" />
  </picture>
  <figcaption>Figure 1. The one-dimensional alignment example: resampling changes the time grid, and the supplied matrix W converts channels. Points show the four target times. These are exact toy values, not learned geometry predictions.</figcaption>
</figure>

Four samples now occupy the target time grid, and each has three output channels. Those are separate accomplishments: interpolation supplied the additional positions, while $W$ supplied the channel conversion. Neither operation created new observed information between the endpoints.

The [runnable example](/code/latent-to-4d-alignment.py) checks this matrix with NumPy. Its central operations are:

```python
resampled = np.column_stack([
    np.interp(target_times, source_times, z[:, channel])
    for channel in range(z.shape[1])
])
tokens = resampled @ projection
```

Now swap the input channels without changing the tensor shape. At time 0, the adapter receives $(0,1)$ and outputs $(0,1,1)$, whereas the original convention produced $(1,0,-1)$. All dimensions still match. The representation does not.

For this artificial swap, multiplying $W$ by the inverse channel permutation restores the original outputs; the script verifies that correction. Real VAE differences are usually not known permutations. They can involve nonlinear representations, different scalings, and different temporal compression. A same-shaped tensor from a different VAE cannot be assumed compatible with an already trained adapter.

## What is trained, and where the latent comes from

During training, the frozen [Wan](https://arxiv.org/abs/2503.20314v1) video VAE encodes observed videos with metric 4D annotations. The pathway uses the VAE posterior mean,

$$
z^{\mathrm{obs}}=\mu(E_v(V)),
$$

and learns the alignment module, lightweight refinement updates, and geometry and camera heads. The video diffusion transformer is neither executed nor optimized in this training pathway.

At generation time, the source changes to the compatible video generator's **final denoised latent**, $z^{\mathrm{gen}}$, immediately before VAE decoding. The downstream alignment and reconstruction pathway stays the same. An intermediate noisy diffusion state is a different input distribution and is not interchangeable with that terminal latent.

The output is structured geometry, including per-frame cameras and world-space pointmaps. In the decoder's ray parameterization, a point is

$$
P_t(u)=o_t+d_t(u)r_t(u),
$$

with ray origin $o_t$, unit direction $r_t(u)$, and predicted depth $d_t(u)$. For example, origin $(1,0,0)$, direction $(0,0,1)$, and depth 2 give point $(1,0,2)$. The metric values come from learned geometric supervision, not from grid interpolation.

## The common-VAE boundary

The paper's shared interface requires the same VAE checkpoint, latent normalization, tensor layout, compression convention, and a supported latent shape. Within that boundary, the diffusion transformer's architecture or conditioning mode can differ. This is the reason a single downstream pathway can potentially serve several compatible generators.

Even that compatibility does not make observed-video and generated-video latent distributions identical. Generated motion may be inconsistent; geometry cues may be ambiguous or absent; training coverage may be limited. The adapter can run successfully while its geometric predictions remain wrong.

Trilinear resampling also cannot recover fine detail that the VAE discarded. If an event is temporally compressed away or two structures become indistinguishable in the latent representation, increasing the grid resolution does not restore the missing evidence. Refinement can use learned priors and context, but those are inferences rather than recovered measurements.

The useful claim is consequently precise: a learned alignment can replace the RGB interface between a compatible video latent space and a 4D hierarchy. It removes an intermediate representation conversion while retaining the need for geometric supervision, global reasoning, and an explicit account of latent conventions.

## References

1. [Beyond Pixels: From Video Priors to 4D Worlds](https://arxiv.org/abs/2608.10744v1). 2026.
2. Yihang Luo et al. [4RC: 4D Reconstruction via Conditional Querying Anytime and Anywhere](https://arxiv.org/abs/2602.10094v1). 2026.
3. Wan Team. [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314v1). 2025.
