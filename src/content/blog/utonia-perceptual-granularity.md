---
title: "Perceptual Granularity Rescaling in Utonia"
description: "How changing the spatial unit makes point-cloud positional hints more comparable across objects and scenes."
date: 2026-04-11
tags: [3D vision, Representation learning]
draft: false
toc: true
---

A local geometric pattern can occupy a few millimeters in an object scan and several centimeters in a scene scan. Feeding both point clouds to the same encoder does not make those distances comparable. A positional representation still sees the numbers supplied to it, including the unit and scale choices that produced them.

[Utonia](https://arxiv.org/abs/2603.03283v1) introduces **perceptual granularity rescaling** before positional encoding. Changing the coordinate unit lets shared geometric operators work at comparable local scales across domains. The chosen granularity describes the resolution of local structure, separately from the point cloud's total extent.

## What should one spatial unit mean?

Consider an object scan with a 0.1-meter extent and a room scan with a 10-meter extent. Both might contain the same number of points. That alone says little about what a displacement of 0.01 meters means: it could cross a substantial part of the object or a tiny fraction of the room.

Normalizing every bounding box to unit width answers one particular question: where is a point relative to the full sample? It also couples every local displacement to the sample's outermost points. Adding a distant wall changes the normalized size of an otherwise identical chair. Removing empty space changes it again. This can be useful for some tasks, but it is a different choice from setting the spatial resolution at which local structure should be interpreted.

Utonia pretrains a [Point Transformer V3](https://arxiv.org/abs/2312.10035v1) encoder across these domains. Section 3.2 starts from this mismatch: spatial extents, sampling patterns, and coordinate conventions differ across sensors and domains. The same architectural unit can then correspond to very different metric extents. The proposed rescaling makes the input coordinates express a more comparable perceptual granularity before those coordinates enter positional encoding.

The paper samples rescaling factors according to the original desired granularity. Choosing that granularity sets the spatial scale at which the representation describes local structure; point count and sample extent describe other properties of the input.


## Change the coordinates before encoding them

For an explanatory notation, let $x_i$ be a point, $c$ a chosen center, and $g>0$ a spatial unit representing the desired granularity. Write

$$
\tilde x_i=\frac{x_i-c}{g}.
$$

I use this notation to isolate scalar rescaling within the paper's broader preprocessing, which also includes coordinate perturbations and sampled scale factors. Relative displacements become:

$$
\tilde x_i-\tilde x_j=\frac{x_i-x_j}{g}.
$$

Translation cancels. A separation of $g$ meters becomes a displacement of one coordinate unit. A separation of $2g$ becomes two units, regardless of whether the original input used small object coordinates or large scene coordinates.

Why do this before positional encoding? Once a position has been turned into a nonlinear signal, scaling that signal generally does not equal encoding a scaled position. For a single rotary plane, a displacement $\delta$ produces a phase difference proportional to $\delta/\theta$, where $\theta$ is the frequency scale for that plane. Rescaling the coordinates first instead produces $\delta/(g\theta)$.

Utonia's next section applies RoPE, introduced in [RoFormer](https://arxiv.org/abs/2104.09864v1), to granularity-aligned coordinates. Attention receives continuous relative-position hints whose spatial unit has already been adjusted. A useful scalar illustration is

$$
\phi_{ij}=\frac{x_i^{(1)}-x_j^{(1)}}{g\theta}.
$$

Here $x^{(1)}$ selects one coordinate axis; this is one phase component, not the complete three-dimensional embedding. If corresponding local displacements differ by a factor of ten and their chosen granularities differ by the same factor, their phase components agree.

Section 3.3 separates RoPE's continuous positional signal from neighborhood construction. Neighborhood selection determines which points interact; RoPE describes their relative positions within those interactions.

## A four-point example

Take two planar squares, embedded in three dimensions:

$$
X_A=0.01\begin{bmatrix}0&0&0\\1&0&0\\0&1&0\\1&1&0\end{bmatrix},
\qquad
X_B=10X_A.
$$

Their edge lengths are 0.01 and 0.10 meters. Choose $g_A=0.01$ meters and $g_B=0.10$ meters for this example. Set both centers to zero. Then

$$
\frac{X_A}{g_A}=\frac{X_B}{g_B}.
$$

Each normalized square has unit edges and a diagonal of $\sqrt{2}$. The original metric distances remain different; we have selected a coordinate system in which the two local patterns are comparable.

For a single rotary plane with $\theta=1$, an edge displacement along the first axis gives phases of 0.01 and 0.10 radians before normalization. Afterwards both give one radian. With identical content vectors $q=k=[1,0]$, the rotary dot-product factor for that displacement is $\cos(1)$ in both cases. The equality comes from equal relative coordinates; no learned invariance is needed to establish this particular arithmetic fact.

Now add a distant point to one cloud. A bounding-box normalizer would change the coordinates of the original square because the full extent has changed. Holding $g$ fixed leaves its local normalized distances unchanged. Conversely, if the task calls for a different perceptual granularity, changing $g$ deliberately changes those distances even when the bounding box stays fixed.


[Run the NumPy example](/code/utonia-perceptual-granularity.py) to verify the distance and phase equalities for the chosen granularities. A translated copy also gives the same pairwise displacements after consistent recentering.

## Scale is not the only coordinate convention

A scene and an object can need different treatment even after their local spatial units become comparable. In an upright room scan, height and gravity often carry useful structure: floors support furniture, and ceilings are above it. Randomly rotating every scene removes that stable reference.

An isolated object, by contrast, may need recognition under arbitrary orientations. Preserving one training orientation can make a feature depend on how the scanner happened to place the object.

Utonia therefore conditions its gravity prior on the original desired granularity. It preserves upright structure for scene-scale, coarser-granularity inputs and encourages arbitrary orientations for object-scale, finer-granularity inputs. Sharing a spatial unit does not require imposing one axis convention on every domain.

The teacher and student views also use asymmetric scales, rotations, and shifts. Feature consistency across those views trains the encoder to tolerate coordinate changes. The square example gives exact coordinate equalities; augmentation-based training extends the idea to views whose normalized arrays differ.

## Choosing granularity for the task

The mechanism is useful when joint training mixes point clouds whose numeric units make comparable geometric patterns look substantially different. The rescaling factor can also be adjusted at inference to match a desired granularity, as the paper describes. It gives a concrete control over the spatial unit that enters the positional signal.

After rescaling, coverage, occlusion, sensor noise, and sampling density still determine which geometry is available. Comparable coordinate units make local patterns easier to share across domains; the observed points determine the neighborhoods those operators can use.

Tasks such as distinguishing miniatures, estimating robot clearance, and measuring structures also use absolute scale. Retaining the chosen $g$ alongside the normalized coordinates preserves the conversion back to metric units.

Finally, a good granularity for one task may be too coarse for another. Rescaling changes the positional frequencies associated with a physical displacement. It makes a specific spatial comparison easier to share; its usefulness still depends on which comparisons the task requires.

## References

1. [Utonia: Toward One Encoder for All Point Clouds](https://arxiv.org/abs/2603.03283v1). 2026.
2. [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/abs/2312.10035v1). 2023 preprint; CVPR 2024.
3. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v1). 2021.
