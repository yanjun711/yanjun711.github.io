---
title: "Joint Shape-and-Layout Generation in SAM 3D"
description: "Why a reconstructed object and its camera-space placement need to agree, down to the meaning of rotation."
date: 2025-12-13
tags: [3D vision, Generative models]
draft: false
toc: true
---

A reconstruction system produces a plausible chair from a photograph. Its legs, back, and seat look reasonable in isolation. Yet when we place it back into the camera view, it faces the wrong direction and floats above the floor. Generating a convincing object and explaining the object in the photograph are different requirements.

SAM 3D addresses this distinction by generating coarse shape and camera-space layout jointly. Its geometry model predicts an occupancy representation together with rotation, translation, and scale. This note examines that coupling: what the variables mean, how their tokens interact, and why a rotation prediction is incomplete without a shape convention.

## A shape does not specify its placement

For an input image $I$ and a target object's binary mask $M$, the geometry model describes the conditional distribution

$$
p(O,R,t,s\mid I,M).
$$

Here $O$ is a coarse $64^3$ occupancy representation, $R$ is rotation represented by six parameters, $t\in\mathbb R^3$ is translation, and $s\in\mathbb R^3$ gives scale along the object axes. The six rotation parameters encode a three-dimensional rotation; they do not imply six physical rotational degrees of freedom.

The output variables answer related questions. Shape determines which points constitute the object. Rotation and scale determine how those points are oriented and stretched. Translation locates the result relative to the camera. The final rendered silhouette depends on all of them.

It is possible to factor the same distribution as $p(O\mid I,M)p(R,t,s\mid O,I,M)$ by the probability chain rule. That does not mean the paper implements two independent predictors. SAM 3D's geometry model denoises shape and layout together, allowing their representations to exchange information during generation.

## Two views of the evidence

The model conditions on four sets of DINOv2 tokens: a cropped object image, its cropped mask, the full image, and the full-image mask. The crop emphasizes object detail. The full image retains scene context that the crop can remove, such as nearby objects and surfaces.

The masks identify which object the model should explain. Without the full view, a tightly cropped image may make a small nearby object and a large distant object look similar. Without the crop, a small object can receive relatively few pixels in the full-image representation. The two views serve different geometric purposes.

These cues do not make single-view reconstruction uniquely determined. An occluded back surface can have several plausible shapes, and absolute scale may remain ambiguous. Conditioning supplies evidence and learned priors; it does not turn every hidden quantity into an observation.

<figure>
  <img src="/images/blog/sam3d-shape-layout/joint-geometry.svg" alt="Cropped object and full-image context condition a geometry model. Shape tokens exchange information with rotation, translation, and scale tokens, and the resulting shape and placement combine in camera coordinates." width="540" height="530" />
  <figcaption>Figure 1. A conceptual view of the geometry model. Shape and layout communicate during denoising. The arrows summarize information flow; they are not a reproduction of the model's structured attention mask.</figcaption>
</figure>

## What is actually mixed?

Appendix C.1 makes the tokenization concrete. The $64^3$ occupancy grid is represented in a coarser $16^3\times8$ latent. Its $4096$ spatial positions become shape tokens. Rotation, translation, and scale each contribute one layout token.

Modality-specific input projections map those tokens into a shared feature width of $1024$. Output projections return predictions to the appropriate parameter spaces. A Mixture of Transformers uses one transformer for shape and another whose parameters are shared across the layout modalities. Joint self-attention permits information exchange between them.

This arrangement matters because shared attention and shared parameters are different choices. Rotation need not use the same input projection as an eight-channel shape latent merely because both participate in an attention operation. Conversely, separate modality processing does not require the streams to ignore one another.

The paper also uses a structured attention mask to support training on datasets with incomplete modality labels, and to permit selective fine-tuning. For example, shape-only supervision need not require reliable camera-space placement annotations. The exact mask is an architectural detail beyond this diagram; the essential point here is that modality-specific processing retains a route for joint context.

Why should shape influence rotation? Imagine a shape generator that chooses a different canonical orientation for a chair. A fixed camera rotation that was correct for the old canonical orientation is now wrong. The output rotation only has meaning relative to the output shape's coordinate frame. Independent plausibility is insufficient; their composition must explain the same image.

## A point transformed back into the camera

Use column vectors and define the transformation for this example as

$$
p_{\mathrm{cam}}
=R\operatorname{diag}(s)p_{\mathrm{obj}}+t.
$$

This is an explicit convention for the calculation, rather than a claim about every matrix-storage convention in the released code. Scale acts in object coordinates, rotation follows, and translation is in camera coordinates. Changing the operation order generally changes the result.

Take

$$
p_{\mathrm{obj}}=
\begin{bmatrix}1\\0\\0\end{bmatrix},
\quad s=
\begin{bmatrix}2\\1\\1\end{bmatrix},
\quad R=
\begin{bmatrix}
0&-1&0\\1&0&0\\0&0&1
\end{bmatrix},
\quad t=
\begin{bmatrix}1\\0\\4\end{bmatrix}.
$$

The point first becomes $(2,0,0)$, then $(0,2,0)$ after a $90$-degree rotation about the $z$ axis, and finally $(1,2,4)$ in camera coordinates. Under a pinhole camera with focal length $f=100$ pixels and principal point at the origin,

$$
u=f\frac{X}{Z}=25,
\qquad v=f\frac{Y}{Z}=50.
$$

This tiny calculation exposes the coupling. The projected location is not a property of $R$, $s$, or $t$ alone. It is a property of their composition with the particular object point.

Now double both $s$ and $t$. The camera point becomes $(2,4,8)$, but its projection remains $(25,50)$. Every point transformed this way preserves its image projection. Without an additional metric cue, the image cannot distinguish those globally scaled configurations.

The [runnable NumPy example](/code/sam3d-shape-layout.py) checks the transformation and this scale ambiguity:

~~~python
import numpy as np

p = np.array([1.0, 0.0, 0.0])
s = np.array([2.0, 1.0, 1.0])
R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
t = np.array([1.0, 0.0, 4.0])

camera_point = R @ (s * p) + t
uv = 100 * camera_point[:2] / camera_point[2]
assert np.allclose(camera_point, [1, 2, 4])
assert np.allclose(uv, [25, 50])
~~~

## What joint prediction can and cannot resolve

Joint generation is useful when evidence for one output changes the interpretation of another: a proposed chair back determines which orientation explains the image; a ground plane helps judge placement; a visible silhouette constrains the combination of shape and scale. Cross-modal interaction gives the model a way to revise these variables together.

It does not establish unique geometry behind an occluder. A plausible reconstruction may still differ from the true hidden surface. A learned size prior may choose a useful scale without measuring the object's physical dimensions. A visually aligned object may also violate contact or collision constraints that are not explicitly enforced by the prediction process.

The geometry model is also only one stage of SAM 3D. A later model handles texture and refinement. The mechanism discussed here should therefore be evaluated by shape-and-placement consistency, without assigning every improvement in the final textured asset to the joint geometry tokens.

The practical distinction is straightforward: an object can look good in its own viewer and still fail to explain its source image. Joint shape-and-layout generation puts those two requirements into the same conditional modeling problem.

## Reference

- [*SAM 3D: 3Dfy Anything in Images*](https://arxiv.org/abs/2511.16624v1), November 2025. This note follows the first version, particularly Sections 2.1–2.2 and Appendix C.1.
