---
title: "DINOv3’s Gram Anchoring for Dense Features"
description: "Preserving patch-to-patch similarities without forcing feature channels to stay fixed."
date: 2025-09-13
tags: [Visual representation, Dense prediction]
draft: false
toc: true
---

An image representation can improve at identifying an image while becoming worse at describing its parts. In [DINOv3](https://arxiv.org/abs/2508.10104v1), extended training improves global recognition metrics, but the patch features can lose the local consistency that dense tasks need. Patches that should be distinct become spuriously similar.

One response would be to keep every patch embedding close to an earlier checkpoint. But that would also constrain changes in the feature coordinate system, including changes that preserve all relative similarities. DINOv3's **Gram anchoring** chooses a narrower target: preserve the matrix of pairwise patch similarities. The following derivation explains what that target fixes, what it leaves free, and why the teacher still matters.

## What should remain stable?

Suppose two patches belong to a person's shirt and another belongs to the background. A dense representation should make the two shirt patches relate differently from a shirt patch and a background patch. The relevant structure is a set of relations between spatial positions, rather than the value of one particular feature channel.

The paper visualizes this issue with patch cosine-similarity maps. As training continues, those maps become less localized even while global classification improves. A global objective can accept many representations with adequate image-level discrimination; not all of them provide equally useful local structure.

An earlier checkpoint with good dense features supplies a reference. The goal is not to declare every relation from that checkpoint permanently correct. It is to give continued training a stable constraint on relations that its other objectives may neglect.

## The Gram matrix is indexed by patches

For one image crop with $P$ patches, let the student's normalized patch features be the rows of

$$
X_S\in\mathbb{R}^{P\times d},
\qquad \|X_{S,p}\|_2=1.
$$

Let $X_G\in\mathbb{R}^{P\times d}$ be the corresponding normalized features from a Gram teacher. Row $p$ in each matrix refers to the same patch position. Their Gram matrices are

$$
G_S=X_SX_S^{\mathsf T},
\qquad G_G=X_GX_G^{\mathsf T}.
$$

Both have shape $P\times P$. Entry $(p,q)$ is the cosine similarity between patches $p$ and $q$. This is a **patch-to-patch** matrix, not the $d\times d$ channel-to-channel Gram matrix used in some other settings.

The anchoring loss is

$$
\mathcal{L}_{\mathrm{Gram}}
=\|G_S-G_G\|_F^2
=\sum_{p=1}^{P}\sum_{q=1}^{P}
\left(\langle X_{S,p},X_{S,q}\rangle
-\langle X_{G,p},X_{G,q}\rangle\right)^2.
$$

The teacher is treated as a target for this loss. Unit normalization makes diagonal entries equal to one, so the useful constraints concern off-diagonal relations. The expression counts each unordered pair twice; implementations can apply scaling, but the equation above follows the paper's unnormalized Frobenius-norm form.

Why not directly match $X_S$ to $X_G$? Let $Q\in\mathbb{R}^{d\times d}$ be orthogonal, with $QQ^{\mathsf T}=I$. If $X_S=X_GQ$, then

$$
X_SX_S^{\mathsf T}
=X_GQQ^{\mathsf T}X_G^{\mathsf T}
=X_GX_G^{\mathsf T}.
$$

A shared rotation or reflection of feature space leaves the loss zero. Channel values can change while patch relationships remain fixed. Direct feature regression would usually penalize the same transformation.

The invariance is specific. Rotating each patch independently does not generally preserve its dot products with other patches. Neither does an arbitrary nonorthogonal transformation. The objective permits a change of shared coordinates, not unrestricted changes to patch relations.

### How the target enters training

DINOv3 computes Gram anchoring on global crops and combines it with the DINO, iBOT, and diversity objectives. It is an additional relation-preserving constraint, not a replacement for the rest of self-supervised training.

The reported refinement phase starts after one million iterations. Its initial Gram teacher comes from an earlier teacher checkpoint with good dense features. Importantly, that teacher is **not frozen forever**: every 10,000 iterations it is refreshed from the main exponential-moving-average teacher. It is a stable target between refreshes, distinct from the main teacher's ongoing EMA updates.

This schedule allows the reference to move as training progresses. Calling it distillation from one permanently frozen early model would miss that part of the mechanism.

## Three patches are enough to check it

Use two-dimensional unit features for three patches:

$$
X_G=
\begin{bmatrix}
1&0\\
0&1\\
1/\sqrt{2}&1/\sqrt{2}
\end{bmatrix}.
$$

Patches 1 and 2 are orthogonal. Patch 3 is equally similar to both. The reference Gram matrix is

$$
G_G=
\begin{bmatrix}
1&0&1/\sqrt{2}\\
0&1&1/\sqrt{2}\\
1/\sqrt{2}&1/\sqrt{2}&1
\end{bmatrix}.
$$

Now rotate all three features with

$$
Q=\begin{bmatrix}0&-1\\1&0\end{bmatrix},
\qquad X_S=X_GQ.
$$

The vectors change, but their Gram matrix does not. Gram loss is zero, while direct feature regression gives $\|X_S-X_G\|_F^2=6$. Each of the three unit vectors has moved by a right angle, contributing squared distance $2$.

Next replace only patch 3 with patch 1. Its similarity to patch 1 rises from $1/\sqrt{2}$ to $1$, while its similarity to patch 2 falls from $1/\sqrt{2}$ to $0$. This time the geometry has changed:

$$
\mathcal{L}_{\mathrm{Gram}}
=2\left(1-\frac{1}{\sqrt{2}}\right)^2
+2\left(0-\frac{1}{\sqrt{2}}\right)^2
=4-2\sqrt{2}
\approx1.1716.
$$

<figure>
  <picture>
    <source media="(max-width: 620px)" srcset="/images/blog/dinov3-gram-anchoring/gram-matrices-mobile.svg" />
    <img src="/images/blog/dinov3-gram-anchoring/gram-matrices.svg" alt="Toy patch-similarity matrices. The reference and a shared feature rotation have identical entries, while replacing patch 3 changes its similarities to patches 1 and 2." width="900" height="370" />
  </picture>
  <figcaption>Figure 1. Cosine similarities for the three-patch example. These are analytical toy features, not similarity maps measured from DINOv3. Darker blue indicates greater similarity.</figcaption>
</figure>

The full calculation is small:

```python
import numpy as np

def gram_loss(student, teacher):
    # Rows must already be L2-normalized and spatially corresponding.
    difference = student @ student.T - teacher @ teacher.T
    return np.sum(difference ** 2)
```

[Download the checked example](/code/dinov3-gram-anchoring.py). In addition to the two losses, it checks the derivative

$$
\nabla_X\|XX^{\mathsf T}-G_G\|_F^2
=4(XX^{\mathsf T}-G_G)X
$$

with finite differences. This derivative treats $X$ as the already-normalized matrix. If the network produces raw features $Z$ and then normalizes them, the backward pass must also differentiate through that normalization. Ignoring its Jacobian would produce a different update.

## What Gram anchoring can and cannot preserve

The loss is useful when relative patch similarities are the structure worth retaining. It constrains a patch's relations to all other patches in the crop, including distant positions. It is not explicitly a nearest-neighbor smoothness penalty, even though the teacher may provide smooth, localized similarity patterns.

It also does not create semantic correctness by itself. A poor teacher can anchor poor relations. If every teacher patch collapsed to the same unit vector, its Gram matrix would be all ones; a similarly collapsed student could match it perfectly. The quality of the teacher and the other training objectives remain essential.

Spatial correspondence is another assumption. Comparing the same matrix indices after unrelated crop transformations would compare different patches. A patch permutation changes the Gram matrix's row and column identities; the loss is not invariant to permuting only one side.

The high-resolution refinement in Section 4.3 keeps this same basic objective. It runs the Gram teacher at twice the input resolution and downsamples the teacher feature map to the student grid with bicubic interpolation. This changes the target relations while retaining corresponding spatial positions. It is not a new pairwise loss or a requirement that the student run at the teacher's resolution.

Finally, forming a dense $P\times P$ matrix has quadratic storage in patch count. The compact equation does not make high-resolution matching free. Its practical scope depends on crop sizes, feature extraction, and implementation choices. What the objective deliberately preserves is a particular geometry of patch relations; feature coordinates, teacher quality, and compute requirements remain separate concerns.

## Source

[DINOv3, v1](https://arxiv.org/html/2508.10104v1), Sections 4.1–4.3, especially Equation 2. The matrices, rotation, and numerical checks above are original examples.
