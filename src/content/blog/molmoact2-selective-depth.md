---
title: "Selective Depth-Token Updates in MolmoAct2-Think"
description: "How an RGB change mask decides which depth tokens to regenerate and which cached tokens to replay before acting."
date: 2026-06-13
tags: [Embodied intelligence, Autoregressive models]
draft: false
toc: true
---

A robot taking another control step often sees much of the same scene. If its policy predicts an intermediate depth representation before choosing an action, should it generate every depth token again?

[MolmoAct2-Think](https://arxiv.org/abs/2605.02881v1) answers with a selective update: compare corresponding RGB patches, regenerate depth codes where the image changed, and replay cached codes elsewhere. The useful detail is that replayed tokens still become part of the model's depth prefix. The method changes how that prefix is produced while keeping a complete representation available for action generation.

## Why the depth prefix can be redundant

In the original May 2026 paper, the depth representation is discrete. During data preparation, [Depth Anything V2](https://arxiv.org/abs/2406.09414v1) estimates a depth map from each RGB frame. The trained depth VQ-VAE used by the earlier [MolmoAct](https://arxiv.org/abs/2508.07917v1) encodes a $320\times320$ depth image with downsampling factor 32 into a $10\times10$ grid. Each of the 100 cells contains a codebook index from $\{0,\ldots,127\}$.

These indices are flattened in raster order. They are **codebook entries**, not scalar distances in meters. A larger code index does not mean a surface is farther away.

At inference, the policy can generate these codes as a depth prefix and then generate the action from the resulting state. On an uncached frame, that means predicting 100 depth-code tokens, in addition to the boundary tokens. Repeating this prediction for static parts of the image creates an opportunity for reuse.

The question is not merely whether a cached code is available. It is whether there is a cheap enough signal for deciding which codes should be reconsidered now. MolmoAct2-Think uses RGB patch similarity as that signal, rather than predicting a full new depth map first and comparing the two depth maps afterwards.

## The mask and the buffer

Resize the RGB image to $320\times320$ and divide it into the corresponding $10\times10$ grid. Each cell contains a $32\times32$ RGB patch. Flatten that patch to a vector $x_{t,i}$ for frame $t$ and cell $i$.

The update mask is

$$
m_{t,i}=\mathbf{1}\left[
\frac{x_{t,i}^{\mathsf T}x_{t-1,i}}
{\lVert x_{t,i}\rVert\lVert x_{t-1,i}\rVert}
<0.996
\right].
$$

A similarity below the threshold marks the cell for an update. The mask compares the same grid position in consecutive images; it does not track the same physical surface through motion.

Let $d_{t,i}$ denote a full depth-code annotation and $b_{t,i}$ the training buffer. Equation 10 defines

$$
b_{t,i}=\begin{cases}
d_{t,i},&m_{t,i}=1,\\
b_{t-1,i},&m_{t,i}=0.
\end{cases}
$$

For an episode's first frame, all cells are updated and $b_1=d_1$. At subsequent frames, unchanged cells inherit the preceding buffer value. The training target is this **buffer**, not an independently refreshed full annotation at every position. Training therefore exposes the model to the representation its adaptive inference procedure will maintain.

At inference, the model generates updated cells using argmax over depth-token logits. Unchanged cells inherit the previous **predicted** buffer. Training thus builds the buffer from depth annotations, while inference builds it from the model's own predictions.


## Four cells, two new predictions

For a compact scheduling example, flatten a $2\times2$ grid in row-major order. Suppose the cached codes are

$$
b_{t-1}=[12,\;33,\;7,\;80].
$$

For the first and third patches, choose toy RGB vectors whose cosine similarity with the previous frame is $\cos(0.05)\approx0.99875$. For the second and fourth, choose similarity $\cos(0.10)\approx0.99500$. The threshold gives

$$
m_t=[0,\;1,\;0,\;1].
$$

<figure class="figure-mechanism">
  <picture>
    <source media="(max-width: 620px)" srcset="/images/blog/molmoact2-selective-depth/replay-mobile.svg" width="360" height="312" />
    <img src="/images/blog/molmoact2-selective-depth/replay.svg" alt="Cached codes twelve, thirty-three, seven, and eighty combine with update mask zero, one, zero, one. The current causal prefix contains replayed code twelve, predicted code thirty-five, replayed code seven, and predicted code eighty-one. All four codes are consumed in the current context; only two are newly predicted." loading="lazy" width="620" height="270" />
  </picture>
  <figcaption>Figure 1. A four-cell teaching example. Zeros replay cached code IDs into the current prefix; ones predict updated codes in raster order. All four codes condition the action; only two are newly predicted.</figcaption>
</figure>

The decoder handles the cells in order:

1. Replay code 12 as a known input.
2. Predict a replacement for code 33; suppose the prediction is 35.
3. Replay code 7 as a known input.
4. Predict a replacement for code 80; suppose the prediction is 81.

The complete prefix becomes $[12,35,7,81]$. There are two new code predictions, yet all four codes are consumed as part of the conditioning sequence. The second prediction is conditioned on the earlier replayed and newly generated codes, in addition to the prompt and current images.

Now consider a mask $[0,0,1,1]$. The first two known codes can be replayed as one consecutive span. The changed cells still require token-by-token generation because each prediction conditions on the preceding prefix. Identical counts of changed cells can therefore give different replay schedules, depending on where unchanged cells occur.

[Run the NumPy example](/code/molmoact2-selective-depth.py) to reproduce the cosine decisions, buffer update, and replay spans using fixed replacement codes. It also checks initialization with an empty cache.

## What replay saves

Once a depth cache exists, the inference loop first prefills the prompt and current images. It computes the same RGB mask used for the training representation. Changed cells are generated autoregressively; consecutive unchanged cells are replayed together. Once all 100 depth positions are filled, the model emits the depth end marker and generates an action.

For the continuous-control path described in section 5.3, the action expert receives the VLM's key-value cache for the prompt and the filled depth prefix. The intermediate representation is therefore still complete at the interface to the action expert.

Known-token replay replaces repeated serial token decisions with larger processing steps for unchanged spans. Replayed tokens still enter the new prefix and participate in attention. Total runtime also includes current-image prefill, mask computation, regenerated tokens, and action generation.

With 20 changed cells, the model makes 20 new depth-code predictions and replays 80 known codes. The resulting speedup depends on how those 80 codes group into replay spans and how much time the other stages take.

The method also caches token identities rather than treating last frame's entire transformer state as a valid prefix for a new observation. Replaying codes under the current prompt lets them participate in the current depth-conditioned state. That distinction matters whenever the image context changes even though a particular cell's code is reused.

## How the RGB mask responds to scene changes

RGB cosine similarity acts as a proxy for depth changes. A textureless surface can move in depth with little appearance change, leaving its cached code in place. Occlusions can change only part of a patch, making mask decisions depend on how that boundary falls within the grid.

Conversely, illumination changes can alter the RGB comparison while geometry stays fixed. For nonzero vectors and positive $\alpha$,

$$
\cos(\alpha x,x)=1.
$$

Thus a perfectly uniform multiplicative brightness change is invisible to this idealized cosine test. Shadows, channel-dependent changes, clipping, or an additive brightness shift need not have that property. The exact response depends on the patch values after preprocessing.

Camera motion can mark many cells as changed because the comparison uses fixed image locations. In that case the opportunity for reuse shrinks, even if the physical scene is static. The cache rule itself supplies no motion compensation.

The buffer preserves earlier predictions until the mask requests an update. A wrong code can therefore persist across several frames with similar RGB patches. Even when an earlier regenerated code changes the autoregressive context, a later unmarked cell replays its cached value.

A new episode starts with an empty cache and generates all 100 depth codes. Subsequent frames use the previous image and predicted buffer for selective updates.

## References

1. [MolmoAct2: Action Reasoning Models for Real-world Deployment](https://arxiv.org/abs/2605.02881v1). 2026.
2. [Depth Anything V2](https://arxiv.org/abs/2406.09414v1). 2024.
3. [MolmoAct: Action Reasoning Models that can Reason in Space](https://arxiv.org/abs/2508.07917v1). 2025.
