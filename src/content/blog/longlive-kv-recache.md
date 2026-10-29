---
title: "KV Re-Caching at Prompt Changes in LongLive"
description: "Why changing a video prompt leaves stale semantics in the cache, and how re-encoding the same visual prefix addresses it."
date: 2025-10-11
tags: [Video generation, Attention]
draft: false
toc: true
---

A video generator has been producing a person walking through a room. The prompt changes: the person should now sit down. The next frame receives the new text, but it also attends to cached representations computed while the instruction was still “walking.” Updating the prompt embedding does not update those stored representations.

[LongLive](https://arxiv.org/abs/2509.22622v1) addresses this with **KV re-caching**: keep the already generated visual prefix, and recompute its attention cache under the new prompt. The video stays fixed while its hidden representations adapt to the current instruction.

## Why a visual cache contains text information

A self-attention KV cache stores projected hidden states from earlier positions. It does not generally store a prompt-independent encoding of the raw frames.

LongLive uses the diffusion-transformer backbone family introduced by [Peebles and Xie](https://arxiv.org/abs/2212.09748v2). In LongLive's video architecture, text cross-attention and causal video self-attention are interleaved. Once cross-attention has injected prompt information into a video hidden state, later operations can write that information into the keys and values used by self-attention.

For a cached visual position $j$ at an affected layer $\ell$, write

$$
k_j^\ell(c)=h_j^\ell(c)W_K^\ell,
\qquad
v_j^\ell(c)=h_j^\ell(c)W_V^\ell,
$$

where $c$ is the prompt conditioning, $h_j^\ell(c)$ is the hidden state reaching that layer's projections, and $W_K^\ell,W_V^\ell$ are learned matrices. The dependence on $c$ can remain even when the underlying video latents do not change.

Prompt dependence enters at specific layers. Keys computed before a prompt-dependent operation can stay unchanged; downstream cached states carry the old text conditioning and need recomputation.

Keeping the old cache offers useful visual continuity, but also carries those old hidden representations. Clearing it removes that influence along with the visual context. Re-caching is a third operation: retain the visual observations while updating their conditional representation.

## Recompute the representation of the same prefix

Let $z_{1:m}$ denote the already generated video latents available as context, and let $c_{\mathrm{old}}$ and $c_{\mathrm{new}}$ be the two prompts. Summarize the model's cache-building computation as

$$
C_{\mathrm{old}}
=\operatorname{Cache}_\theta(z_{1:m};c_{\mathrm{old}}).
$$

At a prompt boundary, LongLive rebuilds

$$
C_{\mathrm{new}}
=\operatorname{Cache}_\theta(z_{1:m};c_{\mathrm{new}}),
$$

then continues causal generation using $c_{\mathrm{new}}$ and $C_{\mathrm{new}}$. The prefix $z_{1:m}$ is unchanged. The operation neither regenerates the preceding video nor retroactively edits what the viewer has seen.

For one attention query $q$, the cache is consumed through

$$
\operatorname{Attn}(q,K,V)
=\operatorname{softmax}\!\left(\frac{qK^{\mathsf T}}{\sqrt{d_k}}\right)V.
$$

Both parts matter. Changed keys can change which earlier positions receive attention; changed values can change the information returned even if the weights stay similar. Merely overwriting a separate text cache leaves any prompt-dependent video keys and values untouched.

The resulting generation conditions are different:

$$
\begin{aligned}
\text{stale: }&\quad(c_{\mathrm{new}},C_{\mathrm{old}}),\\
\text{re-cached: }&\quad(c_{\mathrm{new}},C_{\mathrm{new}}).
\end{aligned}
$$

The first mixes a new current instruction with an old representation of the past. The second evaluates the visual history under the active instruction through a fresh conditional forward pass.

<figure class="figure-mechanism">
  <picture>
    <source media="(max-width: 620px)" srcset="/images/blog/longlive-kv-recache/conditional-cache-mobile.svg" width="360" height="310" />
    <img src="/images/blog/longlive-kv-recache/conditional-cache.svg" alt="The same visual prefix is encoded under the old and new prompts. Text cross-attention changes downstream video states, requiring their self-attention keys and values to be recomputed after a prompt switch." width="620" height="270" />
  </picture>
  <figcaption>Figure 1. Both branches encode the same generated prefix. Changing the prompt changes downstream video states, so their K,V must be recomputed. The diagram shows prompt-dependent layers; states computed before text conditioning may remain unchanged.</figcaption>
</figure>

LongLive also includes the operation in training, extending the generated-prefix rollout setting of [Self Forcing](https://arxiv.org/abs/2506.08009v1) to prompt switches. At a sampled prompt switch it re-caches once, continues the rollout with the updated cache, and supplies the new prompt to the distillation teacher. The student therefore trains on the post-switch condition it uses during generation.

## Two visual tokens and one changed condition

A two-token attention example isolates the effect of a changed condition on a fixed visual input.

Take fixed visual vectors

$$
u_1=(1,0),\qquad u_2=(0,1),
$$

and a scalar prompt condition $c$. Define the toy hidden representation

$$
h_j(c)=\big(u_{j,1},\ u_{j,2}+c\,u_{j,1}\big).
$$

Use identity key and value projections, so $k_j=v_j=h_j$. Under the old condition $c=0$,

$$
K_{\mathrm{old}}=V_{\mathrm{old}}
=\begin{bmatrix}1&0\\0&1\end{bmatrix}.
$$

Under the new condition $c=1$, re-caching the **same** $u_1,u_2$ gives

$$
K_{\mathrm{new}}=V_{\mathrm{new}}
=\begin{bmatrix}1&1\\0&1\end{bmatrix}.
$$

Hold the next query fixed at $q=(1,1)$, with key dimension $d_k=2$. This isolates the effect of rebuilding the prefix cache; a real model's current query may also depend on the new prompt.

The stale keys give equal logits $1/\sqrt{2}$ and equal attention weights $(0.5,0.5)$. The output is $(0.5,0.5)$. Re-cached keys give logits $(2/\sqrt{2},1/\sqrt{2})$, hence

$$
\alpha_1
=\frac{e^{2/\sqrt{2}}}
{e^{2/\sqrt{2}}+e^{1/\sqrt{2}}}
\approx0.6698,
\qquad \alpha_2\approx0.3302.
$$

With the refreshed values, the output becomes

$$
0.6698(1,1)+0.3302(0,1)
\approx(0.6698,1).
$$

Nothing in this calculation moved either visual vector. The changed output comes entirely from re-evaluating their hidden representations under the new condition. Conversely, changing a prompt variable in application code while retaining the old arrays would still return the stale result.

[Download the example](/code/longlive-kv-recache.py) to compute both attention outputs and verify that the visual inputs stay unchanged.

## Continuing from the visible past

The visible past supplies the starting point for the new instruction. A person who is walking needs a transition into sitting; appearance already established in earlier frames also shapes a coherent continuation.

Re-caching updates the hidden interpretation of those frames under the new prompt. Earlier instructions remain visible through the scene they produced, while refreshed keys and values give the next denoising step a representation conditioned on the current instruction.

The cost of a refresh grows with the visual context rebuilt at the switch. A bounded context window or retained sink frames limits that work. Triggering re-caching at prompt boundaries keeps it out of the per-frame generation loop.

The operation follows directly from LongLive's interleaved text conditioning and video attention: when the instruction changes, reuse the rendered past and rebuild the conditional state used to continue it.

## References

1. Shuai Yang et al. [LongLive: Real-time Interactive Long Video Generation](https://arxiv.org/html/2509.22622v1). 2025.
2. Xun Huang et al. [Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/abs/2506.08009v1). 2025.
3. William Peebles and Saining Xie. [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748v2). ICCV, 2023.
