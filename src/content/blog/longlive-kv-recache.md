---
title: "KV Re-Caching at Prompt Changes in LongLive"
description: "Why changing a video prompt leaves stale semantics in the cache, and how re-encoding the same visual prefix addresses it."
date: 2025-10-11
tags: [Video generation, Attention]
draft: false
toc: true
---

A video generator has been producing a person walking through a room. The prompt changes: the person should now sit down. The next frame receives the new text, but it also attends to cached representations computed while the instruction was still “walking.” Updating the prompt embedding does not update those stored representations.

[LongLive](https://arxiv.org/abs/2509.22622v1) addresses this with **KV re-caching**: keep the already generated visual prefix, and recompute its attention cache under the new prompt. This post examines that operation alone. The key distinction is between keeping the past video fixed and keeping its hidden representations fixed.

## Why a visual cache contains text information

A self-attention KV cache stores projected hidden states from earlier positions. It does not generally store a prompt-independent encoding of the raw frames.

In the video diffusion transformer considered by LongLive, text cross-attention and causal video self-attention are interleaved. Once cross-attention has injected prompt information into a video hidden state, later operations can write that information into the keys and values used by self-attention.

For a cached visual position $j$ at an affected layer $\ell$, write

$$
k_j^\ell(c)=h_j^\ell(c)W_K^\ell,
\qquad
v_j^\ell(c)=h_j^\ell(c)W_V^\ell,
$$

where $c$ is the prompt conditioning, $h_j^\ell(c)$ is the hidden state reaching that layer's projections, and $W_K^\ell,W_V^\ell$ are learned matrices. The dependence on $c$ can remain even when the underlying video latents do not change.

This does not imply that every layer's cache must change. A key computed before any prompt-dependent operation could be unaffected. The relevant point is that downstream cached states can depend on the old prompt, so replacing the current text input is not sufficient to recompute the complete conditioning state.

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

The first mixes a new current instruction with an old representation of the past. The second evaluates the available visual history under the active instruction. This is a conditional forward pass through the model, not an algebraic subtraction of an “old prompt vector” from projected keys.

LongLive also includes the operation in training. At a sampled prompt switch it re-caches once, continues the rollout with the updated cache, and supplies the new prompt to the distillation teacher. The student therefore encounters the post-switch condition it will use during generation. An inference-only cache patch would not by itself provide that training experience.

## Two visual tokens and one changed condition

A small attention example can isolate the effect. This is a deliberately simplified conditional representation, not the architecture of the trained video model.

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

<figure>
  <picture>
    <source media="(max-width: 620px)" srcset="/images/blog/longlive-kv-recache/conditional-cache-mobile.svg" />
    <img src="/images/blog/longlive-kv-recache/conditional-cache.svg" alt="In a two-token toy example, re-caching changes attention weights from equal weights to approximately 0.67 and 0.33, and changes the returned value despite unchanged visual inputs." width="800" height="370" />
  </picture>
  <figcaption>Figure 1. Analytical outputs from the toy conditional cache. The query and visual inputs are fixed. These bars illustrate a representation change, not measured prompt adherence or video quality.</figcaption>
</figure>

Nothing in this calculation moved either visual vector. The changed output comes entirely from re-evaluating their hidden representations under the new condition. Conversely, changing a prompt variable in application code while retaining the old arrays would still return the stale result.

[Download the checked example](/code/longlive-kv-recache.py). It computes both attention operations and asserts that the visual inputs remain unchanged. There is no meaningful “softmax over an empty prefix” comparison here: a real generator with a cleared cache follows a different context path, which this small example does not model.

## Boundaries of a cache refresh

Re-caching does not guarantee that a new instruction is physically compatible with the visible past. A person cannot become seated instantly while preserving ordinary motion, and a new prompt may contradict appearance already established in the video. The generated frames still constrain a coherent continuation.

Nor does the recomputation mathematically erase every trace of the old semantics. The previous instruction may already have changed the visible scene, and the model's learned conditional representation can retain related information. What is removed is the avoidable use of hidden states computed under the old prompt. Semantic adherence and smooth transitions remain empirical properties of the trained generator.

The cost also depends on the visual context rebuilt at a switch. A bounded context window or retained sink frames can keep the amount of work manageable, but recomputing arbitrary full history is not a constant-cost operation. Re-caching is triggered at prompt boundaries rather than at every generated frame; this limits its frequency, not the cost of an individual forward pass independently of context length.

Finally, the argument relies on a model that can re-encode a fixed visual prefix under an external condition. It should not be transferred mechanically to a text-only causal language model by rewriting earlier instructions while assuming that the old tokens retain identical meaning. The position and role of conditioning in the architecture determine which cache entries have become stale.

For this video setting, the useful intervention is precise: when the instruction changes, reuse the rendered past and rebuild the conditional state used to continue it.

## Source

[LongLive: Real-time Interactive Long Video Generation, v1](https://arxiv.org/html/2509.22622v1), Section 3.1, Section 4.4, and Appendix Algorithm 2. The two-token attention example and plot are original illustrations.
