---
title: "Aligned RoPE in MOVA’s Audio–Video Bridge"
description: "Why audio and video need a shared clock before cross-attention, and how MOVA rescales rotary positions."
date: 2026-03-14
tags: [Video generation, Multimodal learning]
draft: false
toc: true
---

A video token and an audio token can both have index 10 while describing different moments. Video and audio encoders compress time at different rates. If a model treats their sequence indices as a shared clock, its positional representation can favor a sound that occurs before or after the corresponding visible event.

[MOVA](https://arxiv.org/abs/2602.08794v1) addresses this mismatch inside the cross-attention bridge between its audio and video diffusion transformers. Its **Aligned RoPE** maps both token streams onto the same temporal grid. This note examines that coordinate change: what it preserves, what goes wrong without it, and what a shared clock cannot guarantee about generated content.

## Two indices, two clocks

Let $f_v$ be the number of video latent frames per second and $f_a$ the number of audio latent frames per second. These are the rates **after the respective VAEs**, not the original camera frame rate or audio waveform sampling rate. For zero-based indices $i$ and $j$, assume the streams begin at the same time. Their timestamps are

$$
t_v(i)=\frac{i}{f_v},\qquad t_a(j)=\frac{j}{f_a}.
$$

Equal physical times require $i/f_v=j/f_a$. Equal indices require $i=j$. Those are different statements whenever the latent rates differ.

This matters because cross-attention compares a query from one modality with keys from the other. The content projections can learn whether a visible impact and a sound belong together. Positional encoding supplies additional information about when they occur. An inconsistent clock makes that second signal disagree with the first.

The mismatch is not repaired by giving both streams the same number of channels. Channel dimension determines the size of the vectors being compared; latent frame rate determines the physical meaning of their temporal positions. Nor does padding the shorter sequence change which event each token represents.

<figure>
  <img src="/images/blog/mova-aligned-rope/clocks.svg" alt="Video tokens at two frames per second and audio tokens at eight frames per second. Video index one and audio index four describe the same half-second timestamp. Their aligned rotary positions are both four." width="480" height="340" />
  <figcaption>Figure 1. An illustrative pair of latent rates. Matching native indices would connect different moments; converting video positions to audio time units aligns the half-second event. These rates are chosen for the example, not reported MOVA settings.</figcaption>
</figure>

## Rotary attention measures relative phase

Consider one two-dimensional frequency pair in rotary position embedding. For an angular coordinate $\phi$, define

$$
R(\phi)=
\begin{bmatrix}
\cos\phi&-\sin\phi\\
\sin\phi&\cos\phi
\end{bmatrix}.
$$

Write the unrotated query and key as $q_m,k_m\in\mathbb R^2$, where $m$ identifies a frequency pair, and let $\theta_m>0$ be that pair's positional wavelength parameter. MOVA writes the rotations using position divided by $\theta_m$:

$$
\tilde q_m=R(p_v(i)/\theta_m)q_m,
\qquad
\tilde k_m=R(p_a(j)/\theta_m)k_m.
$$

The contribution of this pair to the attention dot product is

$$
\begin{aligned}
\tilde q_m^{\mathsf T}\tilde k_m
&=q_m^{\mathsf T}
R\!\left(\frac{p_a(j)-p_v(i)}{\theta_m}\right)k_m.
\end{aligned}
$$

The identity follows from $R(\alpha)^{\mathsf T}R(\beta)=R(\beta-\alpha)$. It explains why the choice of coordinates is consequential: the dot product depends on the difference between those coordinates. A full attention head sums contributions from multiple frequency pairs and applies its usual scaling and softmax. The two-dimensional expression isolates the temporal positional component rather than replacing the entire attention computation.

If native positions are used, $p_v(i)=i$ and $p_a(j)=j$, the phase depends on $j-i$. A simultaneous shift of the physical event by $\Delta$ seconds then changes that difference by $(f_a-f_v)\Delta$. The same relative timing can acquire a different positional phase merely because it occurred later in the clip.

## Converting video positions to audio units

MOVA's section 2.2 defines

$$
s=\frac{f_a}{f_v},\qquad
p_v(i)=si,\qquad p_a(j)=j.
$$

Now the difference has a physical interpretation:

$$
p_a(j)-p_v(i)
=j-\frac{f_a}{f_v}i
=f_a\bigl(t_a(j)-t_v(i)\bigr).
$$

Both operands are measured in audio latent-frame units. Tokens representing the same timestamp have zero temporal difference. Tokens separated by the same duration have the same relative positional phase regardless of their absolute location in the clip.

To check the second property, shift both timestamps by $\Delta$. Their new indices are $i+f_v\Delta$ and $j+f_a\Delta$. Substitution cancels the shift exactly. This is a statement about the rotary temporal coordinates under the assumed common origin; it does not claim that the entire generator is invariant to moving an event through a scene.

The scaling need not be an integer. Rotary embeddings accept continuous positions, so a rate ratio such as $2.5$ does not require rounding one stream onto discrete indices from the other. Rounding would introduce another timing error.

## A half-second example

Take illustrative rates $f_v=2$ and $f_a=8$. Video index $i=1$ and audio index $j=4$ both represent $0.5$ seconds. Set $q_m=k_m=(1,0)^{\mathsf T}$ and $\theta_m=2$. For this deliberately simple content pair, the positional dot product reduces to a cosine.

With native indices, the simultaneous pair scores $\cos((4-1)/2)\approx0.0707$. Audio index $j=1$, which actually occurs at $0.125$ seconds, scores $\cos(0)=1$. Under aligned positions, the simultaneous pair scores 1, and that earlier audio token scores approximately 0.0707.

This does not mean real attention must peak at the aligned timestamp: real query and key contents are learned, and meaningful sounds can precede or follow visible events. It demonstrates that positional coordinates should not systematically reverse the timing preference before content is considered.

The [runnable NumPy example](/code/mova-aligned-rope.py) verifies these values, the rotary dot-product identity, and invariance to a common physical-time shift. Its essential calculation is:

```python
video_position = (audio_rate / video_rate) * video_index
audio_position = audio_index
relative_phase = (audio_position - video_position) / theta
score = np.cos(relative_phase)  # Only for q = k = [1, 0].
```

## What alignment does not solve

A common coordinate system is an architectural prior, not a synchronization label. It does not tell the model which syllable belongs to which mouth movement, nor force the generated waveform and pixels to describe the same event. MOVA still needs bidirectional cross-modal exchange and aligned training data. Timing conventions remove an avoidable inconsistency from those interactions.

The derivation also assumes a common temporal origin. If preprocessing introduces an audio offset, different padding conventions, or unequal crop starts, the coordinates must include those offsets. Multiplying indices by the correct rate ratio does not correct a hidden offset. For irregular sampling or dropped frames, actual timestamps are more informative than a nominal constant-rate index.

RoPE is periodic in each frequency pair. Sharing physical units does not eliminate positional aliasing or establish a unique correspondence at every distance. Multiple frequencies and learned content help discriminate positions, while the model's training horizon still limits what can be inferred about extrapolation.

Finally, this mechanism operates on latent time. A waveform sample index, a video pixel frame index, and a compressed latent index are not interchangeable. An implementation should first establish each token's timestamp after the VAE, then apply the shared-coordinate mapping. That bookkeeping is the narrow but necessary job performed by Aligned RoPE.

## References

1. [MOVA: Towards Scalable and Synchronized Video-Audio Generation](https://arxiv.org/abs/2602.08794v1), February 2026 version. Section 2.2, especially “Aligned RoPE,” defines the latent-rate conversion and rotary positions.
2. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), 2021. The relative-rotation identity used in the derivation.
