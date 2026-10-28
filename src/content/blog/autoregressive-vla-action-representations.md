---
title: "Autoregressive VLAs: From Action Bins to Shared Action Codes"
description: "A history of autoregressive robot policies, a close reading of G0.5, and a guide to the action representations behind VLA models."
date: 2026-10-28
tags: [Embodied AI, VLA, Robot control]
draft: false
toc: true
---

A vision-language-action model sees a scene, reads an instruction, and sends commands to a robot. The difficult interface is between the language model's discrete symbols and the robot's continuous, time-sensitive motion. How should an action become something a next-token model can predict without making control too slow or losing the movement that matters?

This question has shaped the development of **autoregressive (AR) VLAs**. The history is not simply a succession of larger vision-language models: it is also a progression from individual quantized coordinates to compressed trajectories and learned, structured action codes. I use [G0.5](https://arxiv.org/abs/2608.11739v1) as a recent, strong example of a model that actually generates reasoning and action tokens with one decoder. Its reported results are impressive, but “best” needs a benchmark and a protocol; there is no universal ranking across robot bodies, data, latency budgets, and tasks.

## What counts as autoregressive action generation?

Let $o_t$ be the current images, $l$ an instruction, $s_t$ the robot state, and $A_t=(a_t,\ldots,a_{t+H-1})$ a future action chunk. An action tokenizer $E$ maps that chunk to discrete codes $z_{1:M}=E(A_t)$. A token-generating VLA models

$$
p_\theta(z_{1:M}\mid o_t,l,s_t)
=\prod_{i=1}^{M}p_\theta(z_i\mid o_t,l,s_t,z_{<i}),
\qquad \hat A_t=D(z_{1:M}),
$$

where $D$ turns the predicted codes back into commands. The next action token depends on the previously generated tokens. The serial length $M$ therefore matters to both the training target and inference latency.

Three distinctions prevent confusion. **Discrete actions do not automatically imply serial action decoding**: [RT-1](https://arxiv.org/abs/2212.06817v1) discretizes its action dimensions but explicitly removes autoregressive action prediction for speed. **An autoregressive VLM does not automatically imply autoregressive actions**: a VLM may encode the scene and pass its hidden states to a separate flow-matching or diffusion action head, as in [$\pi_0$](https://arxiv.org/abs/2410.24164v1) and the continuous-control path of [MolmoAct2](https://arxiv.org/abs/2605.02881v1). Finally, a system can be autoregressive *between chunks* while denoising each chunk with a different mechanism. The level at which the dependency is imposed should be stated explicitly.

## A short history of the interface

| Stage | Representative work | Action interface | What changed |
| --- | --- | --- | --- |
| 2022: discretized control | [RT-1](https://arxiv.org/abs/2212.06817v1) | Each action dimension mapped to one of 256 bins | Established a scalable vision-and-language-conditioned discrete robot controller; action outputs were designed for real-time inference. |
| 2023: actions as language tokens | [RT-2](https://arxiv.org/abs/2307.15818v1) | Quantized end-effector, gripper, and termination values serialized as tokens | A pretrained vision-language model could generate robot commands through its text-token interface. |
| 2024: open next-token VLA | [OpenVLA](https://arxiv.org/abs/2406.09246v1) | One 256-bin token per continuous action dimension | Made the RT-2-style recipe available in an open 7B model trained on diverse robot data. |
| 2025: compressed action chunks | [FAST](https://arxiv.org/abs/2501.09747v1) and $\pi_0$-FAST | DCT coefficients, quantization, and byte-pair encoding (BPE) | Shortened high-frequency action sequences so next-token training could model dexterous motion. |
| 2025–26: learned codes and structured streams | [ActionCodec](https://arxiv.org/abs/2602.15397v2), [NAC](https://arxiv.org/abs/2606.21372v2), [G0.5](https://arxiv.org/abs/2608.11739v1) | Learned discrete codes for whole chunks or body parts | Made compression, token predictability, and cross-robot structure explicit design concerns. |

This is a **development path**, not a claim that each later paper supersedes every earlier one. RT-1 is a precursor in discrete representation rather than an instance of serial action-token decoding. The continuous-action branch also developed in parallel: a flow or diffusion head can represent a full trajectory without asking the VLM to emit every motor value as text-like tokens.

The pressure that drove compression is easy to see. For an illustrative $H=16$ step chunk with $D=7$ continuous values per step, per-coordinate binning requires $HD=112$ action tokens. If a compact codec represents the same chunk with $M$ codes, the serial decoding work depends on $M$ instead. This example counts symbols; it does not assert that any particular codec uses a particular $M$ or that runtime falls in the same proportion. Image processing, prompt prefill, code decoding, and robot communication still take time.

FAST's concrete pipeline first normalizes action dimensions, applies a discrete cosine transform (DCT) along time, quantizes the coefficients, flattens them, and uses BPE to compress the resulting sequence. Smooth trajectories often concentrate information in lower temporal frequencies. The paper argues that predicting every nearly identical high-rate sample can reward copying the preceding token rather than learning the motion's underlying shape. FAST changes the learning target as well as its length. Its universal FAST+ tokenizer and $\pi_0$-FAST experiments showed that this route could make AR action generation competitive with flow-based VLAs on the evaluated settings [\[4\]](https://arxiv.org/abs/2501.09747v1).

<figure class="figure-paper">
  <a href="/images/blog/autoregressive-vla-action-representations/fast-tokenizer.webp" aria-label="Open the FAST tokenizer figure at full size">
    <img src="/images/blog/autoregressive-vla-action-representations/fast-tokenizer.webp" alt="FAST converts a normalized action chunk into frequency coefficients with a DCT, quantizes them, flattens low-frequency coefficients first, then uses BPE to form a shorter sequence of action tokens." loading="lazy" decoding="async" width="2102" height="748" />
  </a>
  <figcaption>Figure 1. FAST compresses a trajectory before autoregressive prediction. The order matters: transform across time, quantize coefficients, then merge frequent symbol sequences with BPE. Select the image to inspect its labels at full size.<small class="figure-source">Source: <a href="https://arxiv.org/abs/2501.09747v1">FAST</a>, original Fig. 4, cropped; <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.</small></figcaption>
</figure>

## A close reading of G0.5

G0.5 is a useful 2026 case study because the action path is unambiguous. Its main policy initializes from a Qwen3.5 2B vision-language model. Multi-view images, embodiment identity, instruction, and proprioceptive state form the conditioning prefix. One decoder then emits optional reasoning fields—such as a subtask, bounding box, or motion trace—followed by discrete action codes. A codec decodes those codes into continuous commands. The training loss is next-token cross-entropy over the generated reasoning and action span, rather than a separate action-regression objective [\[8\]](https://arxiv.org/abs/2608.11739v1).

<figure class="figure-paper">
  <a href="/images/blog/autoregressive-vla-action-representations/g05-stream.webp" aria-label="Open the G0.5 model diagram at full size">
    <img src="/images/blog/autoregressive-vla-action-representations/g05-stream.webp" alt="G0.5 conditions one autoregressive VLM on multiview images, instruction, robot state and embodiment; it emits optional reasoning followed by body-part action codes, which an ActionCodec converts to continuous actions." loading="lazy" decoding="async" width="1344" height="660" />
  </a>
  <figcaption>Figure 2. In G0.5, optional subtask, box, and trace tokens precede the action codes in one autoregressive stream. The ActionCodec maps the generated codes to continuous commands.<small class="figure-source">Source: <a href="https://arxiv.org/abs/2608.11739v1">G0.5</a>, left panel of original Fig. 1, cropped; <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>.</small></figcaption>
</figure>

The distinctive piece is its **cross-embodiment ActionCodec**. G0.5 divides heterogeneous robots into semantic motion groups such as left arm, right arm, grippers, and lower body. It pads each group to a common dimension, learns a residual vector quantizer over grouped action chunks, and inserts group markers before the corresponding codes. Each active group contributes eight action codes per residual round; only groups involved in the current behavior need to be generated. Its tokenizer training also uses a temporal contrastive objective intended to keep nearby motions closer in the token representation [\[8\]](https://arxiv.org/abs/2608.11739v1). The codec is learned separately; the **policy** itself is trained with a shared next-token objective over reasoning and actions.

<figure class="figure-paper">
  <a href="/images/blog/autoregressive-vla-action-representations/g05-codec.webp" aria-label="Open the G0.5 structured action tokenizer diagram at full size">
    <img src="/images/blog/autoregressive-vla-action-representations/g05-codec.webp" alt="G0.5 identifies active robot body parts, encodes action chunks with a temporal contrastive objective and residual vector quantizer, then emits part-labeled codes that a decoder reconstructs as continuous motions." loading="lazy" decoding="async" width="2064" height="622" />
  </a>
  <figcaption>Figure 3. G0.5 keeps active body parts explicit and refines their action codes through residual quantization. Inactive groups do not require action tokens.<small class="figure-source">Source: <a href="https://arxiv.org/abs/2608.11739v1">G0.5</a>, original Fig. 3, cropped; <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>.</small></figcaption>
</figure>

That structure addresses a specific failure of flattening every robot into one long vector. A single-arm demonstration should not have to predict inactive right-arm or base coordinates merely because a universal output vector has slots for them. Group markers make the meaning of each code span explicit. Residual rounds let additional codes refine a coarse reconstruction. They do not remove the need for a reliable decoder: wrong codes still become physical commands.

The paper reports 98.9% average success on LIBERO, 93.3% average across clean and randomized RoboTwin 2.0, and 87.3% on four Bridge-SimplerEnv tasks. Its LIBERO table compares the 98.9% result with 98.7% for Xiaomi-Robotics-0 and 98.5% for two other listed baselines [\[8\]](https://arxiv.org/abs/2608.11739v1). These are **paper-reported results under the paper's own training and evaluation protocols**. They support treating G0.5 as a strong AR VLA example, not as proof that it is the best controller on every robot or that AR decoding alone caused the gains. Pretraining data, model backbone, post-training, prompts, and embodiment coverage differ across systems. The paper also includes an optional flow-matching head for comparison; the reported scores above describe its main AR policy unless a particular experiment says otherwise.

One appealing property of the shared stream is that a language instruction can influence the same decoder that chooses the action codes, and intermediate reasoning can be generated before control. Its cost is serial generation: extra reasoning tokens and residual code rounds add latency. A box or motion trace is also a *predicted* intermediate, not a verified geometric fact. For fast contact-rich control, the practical question remains whether better grounding pays for its added delay.

## Action representations: four separate choices

“Action representation” often refers to several decisions at once. They should be separated before comparing models.

| Choice | Common forms | What it changes |
| --- | --- | --- |
| **Physical command** | Joint position or velocity; end-effector pose or delta pose; gripper state; base motion | Determines what the robot controller can execute and what geometry the policy must learn. |
| **Time unit** | One action; a fixed chunk; a waypoint or spline; a skill/subtask | Determines control horizon, replanning frequency, and how much motion one prediction commits to. |
| **Numerical representation** | Continuous vector; uniform or adaptive bins; DCT+BPE; VQ/RVQ or neural codec | Determines precision, token count, compression artifacts, and the policy's prediction target. |
| **Generator** | Per-coordinate regression; autoregressive token decoder; flow matching; diffusion | Determines the training objective and whether generation is serial over tokens or iterative over a continuous chunk. |

These axes are not interchangeable. An end-effector delta can be represented by bins, by a learned code, or as a continuous vector. A chunk can be emitted as AR tokens or generated by a flow head. “Uses action tokens” says nothing by itself about whether those tokens are physical coordinates, frequency coefficients, or opaque learned codes.

### Bins: simple and direct

For a normalized scalar $a\in[-1,1]$, a $K$-bin tokenizer can assign

$$
q(a)=\operatorname{clip}\!\left(
\left\lfloor\frac{K(a+1)}{2}\right\rfloor,0,K-1
\right).
$$

With $K=256$, each dimension has one categorical target. This is easy to train and inspect. Its costs are fixed resolution, clipping outside the chosen range, and one prediction per dimension per step. Uniform binning also gives equal precision to regions the robot may visit very unequally. [SpatialVLA](https://arxiv.org/abs/2501.15830v5) explores adaptive spatial action grids, illustrating that the geometry of the bins can be changed while keeping a discrete interface.

### Frequency codes: compact trajectories

FAST replaces a sequence of time-domain samples with quantized frequency coefficients and BPE tokens. It can be short when motion is temporally smooth, but sharp contacts and brief corrective motions may require higher-frequency coefficients. Compression is only useful if the decoded trajectory retains the details needed by the controller. The code count is variable, and lower token count does not by itself certify lower task error.

### Learned codes: compress what data makes useful

VQ-style tokenizers learn an encoder, a codebook, and a decoder for action chunks. Residual VQ (RVQ) uses successive codebooks to refine a reconstruction. Neural codecs such as [NAC](https://arxiv.org/abs/2606.21372v2) try to obtain compact sequences with high-fidelity reconstruction; G0.5 additionally structures them by robot body part. [Ordered Action Tokenization](https://arxiv.org/abs/2607.21670v1) trains prefixes to produce increasingly refined chunks, allowing an early stop when time is limited.

The token sequence is now less physically transparent than a list of coordinates. Reconstruction error is necessary to inspect, but it is insufficient as a selection criterion: a policy must also *predict* the right codes from images, and mistakes in code space must decode to tolerable motions. [ActionCodec](https://arxiv.org/abs/2602.15397v2) studies properties such as temporal token overlap and token independence; [ProAct](https://arxiv.org/abs/2610.09170v1) directly trains for predictability and robustness in addition to reconstruction. These works explain why the tokenizer is part of the policy design, not just an offline compression utility.

### Continuous chunks: a different generation contract

Flow-matching and diffusion VLAs generate a continuous chunk through iterative refinement. They avoid a discrete action-code bottleneck, but require a separate action-generation mechanism and several numerical steps. MolmoAct2 illustrates a hybrid: its VLM can produce discrete reasoning or depth tokens, while its continuous-control path uses a flow-matching action expert conditioned on the VLM cache [\[12\]](https://arxiv.org/abs/2605.02881v1). Calling that path “autoregressive action-token decoding” would obscure the actual computation.

## What to compare before choosing a representation

A fair action interface comparison should fix the robot, data, backbone, action horizon, control rate, and evaluation rollouts where possible. Then measure at least four things: decoded trajectory error, closed-loop task success, inference-to-execution latency, and failure behavior under token mistakes or distribution shift. The open-loop metric can mislead. A codec may reconstruct recorded demonstrations accurately while making nearby plausible actions hard for the policy to express; conversely, a slightly less exact codec can yield easier tokens and better closed-loop control [\[10\]](https://arxiv.org/abs/2609.25820v1) [\[11\]](https://arxiv.org/abs/2610.09170v1).

The development arc is therefore a change in what an action token is asked to carry. RT-2 and OpenVLA make a token a quantized coordinate. FAST makes it part of a compressed temporal signal. Learned codecs make it a reusable motion code, and G0.5 gives those codes explicit body-part structure inside a shared reasoning-and-action stream. The best representation is the one whose decoded actions are controllable, whose tokens the model can predict, and whose generation fits the robot's timing budget.

## References

1. Anthony Brohan et al. [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817v1). 2022.
2. Anthony Brohan et al. [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818v1). 2023.
3. Moo Jin Kim et al. [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246v1). 2024.
4. Pertsch et al. [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2501.09747v1). 2025.
5. Kevin Black et al. [$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164v1). 2024.
6. [SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model](https://arxiv.org/abs/2501.15830v5). 2025.
7. [ActionCodec: What Makes for Good Action Tokenizers](https://arxiv.org/abs/2602.15397v2). 2026.
8. [G0.5: One Autoregressive Stream for Robot Reasoning and Action](https://arxiv.org/abs/2608.11739v1). 2026.
9. [NAC: Neural Action Codec for Vision-Language-Action Models](https://arxiv.org/abs/2606.21372v2). 2026.
10. [Beyond Reconstruction Error: Analytical and Data-Driven Action Tokenization for Autoregressive Vision-Language-Action Models](https://arxiv.org/abs/2609.25820v1). 2026.
11. [Beyond Reconstruction: What Matters in Action Tokenization for Robot Policies?](https://arxiv.org/abs/2610.09170v1). 2026.
12. [MolmoAct2: Action Reasoning Models for Real-world Deployment](https://arxiv.org/abs/2605.02881v1). 2026.
13. [Ordered Action Tokens for Visuomotor Policy Learning](https://arxiv.org/abs/2607.21670v1). 2026.
