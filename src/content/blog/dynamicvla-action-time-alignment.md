---
title: "Action-Time Alignment in DynamicVLA"
description: "Discarding expired actions and choosing the freshest overlapping chunk when inference takes real time."
date: 2026-02-14
tags: [Embodied AI, VLA, Robot control]
draft: false
toc: true
---

A robot observes a moving object, predicts six actions, and starts executing them. That description hides a timing question: how old is the first predicted action by the time inference finishes?

If producing the chunk takes two control periods, its first two actions refer to times that have already passed. Executing the chunk from its beginning creates a systematic delay. DynamicVLA's action-streaming rule [\[1\]](https://arxiv.org/abs/2601.22153v1) makes the intended execution time explicit: discard expired actions and, where chunks overlap, use the chunk based on the newer observation.

This note isolates that scheduling mechanism. It does not attempt to explain every component of the model or its dynamic-manipulation benchmark.

## A chunk has a clock

Predicting several future actions in one call is action chunking, as used in ACT [\[2\]](https://arxiv.org/abs/2304.13705v1). To make the timing explicit, use discrete control ticks $k=0,1,2,\ldots$, separated by $\Delta$ seconds. An observation at tick $t$ produces the chunk

$$
A_t=\{a_t^{(t)},a_{t+1}^{(t)},\ldots,a_{t+n}^{(t)}\}.
$$

The subscript is the intended execution tick. The superscript identifies the observation from which the chunk was predicted. This extra notation is useful because several predictions can refer to the same execution tick.

The chunk contains $n+1$ actions, following the paper's inclusive indexing. Suppose inference takes $m$ ticks. The result becomes available at $t+m$. Its actions for ticks before $t+m$ have expired. The valid suffix is

$$
\{a_{t+m}^{(t)},\ldots,a_{t+n}^{(t)}\}.
$$

The first valid action is therefore not necessarily the first array element. It is the element whose timestamp matches the current execution tick. The paper assumes a horizon longer than inference delay, $n>m$, to maintain useful overlap.

This rule corrects the indexing of execution. It does not make inference instantaneous. The chosen action is still conditioned on an observation from $m$ ticks earlier.

## Continuous inference creates overlap

DynamicVLA starts a new inference when the preceding inference finishes, while action execution runs concurrently. With fixed delay $m$, chunk source times are $0,m,2m,\ldots$.

This creates competing predictions. At tick $t+2m$, both $A_t$ and $A_{t+m}$ may contain an action for that tick. The newer chunk is based on a more recent observation, so the streaming rule replaces the unexecuted overlapping part of the older chunk.

The first paper version names this module **Latent-aware Action Streaming**. Its Section III-D states the two rules explicitly: remove the outdated prefix and prioritize the newer overlapping sequence. The mechanism is about action timestamps and observation freshness; the word “newer” must retain that meaning when implementing the queue.

For a general scheduler, let $r_t$ be the tick at which chunk $A_t$ is ready. Among chunks already available and containing an action for current tick $k$, choose

$$
t^*(k)=\max\{t:r_t\leq k,\ t\leq k\leq t+n\},
\qquad a_k=a_k^{(t^*(k))}.
$$

This is an explanatory timestamp-based formulation of the rule. With fixed latency it reduces to the paper's suffix-and-overwrite description. With variable latency it also clarifies that a late-arriving chunk from an older observation should not overwrite a chunk from a newer one merely because it arrived last.

## Six actions and two ticks of delay

Set the control rate to $20$ Hz, so $\Delta=50$ ms. Let inference take $m=2$ ticks, or $100$ ms, and let each chunk contain six actions, so $n=5$.

The first three inference calls behave as follows:

| Chunk | From tick | Action ticks | Ready | Expired |
| --- | ---: | --- | ---: | --- |
| $A_0$ | 0 | 0–5 | 2 | 0, 1 |
| $A_2$ | 2 | 2–7 | 4 | 2, 3 |
| $A_4$ | 4 | 4–9 | 6 | 4, 5 |

At tick 2, only $A_0$ is ready. Execute its action for tick 2, then its action for tick 3. At tick 4, $A_2$ becomes available and takes over. Its action for tick 4 replaces the still-unexecuted action for tick 4 in $A_0$.

<figure class="figure-plot">
  <picture>
    <source media="(max-width: 620px)" srcset="/images/blog/dynamicvla-action-time-alignment/timeline-mobile.svg" width="259" height="162" />
    <img src="/images/blog/dynamicvla-action-time-alignment/timeline.svg" alt="Three action chunks become ready after two ticks. Their first two control periods expire, their next two are executed, and the remaining predictions are superseded by the next chunk." loading="lazy" width="446" height="162" />
  </picture>
  <figcaption>Figure 1. Six-period action chunks with a two-tick inference delay. The first two predictions expire; the next two execute before a newer chunk takes over. Open markers indicate inference completion.</figcaption>
</figure>

For ticks 2 through 7, the source observation ticks are $0,0,2,2,4,4$. Observation age alternates between $100$ and $150$ ms. Dropping expired actions aligns the execution timestamp; it does not reduce that age to zero.

For scale, an object moving at $0.5$ m/s travels $5$ cm in $100$ ms. This is a constant-velocity illustration, not a DynamicVLA benchmark result. Whether the action suffix handles that motion depends on what the policy predicted about the future. A queue operation alone cannot recover a future state the model never anticipated. Related work such as VLASH [\[3\]](https://arxiv.org/abs/2512.01031v1) explicitly studies future-state-aware asynchronous inference, highlighting why execution timing and future-state conditioning are connected concerns.

## A minimal scheduler

The [runnable standard-library example](/code/dynamicvla-action-time-alignment.py) represents each chunk by its source tick, ready tick, and action timestamps. Selection can be written as:

~~~python
def action_for_tick(chunks, tick):
    candidates = [
        chunk for chunk in chunks
        if chunk.ready <= tick and tick in chunk.actions
    ]
    if not candidates:
        return None
    newest = max(candidates, key=lambda chunk: chunk.source)
    return newest.actions[tick]
~~~

The action values in the downloadable example are labels such as “action for tick 4 from observation 2.” They make scheduling visible without pretending to simulate a robot or measure task success. The script verifies the timeline above and includes a case in which an older chunk arrives after a newer one.

Real queues may implement the same decision by replacing future entries when each chunk arrives. Either representation must preserve the timestamps. Advancing an array pointer without accounting for inference time would silently shift every intended action into the future.

## What the rule leaves unresolved

**The horizon must cover the delay.** If a chunk is ready only after all of its action timestamps have passed, there is no valid suffix. In the fixed-delay discrete schedule above, the next chunk is ready at $t+2m$, while the current chunk ends at $t+n$. Gapless coverage after startup therefore requires $n\geq 2m-1$, stronger than merely retaining a valid suffix. Variable latency can still leave a gap. A deployed controller needs an explicit fallback when no suitable action is available; this scheduling rule does not define one.

**Prediction and execution must agree about time.** The suffix is useful only if the model's later entries mean actions for those later ticks. Discarding a prefix does not justify reinterpreting a sequence whose timing semantics are different. Sensor capture time, inference start time, and actuator dispatch time may also differ.

**Already executed actions cannot be replaced.** Overwriting applies to future queue entries. It cannot undo a command already sent to a slow actuator. Additional actuator delay may require another timing model beyond inference latency.

**Freshness is not continuity or safety.** A newer chunk can disagree sharply with the older one at the handoff. Always choosing it does not guarantee smooth forces, collision avoidance, or contact stability. Blending might address discontinuities, but introduces its own behavior and is not the rule analyzed here.

The useful contribution is narrower: a policy produces time-indexed predictions, inference consumes time, and the execution layer respects both facts. Removing expired entries and preferring fresher observations makes that temporal contract explicit.

## References

1. Haozhe Xie et al. [DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation](https://arxiv.org/abs/2601.22153v1). 2026.
2. Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705v1). 2023.
3. Jiaming Tang et al. [VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference](https://arxiv.org/abs/2512.01031v1). 2025.
