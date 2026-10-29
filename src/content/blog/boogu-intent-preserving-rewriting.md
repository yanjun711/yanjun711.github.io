---
title: "Intent Preservation in Boogu-Image’s Prompt Rewriter"
description: "When rewriting helps image generation: resolve implicit constraints without changing the picture the user requested."
date: 2026-08-08
tags: [Image generation, Multimodal reasoning]
draft: false
toc: true
---

Ask an image generator to “put the result of seven times eight above three red circles to the left of a blue square.” The sentence specifies a simple picture, but it combines arithmetic, object counting, layout, and exact text. A model can produce an attractive image while failing any one of those requirements.

[Boogu-Image](https://arxiv.org/abs/2607.13125v1) places a reasoning-capable prompt rewriter before generation. The rewriter resolves implicit instructions into concrete image details: “seven times eight” becomes “56,” while the requested objects, colors, and spatial relationships stay intact.

## Two different places for an error

A prompt rewriter creates an intermediate instruction $z$ from the user's request $x$. The generator then produces an image $I$ conditioned on $z$. A useful probabilistic abstraction is

$$
p(I\mid x)=\sum_z q_\psi(z\mid x)\,p_\phi(I\mid z).
$$

Here $q_\psi$ denotes the rewriter and $p_\phi$ the image generator. I use this abstraction to describe the two-stage interface. A deterministic rewrite is the special case where $q_\psi$ concentrates on one instruction.

The decomposition separates two failures. The rewrite can faithfully specify three circles, while the generator draws four. Or the rewrite can already ask for four, in which case even perfect instruction following produces the wrong picture. Better text-to-image fidelity cannot repair a requirement that was changed upstream.

Boogu's section 3.1.2 discusses rewriting as a reasoning process. Its examples include resolving indirectly described subjects, enumerating objects, and specifying literal strings and their placement. Appendix B.2.1 distinguishes text-to-image instructions from editing instructions according to whether a reference image is supplied. In each case, the rewriter translates intent into explicit generation conditions.

## Treat the request as a set of constraints

Consider this request:

> On a white background, put three red circles to the left of one blue square. The title should be the result of seven times eight.

Its requirements can be recorded as five predicates:

1. There are exactly three circles, and all are red.
2. There is exactly one square, and it is blue.
3. Every circle is to the left of the square.
4. The title is the literal string “56.”
5. The background is white.

For a hypothetical image or scene description $u$, write these tests as $C_1(u),\ldots,C_5(u)$. A faithful result satisfies

$$
C(u)=\bigwedge_{j=1}^{5}C_j(u).
$$

These predicates give us a compact way to compare the request with a rewrite: specified properties stay fixed, while unspecified details leave room for the generator to choose.

For example, circle radius, title font, and exact coordinates are open choices. The count, colors, background, and “left of” relation define the requested composition. A rewrite can clarify that composition while leaving the remaining visual details open.

## Comparing two rewrites

A compact rewrite is:

> Use a white background. Place exactly three separate red circles in a horizontal row on the left, with one blue square to their right. Keep all three circles entirely left of the square. Center the title “56” above the shapes.

The arithmetic is resolved once, before image generation. “Three separate” and the explicit enumeration of the object groups reduce the ambiguity of the count. Quotation marks delimit the text to render: the image should contain the digits, not an equation, a verbal explanation, or the phrase “the result of seven times eight.”

In this example rewrite, I chose a horizontal row and a centered title to give the shapes a concrete layout. Other arrangements can satisfy the same five requirements.

Now compare an embellished rewrite:

> Create a cinematic science poster titled “ORBIT,” with four glowing red circles surrounding a blue square on a navy background.

It supplies more visual detail, but violates the count, title, background, and potentially the relation. Word count and aesthetic specificity are poor proxies for intent preservation.

The [downloadable example](/code/boogu-prompt-constraints.py) compares two hand-specified scene graphs. The faithful scene passes all five predicates. A scene with four circles, a navy background, and the title “ORBIT,” while retaining the left relation, passes two: the square requirement and the relation requirement.

| Requirement | Faithful specification | Drifting specification |
| --- | --- | --- |
| Exactly three red circles | Pass | Fail: four circles |
| Exactly one blue square | Pass | Pass |
| Circles left of the square | Pass | Pass |
| Literal title “56” | Pass | Fail: “ORBIT” |
| White background | Pass | Fail: navy |


Its relation check is intentionally simple:

```python
left_of_square = (
    len(squares) == 1
    and bool(circles)
    and all(c["x"] < squares[0]["x"] for c in circles)
)
```

Coordinates here represent object centers in a graph of small, separated shapes. Testing whether entire shapes lie to the left without overlap would use their bounding boxes or masks.

## Clear prompts need little rewriting

Boogu argues that an already clear, conflict-free prompt should be left essentially unchanged. In functional notation, that aspiration is

$$
\mathcal R(x)\approx x
\quad\text{when }x\text{ already states the intended conditions clearly.}
$$

This near-identity behavior keeps already explicit instructions intact. Equivalent wordings can produce different image distributions, so rewriting is most useful where it resolves an actual ambiguity or an implicit calculation.

The right amount of rewriting is also generator-dependent. Models learn different caption styles, text conventions, and compositional priors. An enumeration that helps one model may be redundant or confusing for another. A faithful meaning and an effective conditioning format are related requirements, but neither implies the other.

## Separating rewrite errors from rendering errors

Rewriting resolves arithmetic and composition into text that the image model can use. The generator then renders the specified count, spatial relation, and literal digits. Separating these stages helps locate an error: did the rewrite change “three” to “four,” or did the generator draw four circles from a correct instruction?

Object-focused benchmarks such as [GenEval](https://arxiv.org/abs/2310.11513v1) and compositional evaluations such as [T2I-CompBench](https://arxiv.org/abs/2307.06350v1) motivate checking counts, colors, and relations separately instead of reducing everything to aesthetic preference.

Paired samples from the same generator make the effect of rewriting visible. Checking the rewritten text alongside the rendered objects, colors, and relations separates changes introduced by the rewriter from errors introduced during generation.

The rewriter's value comes from turning implicit reasoning into explicit conditions that survive through image generation.

## References

1. [Boogu-Image-0.1: Boosting Open-Source Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2607.13125v1). 2026.
2. Kaiyi Huang et al. [T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation](https://arxiv.org/abs/2307.06350v1). 2023.
3. Dhruba Ghosh, Hanna Hajishirzi, and Ludwig Schmidt. [GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment](https://arxiv.org/abs/2310.11513v1). 2023.
