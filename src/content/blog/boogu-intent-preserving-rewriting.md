---
title: "Intent Preservation in Boogu-Image’s Prompt Rewriter"
description: "When rewriting helps image generation: resolve implicit constraints without changing the picture the user requested."
date: 2026-08-08
tags: [Image generation, Multimodal reasoning]
draft: false
toc: true
---

Ask an image generator to “put the result of seven times eight above three red circles to the left of a blue square.” The sentence specifies a simple picture, but it combines arithmetic, object counting, layout, and exact text. A model can produce an attractive image while failing any one of those requirements.

[Boogu-Image](https://arxiv.org/abs/2607.13125v1) places a reasoning-capable prompt rewriter before generation. The narrow question here is what that rewriter should change. Converting “seven times eight” to “56” makes an implicit requirement explicit. Adding a fourth circle or replacing a white background with a dramatic gradient changes the requested image.

## Two different places for an error

A prompt rewriter creates an intermediate instruction $z$ from the user's request $x$. The generator then produces an image $I$ conditioned on $z$. A useful probabilistic abstraction is

$$
p(I\mid x)=\sum_z q_\psi(z\mid x)\,p_\phi(I\mid z).
$$

Here $q_\psi$ denotes the rewriter and $p_\phi$ the image generator. This equation describes the two-stage interface; it is **not a training objective or formal model introduced by the Boogu report**. A deterministic rewrite is the special case where $q_\psi$ concentrates on one instruction.

The decomposition separates two failures. The rewrite can faithfully specify three circles, while the generator draws four. Or the rewrite can already ask for four, in which case even perfect instruction following produces the wrong picture. Better text-to-image fidelity cannot repair a requirement that was changed upstream.

Boogu's section 3.1.2 discusses rewriting as a reasoning process. Its examples include resolving indirectly described subjects, enumerating objects, and specifying literal strings and their placement. Appendix B.2.1 also distinguishes text-to-image instructions from editing instructions according to whether a reference image is supplied. The mechanism considered here is the translation of intent into explicit generation conditions, not the rest of the agentic system.

## Treat the request as a set of constraints

Use this deliberately small request:

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

This constraint notation is an explanatory tool, not a claim that Boogu runs a symbolic constraint solver. It makes a distinction that prose can hide: requested properties must survive rewriting, while unspecified properties still leave room for a generator to choose.

For example, the request does not fix the circle radius, the title font, or the square's exact coordinates. A layout description may choose practical positions to express “left of,” but it should not accidentally add a competing subject, change the count, or overwrite the specified background. An image necessarily realizes unspecified details; the rewriter's job is to avoid making unnecessary choices into new mandatory conditions.

## One rewrite, checked line by line

A compact rewrite is:

> Use a white background. Place exactly three separate red circles in a horizontal row on the left, with one blue square to their right. Keep all three circles entirely left of the square. Center the title “56” above the shapes.

The arithmetic is resolved once, before image generation. “Three separate” and the explicit enumeration of the object groups reduce the ambiguity of the count. Quotation marks delimit the text to render: the image should contain the digits, not an equation, a verbal explanation, or the phrase “the result of seven times eight.”

The horizontal row and centered title are illustrative layout choices used to make this example inspectable. They are not mathematical consequences of the original request, nor measured outputs of the Boogu rewriter. In a real application, the appropriate amount of such layout detail depends on the user's tolerance for specification and the generator's conditioning conventions.

<figure>
  <img src="/images/blog/boogu-intent-preserving-rewriting/specification.svg" alt="A white scene with the title 56 above exactly three red circles on the left and one blue square on the right. A lower checklist names count, relation, literal text, and background." width="480" height="340" />
  <figcaption>Figure 1. A manually drawn realization of the five constraints. This SVG illustrates the intended specification; it is not an image generated by Boogu-Image.</figcaption>
</figure>

Now compare an embellished rewrite:

> Create a cinematic science poster titled “ORBIT,” with four glowing red circles surrounding a blue square on a navy background.

It supplies more visual detail, but violates the count, title, background, and potentially the relation. Word count and aesthetic specificity are poor proxies for intent preservation.

The [small downloadable audit](/code/boogu-prompt-constraints.py) makes this comparison precise on **hand-specified scene graphs**. Its faithful scene passes all five predicates. A drifting scene with four circles, a navy background, and the title “ORBIT,” while retaining the left relation, passes only two: the square requirement and the relation requirement. It does not run Boogu, inspect rendered pixels, or estimate a model's success rate.

Its relation check is intentionally simple:

```python
left_of_square = (
    len(squares) == 1
    and bool(circles)
    and all(c["x"] < squares[0]["x"] for c in circles)
)
```

Coordinates here represent object centers. They suffice for the toy graph, whose shapes are small and separated. If the actual requirement is that entire shapes lie to the left without overlap, a pixel evaluator needs object extents or masks as well. Even the definition of a predicate deserves care before it becomes an evaluation score.

## When doing less is the correct operation

Boogu argues that an already clear, conflict-free prompt should be left essentially unchanged. In functional notation, that aspiration is

$$
\mathcal R(x)\approx x
\quad\text{when }x\text{ already states the intended conditions clearly.}
$$

This is a **design criterion**, not a proved lower bound on generation quality. A reasoning model can still misread a prompt, and equivalent wordings can trigger different image distributions. Leaving the original wording available as a fallback is sensible, but the report does not establish a formal guarantee that every rewrite improves every output.

The right amount of rewriting is also generator-dependent. Models learn different caption styles, text conventions, and compositional priors. An enumeration that helps one model may be redundant or confusing for another. A faithful meaning and an effective conditioning format are related requirements, but neither implies the other.

## Where the interface remains fragile

Making a count explicit does not solve visual counting. Writing “56” does not ensure legible digits. Rewriting relocates some reasoning out of the image model; it does not eliminate the generator's limitations.

Ambiguous or contradictory requests create a different boundary. A rewriter cannot preserve two mutually exclusive requirements by silently selecting one and presenting it as certain. Practical systems need a policy for clarification or explicit assumptions. That policy is separate from the arithmetic and enumeration example above.

To evaluate rewriting, compare the original and rewritten instructions using a fixed generator and documented settings, then assess individual constraints across multiple samples. Matching random seeds can make paired comparisons more informative, but it does not establish universal improvement. A separate audit of the rewritten text helps locate whether a failure arose before or during rendering.

The useful role of this module is specific: turn implicit reasoning into explicit conditions while preserving the user's specified content. Its success should be judged by those conditions, rather than by how elaborate the rewritten prompt sounds.

## Source

[Boogu-Image-0.1: Boosting Open-Source Unified Multimodal Understanding and Generation, v1](https://arxiv.org/html/2607.13125v1), July 2026. Section 3.1.2 describes intent-preserving reasoning, counting, text rendering, and the identity-transformation design principle; Appendix B.2.1 describes the instruction-reasoner interface. The constraint formalization, SVG, and symbolic audit in this note are explanatory additions.
