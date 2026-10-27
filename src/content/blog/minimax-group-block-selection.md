---
title: "Group-Specific Block Selection in MiniMax Sparse Attention"
description: "How a lightweight indexer gives each GQA group its own attention blocks, and what happens to the probability mass it leaves out."
date: 2026-07-11
tags: [LLM, Efficient attention]
draft: false
toc: true
---

Suppose a query near the end of a long document needs an early definition and the paragraph immediately before it. Another attention head may need a different definition. A sparse attention layer has two decisions to make: which past tokens to read, and how many different selection patterns it can afford to maintain.

[MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392v1), or MSA, makes its selection at the level of **blocks and grouped-query attention groups**. Heads within one group share a block list; different groups can read different parts of the context. This note follows that selection rule through an eight-token example, then examines what restricted attention loses.

## Which heads should share a reading list?

In [grouped-query attention](https://arxiv.org/abs/2305.13245v1), or GQA, several query heads share one key head and one value head. Let there be $H_q$ query heads and $H_{kv}$ key–value heads, so each group contains $G=H_q/H_{kv}$ query heads. Sharing keys and values reduces cache size, but it does not make the query heads identical: they still have distinct queries and attention distributions.

A single global sparse mask would make all groups read the same locations. A separate mask for every query head offers more freedom, but introduces more selection and gathering work. MSA chooses an intermediate granularity: one mask per query position and GQA group.

Its small index branch has one index query per group and one index key shared across all groups. For query position $i$, key position $j$, and group $g$, the index score is

$$
s^{\mathrm{idx},g}_{ij}
=\frac{q^{\mathrm{idx},g}_i\cdot k^{\mathrm{idx}}_j}
{\sqrt{d_{\mathrm{idx}}}}.
$$

Here $d_{\mathrm{idx}}$ is the index-vector dimension. The scores decide where to look; the main branch's own query and key vectors decide how to weight the tokens after selection. These are separate computations.

## From token scores to block selection

Partition the context into contiguous blocks $B_1,\ldots,B_L$. The indexer assigns a block the largest score of any causally visible token it contains:

$$
m^g_{i,b}=\max_{j\in B_b,\,j\le i}s^{\mathrm{idx},g}_{ij}.
$$

Blocks with no visible token receive $-\infty$. The maximum lets one strongly matching token bring its entire block into consideration. This is different from averaging token scores: a useful definition is not diluted merely because its neighboring tokens have small scores.

For a budget of $k$ blocks, MSA reserves one slot for the local block containing $i$, then selects the best $k-1$ nonlocal blocks. The local block is part of the budget, not an additional free block. Denote the resulting visible token set by $S_i^g$.

Every query head $h$ in group $g$ then computes ordinary attention on that support:

$$
o_i^h=\sum_{j\in S_i^g}
\frac{\exp(a^h_{ij})}{\sum_{u\in S_i^g}\exp(a^h_{iu})}v_j^g,
\qquad
a^h_{ij}=\frac{q_i^h\cdot k_j^g}{\sqrt{d_h}}.
$$

Sharing the support does not share the probabilities. Two heads in one group can attend differently to the same selected tokens. Two groups can select different blocks in the first place.

## Eight tokens, two reading lists

Consider four blocks of two tokens each. Query token 8 belongs to block 4. Let $k=2$, so every group gets the local block and one nonlocal block. Supply the following index scores:

| Group | Block 1 | Block 2 | Block 3 | Block 4 |
| --- | --- | --- | --- | --- |
| 1 | 3.0, 2.5 | 1.0, 0.0 | 0.0, 0.2 | −1.0, −2.0 |
| 2 | 0.1, 0.2 | 2.5, 3.0 | 0.2, 0.3 | −1.0, −2.0 |

Group 1's block maxima are $(3,1,0.2,-1)$; group 2's are $(0.2,3,0.3,-1)$. Group 1 selects blocks 1 and 4. Group 2 selects blocks 2 and 4. Both retain block 4 even though its score is lowest.

<figure class="figure-mechanism">
  <picture>
    <source media="(max-width: 620px)" srcset="/images/blog/minimax-group-block-selection/selection-mobile.svg" width="360" height="300" />
    <img src="/images/blog/minimax-group-block-selection/selection.svg" alt="A two-group, eight-token support matrix. Group one reads tokens one, two, seven, and eight; group two reads three, four, seven, and eight. Local block four uses one of each group's two block slots. Heads share token support but have separate attention weights. The indexer first scores all visible tokens." loading="lazy" width="620" height="260" />
  </picture>
  <figcaption>Figure 1. Selected supports for the eight-token example. Each group reserves block 4 within its two-block budget. Query heads in a group share these tokens but compute separate attention weights.</figcaption>
</figure>

To see the output consequence, choose scalar values

$$
v=(1,1,-1,-1,0.5,0.5,0,0).
$$

Use main-branch logits $(2,1,-2,-2,-2,-2,0,0)$ for one head in group 1, and $(-2,-2,1,2,-2,-2,0,0)$ for one in group 2. These logits are supplied independently of the index scores, as the architecture permits. Computing both full and selected-support softmax gives:

| Head | Dense output | Sparse output | Retained dense mass |
| --- | ---: | ---: | ---: |
| Group 1 | 0.788383 | 0.834811 | 0.957202 |
| Group 2 | −0.766984 | −0.834811 | 0.957202 |

Deleting a token does not simply remove its contribution while leaving everything else unchanged. The selected probabilities are renormalized. If dense attention assigns mass $M$ to the retained support, its probabilities there are divided by $M$ in the sparse calculation.

This gives a useful diagnostic. Let $o_S$ and $o_O$ be the separately normalized outputs on selected and omitted tokens. Then

$$
o_{\mathrm{dense}}=M o_S+(1-M)o_O,
$$

and, if every value has norm at most $V_{\max}$,

$$
\|o_S-o_{\mathrm{dense}}\|
\le 2V_{\max}(1-M).
$$

For this example, $V_{\max}=1$ and the bound is $0.085596$. The actual absolute errors, $0.046428$ and $0.067827$, are below it. The [NumPy example](/code/minimax-sparse-selection.py) checks the selections, renormalization, and bound. Computing retained dense mass is an explanatory diagnostic here; doing it for every real query would require the dense calculation the layer seeks to avoid.

## How does the selector learn?

Top-$k$ selection does not provide an ordinary gradient through its chosen indices. MSA therefore trains the indexer with an auxiliary KL loss. On the selected token support, each main head produces a probability distribution. The teacher is their **average probability distribution** within the group:

$$
P_j^g=\frac{1}{G}\sum_{h\in\mathcal H_g}
\operatorname{softmax}_{S_i^g}(a^h_i)_j.
$$

The index branch produces its own softmax on that same support and minimizes $D_{\mathrm{KL}}(\operatorname{stopgrad}(P^g)\|P^{\mathrm{idx},g})$. Averaging logits before softmax would define a different teacher. During sparse training, this is also not a KL against the complete dense distribution: omitted tokens are outside the support being compared.

Both the teacher and the hidden-state input to the index projections are detached. Consequently, this loss trains the index projections without pulling the backbone toward an easier indexing problem. A dense-attention warmup first gives the new indexer a learning signal before its selections control the main branch.

## What the block budget cannot promise

A block maximum ranks the best individual index match, not total main-attention mass. A block with several moderately important tokens can lose to one with a single high index score. The retained-mass bound describes the consequence of a selection; it does not prove the indexer will choose a good one.

Block-based selection also appears in [Native Sparse Attention](https://arxiv.org/abs/2502.11089v1); its compression and selection branches use a different routing design. In MSA, the local reservation provides immediate context but spends one slot even when a distant block would carry more attention mass. Group sharing likewise trades flexibility for reuse: a minority query head may need a block that its group's average teacher does not emphasize.

Finally, the main branch reads at most $k$ blocks per query, but the index branch still scores the visible context. Fixed main-attention budget does not imply constant total layer cost or linear-cost prefill. MSA's mechanism is a practical division of labor: scan with a lightweight indexer, then apply the more expensive attention calculation to a group-specific subset.

## References

1. [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392v1). 2026.
2. Joshua Ainslie et al. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245v1). 2023.
3. Jingyang Yuan et al. [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089v1). 2025.
