---
title: "Sinkhorn-Constrained Residual Mixing in mHC"
description: "What doubly stochastic residual mixing preserves, what it contracts, and why twenty Sinkhorn iterations are an approximation."
date: 2026-01-10
tags: [LLM, Optimization, Neural networks]
draft: false
toc: true
---

A standard residual connection carries a state forward through an identity map. If we replace that state with several parallel streams, we can learn how to mix them before the next layer. But an unconstrained mixing matrix can amplify the state every time it is applied. Small per-layer gains become large gains across depth.

mHC, *Manifold-Constrained Hyper-Connections* [\[1\]](https://arxiv.org/abs/2512.24880v1), restricts this residual mixing matrix to be doubly stochastic: nonnegative, with every row and every column summing to one. This note focuses on that constraint. It gives a precise statement about a frozen linear residual path, while leaving important qualifications for the full nonlinear network.

## A residual path with several streams

The expanded residual stream comes from Hyper-Connections [\[2\]](https://arxiv.org/abs/2409.19606v1). Write its state as $X_l\in\mathbb R^{n\times C}$: $n$ streams, each with feature width $C$. A schematic layer is

$$
X_{l+1}
=B_lX_l+q_l^{\mathsf T}F_l(p_lX_l),
$$

where $B_l\in\mathbb R^{n\times n}$ mixes the residual streams, $p_l\in\mathbb R^{1\times n}$ reads from them, and $q_l\in\mathbb R^{1\times n}$ writes the branch output back. These are simplified names for the paper's residual, pre-, and post-mappings. We will examine $B_l$ alone.

An unconstrained choice such as $B_l=1.1I$ multiplies the residual-path magnitude by $1.1$ at every layer. After $L$ repetitions, the factor is $1.1^L$. The nonlinear branch need not be active for this amplification to occur.

Setting $B_l=I$ prevents that problem but also prevents direct communication between streams along this path. The identity-path motivation follows the signal-propagation analysis of He et al. [\[3\]](https://arxiv.org/abs/1603.05027v1). mHC instead chooses matrices in the set

$$
\mathcal B_n=
\left\{B\in\mathbb R^{n\times n}:
B_{ij}\geq0,\ B\mathbf1=\mathbf1,
\ \mathbf1^{\mathsf T}B=\mathbf1^{\mathsf T}
\right\}.
$$

Each output is a convex combination of inputs, and each input contributes a total weight of one across the outputs. This is the Birkhoff polytope, the convex hull of permutation matrices. Its elements need not be permutations, orthogonal matrices, or identities.

## Why both sums matter

For a vector $v\in\mathbb R^n$, apply convexity of the square to each row:

$$
\begin{aligned}
\|Bv\|_2^2
&=\sum_i\left(\sum_j B_{ij}v_j\right)^2\\
&\leq\sum_i\sum_j B_{ij}v_j^2\\
&=\sum_jv_j^2\sum_iB_{ij}
=\|v\|_2^2.
\end{aligned}
$$

The inequality uses nonnegativity and the row sums. The final equality uses the column sums. Thus the linear map is nonexpansive in Euclidean norm. Because $B\mathbf1=\mathbf1$, its spectral norm is in fact exactly one, though many individual vectors can shrink.

Row normalization alone does not provide this bound. For example,

$$
A=\begin{bmatrix}1&0\\1&0\end{bmatrix}
$$

has row sums equal to one, but takes $(1,0)$ to $(1,1)$ and increases its norm by $\sqrt2$. The column constraint rules out concentrating an input into several outputs without accounting for its total contribution.

Products preserve the constraint. If $A$ and $B$ are doubly stochastic, $AB$ is nonnegative and

$$
AB\mathbf1=\mathbf1,
\qquad \mathbf1^{\mathsf T}AB=\mathbf1^{\mathsf T}.
$$

Consequently, a chain of these **frozen linear mixing maps** remains nonexpansive. Their transposes are also doubly stochastic, giving the corresponding bound for backpropagation through this fixed linear chain.

## Turning logits into a mixing matrix

mHC parameterizes positive entries with exponentiation, then approximately enforces the row and column constraints using the alternating scaling procedure studied by Sinkhorn and Knopp [\[4\]](https://msp.org/pjm/1967/21-2/p14.xhtml). For a logit matrix $Z$, begin with $M^{(0)}=\exp(Z)$, element by element. Define

$$
[T_c(M)]_{ij}=\frac{M_{ij}}{\sum_kM_{kj}},
\qquad
[T_r(M)]_{ij}=\frac{M_{ij}}{\sum_kM_{ik}}.
$$

The paper applies column normalization followed by row normalization:

$$
M^{(k)}=T_r\bigl(T_c(M^{(k-1)})\bigr).
$$

It uses twenty iterations. This is a finite approximation: the final row normalization makes row sums one to floating-point accuracy, while column sums may still differ from one. The exact norm argument above applies to an exactly doubly stochastic matrix, not automatically to every finite-iteration output.

Sinkhorn scaling also should not be read as the Euclidean nearest-point projection of an arbitrary matrix onto $\mathcal B_n$. It alternately rescales a positive matrix. For this note, its role is to provide a differentiable parameterization that approaches the desired normalization constraints.

## A two-stream calculation

Take logits whose exponentials are

$$
M^{(0)}=\begin{bmatrix}3&1\\1&3\end{bmatrix}.
$$

One column-then-row iteration already gives

$$
B=\begin{bmatrix}0.75&0.25\\0.25&0.75\end{bmatrix}.
$$

This matrix preserves the shared component and contracts the difference component:

$$
B\begin{bmatrix}1\\1\end{bmatrix}
=\begin{bmatrix}1\\1\end{bmatrix},
\qquad
B\begin{bmatrix}1\\-1\end{bmatrix}
=\frac12\begin{bmatrix}1\\-1\end{bmatrix}.
$$

After ten identical mixing steps, the difference is multiplied by $2^{-10}=1/1024$. Its disappearance is compatible with the nonexpansive bound. “Norm bounded by one” does not mean every direction retains its magnitude or all information remains recoverable.

<figure>
  <picture>
    <source media="(max-width: 620px)" srcset="/images/blog/mhc-sinkhorn-residual-mixing/mixing-depth-mobile.svg" width="346" height="307" />
    <img src="/images/blog/mhc-sinkhorn-residual-mixing/mixing-depth.svg" alt="Logarithmic norm ratios for repeated toy mixing. An unconstrained gain grows, the shared mode stays constant, and the difference mode halves at every step." loading="lazy" width="595" height="307" />
  </picture>
  <figcaption>Figure 1. Exact norm ratios for fixed toy matrices. Doubly stochastic mixing preserves the shared mode and contracts the difference mode; an unconstrained gain accumulates across depth. This is a residual-path calculation, not a whole-network stability measurement.</figcaption>
</figure>

The [runnable NumPy example](/code/mhc-sinkhorn-residual-mixing.py) checks these calculations and reports the actual normalization residuals for a random matrix. The core iteration is:

~~~python
import numpy as np

def sinkhorn(logits, iterations=20):
    matrix = np.exp(logits - np.max(logits))
    for _ in range(iterations):
        matrix /= matrix.sum(axis=0, keepdims=True)
        matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix
~~~

Subtracting the largest logit prevents overflow in this small example; a shared positive scale disappears during normalization. Very large logit ranges can still underflow. A production implementation may require log-domain calculations and attention to numerical precision.

## The boundary of the stability claim

There are two separate limitations. First, finite Sinkhorn iterations leave a normalization error. Measuring row and column residuals is more informative than assuming twenty iterations make every possible matrix exact. Highly imbalanced logits can converge slowly.

We can still quantify a useful approximate bound. Suppose the final nonnegative matrix has row sums one and its largest column sum is at most $1+\delta$. The standard induced-norm inequality gives

$$
\|B\|_2\leq\sqrt{\|B\|_1\|B\|_\infty}
\leq\sqrt{1+\delta}.
$$

Here $\|B\|_1$ is the largest absolute column sum and $\|B\|_\infty$ is the largest absolute row sum. The downloadable example checks this bound for its finite-iteration output. This is weaker than exact nonexpansiveness. Even a small gain can accumulate when multiplied across many layers, so a small normalization residual has to be interpreted in the context of depth.

Second, the full residual matrix depends on the input. For $g(X)=B(X)X$, a perturbation obeys

$$
dg=B(X)\,dX+[dB(X)]X.
$$

The doubly stochastic bound controls the first term. It does not remove the second term. Nor does it bound the Jacobian of the nonlinear branch, its read/write mappings, or their sum with the residual path. It would therefore be incorrect to conclude that the entire network has Jacobian norm at most one.

The constraint still has a useful, concrete purpose: it permits stream mixing while preventing the fixed residual transport matrices from accumulating arbitrary multiplicative gains. The price is that mixing can suppress stream differences. The rest of the architecture determines whether that suppression helps combine useful information or erases distinctions the model needs.

The constraint also preserves the sum across streams for every feature channel: $\mathbf1^{\mathsf T}BX=\mathbf1^{\mathsf T}X$. It therefore conserves a stream mean without conserving each stream's identity. That distinction is visible in the two-stream example: both streams approach their shared mean, while the original difference becomes harder to recover.

## References

1. Zhenda Xie et al. [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880v1). 2025.
2. Defa Zhu et al. [Hyper-Connections](https://arxiv.org/abs/2409.19606v1). 2024.
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027v1). 2016.
4. Richard Sinkhorn and Paul Knopp. [Concerning nonnegative matrices and doubly stochastic matrices](https://msp.org/pjm/1967/21-2/p14.xhtml). *Pacific Journal of Mathematics* 21(2), 343–348, 1967.
