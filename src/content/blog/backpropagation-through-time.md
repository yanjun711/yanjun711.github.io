---
title: "Backpropagation Through Time"
description: "Deriving BPTT, checking a small RNN by hand, and understanding what truncation changes."
date: 2026-10-27
tags: [Optimization, Sequence models]
draft: false
toc: true
---

A recurrent model uses its previous state to compute its next state. A mistake at the end of a sequence can therefore depend on a computation that happened many steps earlier. [Backpropagation through time (BPTT)](https://doi.org/10.1109/5.58337) computes how much each of those earlier computations contributed to the loss.

The procedure is ordinary reverse-mode differentiation applied to an unfolded recurrent computation. The same parameters appear at every step, so their gradient is the **sum of the contributions from every use**. Keeping track of that sum is the main idea behind the algorithm.

<figure>
  <picture>
    <source media="(max-width: 620px)" srcset="/images/blog/bptt/unrolled-mobile.svg" width="346" height="221" />
    <img src="/images/blog/bptt/unrolled.svg" alt="Three recurrent steps share the same transition parameters. Gray arrows carry states forward, while blue arrows carry gradients back from each loss and from later states." width="595" height="221" />
  </picture>
  <figcaption>Figure 1. Unfolding a recurrence makes its dependencies explicit. Each state receives a gradient from its own loss and from the states that follow it.</figcaption>
</figure>

## Unrolling the computation

Consider a sequence of inputs $x_1,\ldots,x_T$ and a recurrent state

$$
h_t = f_\theta(h_{t-1}, x_t),
\qquad
L = \sum_{t=1}^{T} \ell_t(h_t).
$$

Here $h_t\in\mathbb{R}^H$, and $\theta\in\mathbb{R}^P$ contains the parameters shared by every transition. Assume that $h_0$ is fixed and that each loss depends on $\theta$ only through the states. We will add a trainable output layer in the RNN example below.

Running the recurrence for $T$ steps produces a computation graph with $T$ copies of $f_\theta$. These are copies of the operation, with references to the same parameter vector. They are not separate models with independently trained weights.

In the forward pass, we compute the states and losses. In the backward pass, we traverse this graph in reverse, starting from the last state. We need the forward activations to evaluate the local derivatives, which is why longer sequences require more memory unless we recompute some activations.

## The backward recursion

Let $g_t=\nabla_{h_t}L$ denote the total gradient at state $h_t$. Define three local quantities:

$$
r_t = \nabla_{h_t}\ell_t,
\qquad
J_t = \frac{\partial f_\theta(h_{t-1},x_t)}{\partial h_{t-1}},
\qquad
B_t = \frac{\partial f_\theta(h_{t-1},x_t)}{\partial\theta}.
$$

The Jacobians $J_t\in\mathbb{R}^{H\times H}$ and $B_t\in\mathbb{R}^{H\times P}$ are local derivatives: their other inputs are held fixed. Treating gradients as column vectors, the chain rule gives

$$
\begin{aligned}
g_T &= r_T,\\
g_t &= r_t + J_{t+1}^{\mathsf T}g_{t+1},
\qquad t=T-1,\ldots,1.
\end{aligned}
$$

The first term is the effect on the current loss. The second term is the effect on all later losses, passed back through the next transition. Once these state gradients are known, the shared parameter gradient is

$$
\nabla_\theta L = \sum_{t=1}^{T} B_t^{\mathsf T}g_t.
$$

This computation does not require constructing either Jacobian as a dense matrix. An automatic differentiation system evaluates the corresponding vector–Jacobian products. The backward sweep visits each transition once, instead of separately tracing every possible path from each loss to each earlier state.

If $h_0$ is learned, it also receives the gradient $J_1^{\mathsf T}g_1$. If a loss depends explicitly on a parameter, its direct derivative must be added as well.

### A three-step example

Take a scalar recurrence $h_t=w h_{t-1}$, with $h_0=1$, three steps, and a loss only at the final step:

$$
h_3=w^3,
\qquad L=\tfrac12 h_3^2=\tfrac12 w^6,
\qquad \frac{dL}{dw}=3w^5.
$$

At $w=0.5$, the states are $0.5$, $0.25$, and $0.125$. Each use of $w$ contributes $0.03125$ to the gradient, giving $0.09375$ in total. Differentiating only the final transition would retain one of those three contributions and miss the other two.

## A tanh RNN

For a concrete implementation, use a tanh state transition and a linear output layer:

$$
\begin{aligned}
a_t &= W h_{t-1} + U x_t + b,\\
h_t &= \tanh(a_t),\\
\hat y_t &= Vh_t+c,\\
\ell_t &= \tfrac12\|\hat y_t-y_t\|_2^2.
\end{aligned}
$$

Write $e_t=\hat y_t-y_t$ and $\delta_t=\nabla_{a_t}L$. The reverse recursion becomes

$$
\delta_t =
\left(V^{\mathsf T}e_t + W^{\mathsf T}\delta_{t+1}\right)
\odot (1-h_t\odot h_t),
\qquad \delta_{T+1}=0.
$$

The term $V^{\mathsf T}e_t$ comes from the current prediction; $W^{\mathsf T}\delta_{t+1}$ comes from the future. The elementwise factor is the derivative of tanh. The parameter gradients are

$$
\begin{aligned}
\nabla_W L &= \sum_t \delta_t h_{t-1}^{\mathsf T},
&\nabla_U L &= \sum_t \delta_t x_t^{\mathsf T},\\
\nabla_b L &= \sum_t \delta_t,
&\nabla_V L &= \sum_t e_t h_t^{\mathsf T},\\
\nabla_c L &= \sum_t e_t.
\end{aligned}
$$

These equations use a sum over time and output coordinates. Averaging the loss would scale every gradient by the same averaging factor.

## Implementation

The following NumPy implementation follows the equations directly. It handles one sequence, starts from a zero state, and stores the states and output errors during the forward pass. The arrays and parameters use double precision in the accompanying example.

```python
import numpy as np


def forward(p, x, y):
    states = [np.zeros(p["W"].shape[0], dtype=x.dtype)]
    errors = []
    loss = 0.0
    for xt, yt in zip(x, y):
        h = np.tanh(p["W"] @ states[-1] + p["U"] @ xt + p["b"])
        e = p["V"] @ h + p["c"] - yt
        states.append(h)
        errors.append(e)
        loss += 0.5 * (e @ e)
    return float(loss), (states, errors)


def backward(p, x, cache):
    states, errors = cache
    grads = {name: np.zeros_like(value) for name, value in p.items()}
    dh_next = np.zeros_like(states[0])
    for t in reversed(range(len(x))):
        h_prev, h = states[t], states[t + 1]
        e = errors[t]
        dh = p["V"].T @ e + dh_next
        da = dh * (1.0 - h * h)
        grads["V"] += np.outer(e, h)
        grads["c"] += e
        grads["W"] += np.outer(da, h_prev)
        grads["U"] += np.outer(da, x[t])
        grads["b"] += da
        dh_next = p["W"].T @ da
    return grads
```

`dh_next` carries the contribution from later time steps. The `+=` operations accumulate the gradients of parameters reused across the sequence. The parameters must remain unchanged throughout this forward and backward pass; an optimizer update happens afterward.

### Checking the gradient

A useful check is to perturb each parameter coordinate and recompute the entire sequence loss:

$$
\frac{\partial L}{\partial\theta_i}
\approx
\frac{L(\theta+\varepsilon e_i)-L(\theta-\varepsilon e_i)}{2\varepsilon}.
$$

Here $e_i$ is a coordinate basis vector, not the prediction error defined earlier. The [complete runnable script](/code/bptt.py) checks every entry of $W$, $U$, $b$, $V$, and $c$ against central differences with $\varepsilon=10^{-6}$. For the included five-step, three-hidden-unit example, the largest absolute discrepancy is about $3.4\times10^{-10}$.

```sh
python -m pip install numpy
python bptt.py
```

Finite differences are expensive and sensitive to numerical precision, so this is a check for small examples. In a training implementation, automatic differentiation performs the same reverse traversal. For example, PyTorch records the operations used to build the loss and traverses that graph when `loss.backward()` is called.

## Long gradient paths

The contribution from a loss at time $s$ to a state at an earlier time $t$ contains a product of transition Jacobians:

$$
\nabla_{h_t}\ell_s
=J_{t+1}^{\mathsf T}J_{t+2}^{\mathsf T}\cdots
J_s^{\mathsf T}r_s,
\qquad s>t.
$$

If every Jacobian on a path has operator norm at most $q<1$, the norm of this contribution is bounded by $q^{s-t}\|r_s\|$. The signal can become very small over a long sequence. Expanding directions can instead amplify gradients; in a multidimensional model, the directions and alignment of successive Jacobians also matter.

<figure>
  <picture>
    <source media="(max-width: 620px)" srcset="/images/blog/bptt/gradient-paths-mobile.svg" width="346" height="307" />
    <img src="/images/blog/bptt/gradient-paths.svg" alt="A logarithmic plot of gradient magnitude versus the number of recurrent steps. A local multiplier of 0.8 decays, 1.0 remains constant, and 1.2 grows." loading="lazy" width="595" height="307" />
  </picture>
  <figcaption>Figure 2. For the scalar recurrence h<sub>t</sub> = a h<sub>t−1</sub>, the state-to-state derivative over k steps has magnitude |a|<sup>k</sup>. These are exact values for a toy recurrence, not measurements from a trained RNN.</figcaption>
</figure>

For a tanh RNN, $J_t=\operatorname{diag}(1-h_t\odot h_t)W$. Saturated tanh units have small derivatives, so even a recurrent weight matrix with moderate norm can produce weak long-range gradients.

[Pascanu et al.](https://arxiv.org/abs/1211.5063) analyze these long-path difficulties and motivate gradient norm clipping. Clipping limits the magnitude of an update when the gradient becomes too large. For a gradient vector $g$ and threshold $\tau>0$, clipping by global norm uses

$$
g \leftarrow g\min\left(1,\frac{\tau}{\|g\|_2}\right),
$$

with a zero gradient left unchanged. Clipping does not recover a signal that has already vanished. Gated architectures such as [LSTMs](https://doi.org/10.1162/neco.1997.9.8.1735) provide additive state paths that can help preserve gradients, but they do not guarantee successful learning over arbitrary horizons.

## Truncated BPTT

Full BPTT lets a loss propagate through the entire available sequence. Truncated BPTT cuts selected backward paths to keep the graph manageable. A common implementation processes non-overlapping chunks of $K$ steps, carries the final hidden state into the next chunk, and detaches it at each boundary.

The hidden state's numerical value survives the boundary, so a prediction can still use information from earlier inputs. Its computational history does not survive: losses in the new chunk do not assign gradient credit to computations in the previous chunk.

For an existing PyTorch `model`, `optimizer`, and `criterion`, the training loop has this form:

```python
h = None
for x_chunk, y_chunk in chunks:
    if h is not None:
        h = h.detach()  # For an RNN or GRU tensor state.
    optimizer.zero_grad()
    prediction, h = model(x_chunk, h)
    loss = criterion(prediction, y_chunk)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

This snippet assumes `torch` is imported and chunks belong to one continuing sequence. Reset the state for a new independent sequence. An LSTM carries a tuple of hidden and cell states; detach both tensors at a boundary.

With non-overlapping chunks, paths are at most $K$ steps long, and often shorter near a boundary. Overlapping windows and other update schedules make different tradeoffs. In general, truncation changes the gradient estimate; it is not merely a memory-saving way to compute the full gradient. [Activation checkpointing](https://arxiv.org/abs/1604.06174v1) offers a different tradeoff: it recomputes activations during the backward pass while retaining the full gradient paths.

## Practical notes

For the simple RNN above, storing the hidden states requires $O(TH)$ memory. Real implementations also retain other activations and intermediate values. Forward and backward computation both scale linearly with the number of time steps, for a fixed transition architecture.

Three details are worth checking when implementing the algorithm:

- **Loss reduction.** Summing and averaging over time produce different gradient scales. Padding masks must also be reflected in the loss and its normalization.
- **State boundaries.** Decide whether a hidden state should be reset, carried with its graph, or carried after detaching. Each choice changes which dependencies the gradient can follow.
- **Gradient accumulation.** Accumulate contributions across time within one backward pass, and clear gradients between optimizer steps unless accumulation across steps is intentional.

The same reasoning applies to a differentiable world model rolled forward over several steps: later prediction or planning losses can send gradients through earlier model states. The resulting gradient is a derivative of that model's rollout. It is not automatically a derivative through the physical environment, and its usefulness depends on how accurately the learned dynamics represent the rollout.

## References

1. Paul J. Werbos. [Backpropagation through time: what it does and how to do it](https://doi.org/10.1109/5.58337). *Proceedings of the IEEE*, 78(10), 1550–1560, 1990.
2. Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. [On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063). *ICML*, 2013.
3. Sepp Hochreiter and Jürgen Schmidhuber. [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735). *Neural Computation*, 9(8), 1735–1780, 1997.
4. Tianqi Chen et al. [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174v1). 2016.
