---
title: "Test post 2: Figures, tables, and code"
description: "A sample post for checking figures, table layout, and syntax highlighting."
date: 2026-10-25
tags: [Test]
draft: true
---

This is a second test post. The figure comes from my Language-Forcing paper; the table and code below are illustrative examples.

## Figure

![Figure from Language-Forcing showing humanoid tasks and experimental results.](/images/language-forcing.webp)

Figure: Language-Forcing, 2026. [Paper](https://arxiv.org/abs/2610.23894).

## Table

The values below are placeholders for checking table rendering. They are not experimental results.

| Setting | Demonstrations | Trials |
| --- | ---: | ---: |
| Example A | 10 | 5 |
| Example B | 20 | 5 |

## Code

The snippet computes a success rate from a list of outcomes.

```python
def success_rate(outcomes):
    if not outcomes:
        return 0.0
    return sum(outcomes) / len(outcomes)


print(success_rate([True, False, True]))
```

Inline code such as `success_rate(outcomes)` should remain readable in a paragraph.

Return to [all blogs](/blog/) or the [homepage](/).
