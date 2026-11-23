---
title: "Your post title"
description: "One or two sentences. Shown in post lists, link previews and search results."
tags: [machine-learning, reflections]
math: false   # set to true if the post uses LaTeX
# last_modified_at: 2026-11-20   # uncomment when you meaningfully revise a post
---

Opening paragraph: what you set out to learn and why.

<!-- Optional table of contents: remove these two lines if not needed -->
* TOC
{:toc}

## What I expected

Regular **Markdown** works: *italics*, `inline code`, [links](https://example.com), lists, tables.

## What actually clicked

> A quote or a key takeaway.

> **Note:** callouts are blockquotes with a class added on the next line.
{: .note}

```python
import torch

x = torch.randn(3, 4)
print(x.mean(dim=0))
```

Inline math (needs `math: true`): $$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Display math goes on its own lines:

$$
\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \log p_\theta(y_i \mid x_i)
$$

| Approach | What I liked | What tripped me up |
|----------|--------------|--------------------|
| A        | ...          | ...                |
| B        | ...          | ...                |

![Alt text describing the image](/assets/images/your-post-slug/figure.png)
*Optional caption in italics right below the image.*

## What I'd do differently

Closing thoughts.
