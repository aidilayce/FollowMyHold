# Follow My Hold: Hand-Object Interaction Reconstruction through Geometric Guidance

Official implementation of **FollowMyHold** (3DV 2026).
<a href='https://aidilayce.github.io/FollowMyHold-page/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/pdf/2508.18213'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>

> **Status:** 🚧 **Core Logic Available.** The full initialization & optimization pipeline and demo releases are being refactored and are planned for **January 2026**.

### 🚀 Updates:
- **[December 4 2025]** Added core model logic files (see `src/`) to make it easier to understand how the method works.
- **[November 14 2025]** <a href="https://github.com/aidilayce/FollowMyHold/tree/main/test_splits">Test splits</a> are now available.
- **[November 9 2025]** Got accepted to 3DV 2026!

### 📂 Core Implementation Preview
To support early research, we’re releasing the core logic files. These are meant for reference to understand how optimization-in-the-loop guidance is implemented; the runnable code will arrive with the full release.

* `src/hunyuan_guided.py`: Main architecture and optimization-in-the-loop guidance implementation.
* `src/kaolin_sdf_ops.py`: Custom SDF conversion utilities using the Kaolin library.

### 🚀 Updates:
- **[November 14 2025]** <a href="https://github.com/aidilayce/FollowMyHold/tree/main/test_splits">Test splits</a> are now available.
- **[November 9 2025]** Got accepted to 3DV 2026!

![teaser](assets/teaser.png)
