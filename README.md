# Yanjun Li — personal website

A simple academic homepage with biography, publications, research experience,
internship experience, and education. The **Blogs** text link opens a separate archive; posts and excerpts
never appear on the homepage. Built with Astro and Markdown.

## Local development

Use Node.js 22.23.4 or later (`nvm install && nvm use` if you use nvm).

```sh
npm ci
npm run dev
```

Open `http://localhost:4321`. To check the production build:

```sh
npm run build
npm run preview
```

## Update your information

- Personal links, publications, and experience: `src/data/site.ts`.
- Biography, research interests, and education: `src/pages/index.astro`.
- Colors, spacing, and typography: `src/styles/global.css`.
- Publication images: `public/images/`.
- Institutional logos and their original sources: `public/images/logos/`.
- Site URL: `astro.config.mjs`, `src/pages/rss.xml.ts`,
  `src/pages/sitemap.xml.ts`, and `public/robots.txt`.

The biography and experience are based on the supplied CV. Only the three public
papers confirmed by the owner are included. Unpublished manuscripts, phone
numbers, grades, and the original CV PDF are excluded. The undergraduate status
follows the owner's current description; education dates follow the CV.

## Publish a blog post

Create `src/content/blog/your-post-slug.md`:

```markdown
---
title: "A question worth exploring"
description: "A short description for RSS and social previews."
date: 2026-10-26
tags: [Robot learning, Research notes]
draft: false
---

Your article starts here.

## An observation

Write in Markdown. Images go in `public/images/` and can be embedded with
![A descriptive caption](/images/your-image.webp).
```

The filename becomes `/blog/your-post-slug/`. Draft and future-dated articles
are excluded from the archive, generated pages, RSS, and sitemap.
Future-dated posts appear after a build on or after their date. The included
`first-note.md` is an unpublished template, not a real post.

The first published article is `backpropagation-through-time.md`, with equations,
original figures, references, and a runnable NumPy implementation at
`public/code/bptt.py`. Run `python public/code/bptt.py` to check its analytical
gradients against central differences. Set `toc: true` to add a desktop sidebar
and a collapsible contents list on smaller screens. Math uses KaTeX with
`remark-math` and `rehype-katex` through Astro's unified Markdown processor.

The earlier samples, `test-post-text.md` and `test-post-media.md`, remain as
drafts for formatting checks. They are excluded from the archive and homepage.

### Monthly mechanism notes

The archive also contains sixteen English notes dated July 2025–October 2026.
Each date is the month's second Saturday; each primary paper first appeared in
the preceding month. Technical explanations cite that paper's original arXiv
version, rather than relying on later revisions.

Each note develops a concrete problem, the relevant equation or structure, a
worked minimal example, and its applicability limits. The series covers video
and image generation, LLM optimization, embodied control, and 3D perception.
Every note includes an analytical figure or a structured comparison and a
downloadable Python example. Each References section cites three or four papers,
with citations placed in the relevant parts of the explanation. Examples are analytical toys or symbolic audits, not reproductions of
the papers' model evaluations.

| Date | Mechanism | Original source |
| --- | --- | --- |
| 2025-07-12 | Self Forcing: generated-prefix training | [2506.08009v1](https://arxiv.org/abs/2506.08009v1) |
| 2025-08-09 | GSPO: sequence-level importance ratios | [2507.18071v1](https://arxiv.org/abs/2507.18071v1) |
| 2025-09-13 | DINOv3: patch Gram anchoring | [2508.10104v1](https://arxiv.org/abs/2508.10104v1) |
| 2025-10-11 | LongLive: prompt-conditioned KV re-caching | [2509.22622v1](https://arxiv.org/abs/2509.22622v1) |
| 2025-11-08 | RAE: dimension-dependent noise scheduling | [2510.11690v1](https://arxiv.org/abs/2510.11690v1) |
| 2025-12-13 | SAM 3D: joint shape and layout | [2511.16624v1](https://arxiv.org/abs/2511.16624v1) |
| 2026-01-10 | mHC: constrained residual mixing | [2512.24880v1](https://arxiv.org/abs/2512.24880v1) |
| 2026-02-14 | DynamicVLA: action-time alignment | [2601.22153v1](https://arxiv.org/abs/2601.22153v1) |
| 2026-03-14 | MOVA: aligned temporal RoPE | [2602.08794v1](https://arxiv.org/abs/2602.08794v1) |
| 2026-04-11 | Utonia: perceptual granularity | [2603.03283v1](https://arxiv.org/abs/2603.03283v1) |
| 2026-05-09 | RLSD: token update sign and magnitude | [2604.03128v1](https://arxiv.org/abs/2604.03128v1) |
| 2026-06-13 | MolmoAct2-Think: selective depth updates | [2605.02881v1](https://arxiv.org/abs/2605.02881v1) |
| 2026-07-11 | MiniMax: group-specific block selection | [2606.13392v1](https://arxiv.org/abs/2606.13392v1) |
| 2026-08-08 | Boogu-Image: intent-preserving rewriting | [2607.13125v1](https://arxiv.org/abs/2607.13125v1) |
| 2026-09-12 | Latent-to-4D: latent-grid alignment | [2608.10744v1](https://arxiv.org/abs/2608.10744v1) |
| 2026-10-10 | FuseReg: random layer-subset means | [2609.31620v1](https://arxiv.org/abs/2609.31620v1) |

The examples use Python 3 and NumPy, except the Boogu and DynamicVLA examples,
which require only the standard library. To run all article examples:

```sh
for example in public/code/*.py; do python "$example" || exit; done
```

## Deploy to GitHub Pages

1. In the repository, open **Settings → Pages**.
2. Set **Source** to **GitHub Actions**.
3. Push these files to `main`, or run **Deploy homepage** manually in **Actions**.
4. The workflow builds the site and deploys `dist/` to
   `https://yanjun711.github.io/`.

After setup, publishing only requires committing your content changes to `main`.

## Design reference

The layout follows the straightforward academic structure of
[Jon Barron's homepage](https://jonbarron.info/): a short biography, compact
publication entries with figures, and plain links to papers and projects. It
uses the reference's Lato font, 14px body and paper text, regular 32px name,
regular 22px section headings, and 800px maximum page width. The blog is a
separate page reached through the Blogs link beside Email, Scholar, and GitHub.
The archive follows [Thinking Machines Lab's blog](https://thinkingmachines.ai/blog/):
dates beside article titles and authors, with a thin rule between entries.
The homepage, archive, and articles share Lato and the same profile navigation.
Institutional logos occupy an 80 × 80 pixel area, preserving their proportions.

Publication figures are reproduced from the author's own papers. MotionVLA and
HumanoidTTT identify their arXiv licenses as CC BY-NC-SA 4.0; Language-Forcing
identifies its license as CC BY 4.0. Each figure links to its project, and each
publication links to its arXiv record. Font licenses are included by Fontsource
in the corresponding installed packages.

The Language-Forcing project URL printed in the paper returned HTTP 404 during
setup. Its figure currently links to arXiv; add a project URL in `site.ts` when
the project page becomes available.

## Verification

The initial version was checked with a production build and real Chromium:

- No horizontal overflow at 320, 360, 390, 768, 1024, and 1440 pixels.
- All publication figures and internal links load.
- BibTeX expansion, clipboard copying, and download endpoints work.
- A temporary published Markdown article appeared in the archive, article route,
  RSS, and sitemap, while staying absent from the homepage. Drafts and future
  articles stayed excluded.
- No WCAG 2 A/AA or WCAG 2.1 AA violations reported by axe for the homepage
  at desktop and mobile sizes, or the blog archive at desktop size.
- Valid RSS and sitemap XML, English-only page text, and no browser exceptions.
- No known dependency vulnerabilities reported by `npm audit` at setup time.

The sixteen-note series was additionally checked with a production build and
Chromium at 320, 390, and 1440 pixels. All seventeen published articles load,
their figures decode, code downloads resolve, TOC anchors work, and no page-wide
horizontal overflow occurs. Formula and code regions remain keyboard-scrollable.
All example scripts pass their numerical or symbolic checks. The homepage,
archive, and every article report no axe WCAG 2 A/AA or WCAG 2.1 AA violations
at mobile and desktop sizes. RSS contains seventeen entries, the sitemap
contains the matching article routes, and drafts stay excluded.

### Editing the analytical figures

`python scripts/render-blog-figures.py` regenerates the analytical SVGs from the
worked examples. It requires NumPy and Matplotlib. Use `--only sam latent` to
regenerate a subset; `--font-dir /path/to/lato` uses Lato-Regular.ttf and
Lato-Bold.ttf. SVGs embed outlined glyphs, so readers do not need those fonts
installed. Desktop and mobile variants have separately arranged panels and
legends. Figures illustrate equations or measured quantities in the toy examples;
conceptual framework boxes are not used as substitutes for those explanations.
