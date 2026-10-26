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
description: "A short description for the archive, RSS, and social previews."
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

Two published samples, `test-post-text.md` and `test-post-media.md`, are included
to check the blog archive and article formatting. They are explicitly labeled
as test posts. Delete them, or set `draft: true`, when they are no longer needed.
They never appear on the homepage.

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
