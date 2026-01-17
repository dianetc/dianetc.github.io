# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Deploy Commands
- Preview site locally: `python -m http.server` (navigate to http://localhost:8000)
- Deploy: Site auto-deploys when changes are pushed to the master branch

## Site Architecture

This is a personal academic website built with static HTML pages using the Tachyons CSS framework. The site follows a simple, flat structure:

### Page Types and Structure
1. **Landing page** (`index.html`): Personal intro, research interests, contact info, link to musings
2. **Section pages** (`research/index.html`, `presentations/index.html`): CV-style content with timeline formatting
3. **Musings index** (`musings/index.html`): Blog-style list of entries with titles, dates, and links
4. **Individual blog posts** (`musings/[topic]/index.html`): Long-form content with consistent layout
5. **Template** (`blog_template.html`): Base template for new blog posts with placeholders

### Common Patterns
- All pages share similar base structure (DOCTYPE, Tachyons CDN, monospace font, cream background `#faf9f6`)
- Custom `<style>` blocks define page-specific colors and classes (each page uses different link colors)
- Content contained in `.content-box` with `max-width: 700px`
- Navigation links: "← Back to home" or "← Back to musings" at bottom
- Semantic HTML5 elements (`<main>`, `<section>`) used throughout
- Images stored inline within blog post directories (e.g., `musings/rir/rir_example_entry.png`)

## Code Style Guidelines

### HTML
- Use 4-space indentation
- Include `lang="en"` attribute in html tag
- Use viewport meta tag: `<meta name="viewport" content="width=device-width, initial-scale=1">`
- External CSS from CDNs (Tachyons via unpkg)
- Custom styles in `<style>` block in `<head>`
- Use semantic HTML5 elements (`<main>`, `<section>`, `<nav>`, etc.)

### CSS
- Primary framework: Tachyons utility classes
- Custom CSS uses kebab-case for class names
- Define colors semantically in inline `<style>` blocks
- Link colors vary by page type:
  - Main index: `#8B6914` (dark yellow/golden brown)
  - Musings index: `#006400` (dark green)
  - Blog posts: `#6E5BAA` (purple) or `#8B6914` (golden brown)
- Common custom classes: `.content-box`, `.last-updated`, `.semi-bold`, `.entry-spacing`
- Code blocks: use `<pre><code>` with background `#f0eee9` and border

### File Structure
- Root contains main pages: `index.html`, `operations-research.html`, `or-nlp-connection.html`
- Section directories: `research/`, `presentations/`, `work/`, `publications/`
- Blog content: `musings/[topic]/index.html` (each post in its own directory)
- Assets: PDFs and images stored in root directory (e.g., `Tchuindjo_CV.pdf`, `love_and_math_first_10.pdf`)
- Blog post assets: images stored within each post's directory
- Template for new posts: `blog_template.html` with `TITLE_PLACEHOLDER` and `<!-- CONTENT_PLACEHOLDER -->`

### Creating New Blog Posts
1. Create new directory in `musings/[topic-name]/`
2. Copy `blog_template.html` to `musings/[topic-name]/index.html`
3. Replace `TITLE_PLACEHOLDER` with post title
4. Replace `<!-- CONTENT_PLACEHOLDER -->` with content inside `<div class="mw6 ph1">...</div>`
5. Update `musings/index.html` to add new entry at the top with title, date, and link

### Git Workflow
- Main branch: `master`
- Keep commit messages concise and descriptive (e.g., "less words", "more flow", "arxiv")
- Use `git commit --amend` for quick fixes to recent commits