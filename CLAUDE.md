# CLAUDE.md - Guide for Agentic Coding Assistants

## Build & Deploy Commands
- Preview site locally: `python -m http.server` (navigate to http://localhost:8000)
- Deploy: Site auto-deploys when changes are pushed to the master branch

## Code Style Guidelines

### HTML
- Use 4-space indentation
- Use semantic HTML5 elements (`section`, `nav`, etc.)
- Include lang attribute in html tag
- Use viewport meta tag for responsive design
- External CSS from CDNs (Tachyons)
- Custom styles in `<style>` block in head

### CSS
- Follow Tachyons naming conventions for utility classes
- Custom CSS uses kebab-case for class names
- Define colors semantically (e.g., darkgreen for links)

### File Structure
- Main content: index.html
- Blog posts: musings/[topic]/index.html
- Assets: Store PDFs and images in root directory
- Maintain consistent directory structure for new content

### Git Workflow
- Keep commit messages concise and descriptive
- Use `git commit --amend` for quick fixes to recent commits