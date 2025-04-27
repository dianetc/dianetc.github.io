#!/usr/bin/env python3
import os
import re
from pathlib import Path

# The directory containing blog posts
base_dir = Path('/home/dtchuindjo/dianetc.github.io/musings')

# Read the template file
with open('/home/dtchuindjo/dianetc.github.io/blog_template.html', 'r') as f:
    template = f.read()

# Find all blog post directories
blog_dirs = [d for d in base_dir.iterdir() if d.is_dir() and (d / 'index.html').exists()]

for blog_dir in blog_dirs:
    blog_path = blog_dir / 'index.html'
    print(f"Processing {blog_path}")
    
    # Read the current blog post
    with open(blog_path, 'r') as f:
        content = f.read()
    
    # Extract the title
    title_match = re.search(r'<title>(.*?)</title>', content)
    if title_match:
        title = title_match.group(1)
    else:
        title = blog_dir.name.replace('_', ' ').title()
    
    # Extract the main content
    content_match = re.search(r'<main.*?>(.*?)</main>', content, re.DOTALL)
    if content_match:
        main_content = content_match.group(1)
        
        # Try to extract just the content div
        div_match = re.search(r'<div.*?>(.*?)</div>', main_content, re.DOTALL)
        if div_match:
            blog_content = div_match.group(0)
        else:
            blog_content = main_content
    else:
        print(f"Couldn't extract content from {blog_path}")
        continue
    
    # Create new blog post with template
    new_content = template.replace('TITLE_PLACEHOLDER', title)
    new_content = new_content.replace('<!-- CONTENT_PLACEHOLDER -->', blog_content)
    
    # Write the new content
    with open(blog_path, 'w') as f:
        f.write(new_content)
    
    print(f"Updated {blog_path}")

print("All blog posts updated!")