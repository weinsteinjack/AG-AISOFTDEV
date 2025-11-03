# React Component Preview

This folder contains a simple setup to preview your AI-generated React components in the browser.

## Quick Start

### Option 1: Using Python's Built-in Server (Easiest)

1. Open a terminal and navigate to this folder:
   ```bash
   cd frontend_preview
   ```

2. Start a local web server:
   ```bash
   python -m http.server 8000
   ```

3. Open your browser and go to:
   ```
   http://localhost:8000
   ```

4. You should see your React component rendered!

### Option 2: Using Node.js (if you have it installed)

```bash
cd frontend_preview
npx serve
```

Then open the URL shown in the terminal.

## How It Works

- **index.html**: Loads React, ReactDOM, Babel, and Tailwind CSS from CDNs
- **component.jsx**: Your generated React component (currently the refactored onboarding dashboard)
- The browser compiles JSX on-the-fly using Babel

## Making Changes

1. Edit `component.jsx` with your own React component
2. Save the file
3. Refresh your browser to see the changes

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Blank page | Open browser DevTools (F12) and check the Console tab for errors |
| Tailwind classes not working | Make sure the Tailwind CDN script is in index.html |
| Component not rendering | Check that the component is exported and rendered at the bottom of component.jsx |
| CORS errors | Make sure you're using a local server (python -m http.server) and not opening the HTML file directly |

## Tips

- Press F12 to open browser DevTools and inspect elements
- Use the Console tab to see any JavaScript errors
- Use the Network tab to check if all resources are loading
- The component includes interactive buttons - try clicking them!

## What's Included

The current component is the **Onboarding Dashboard** from Day 8 Lab 1, which includes:
- A responsive header with navigation
- A welcome banner with video thumbnail
- Three action cards (Forms, Team, Knowledge Base)
- A sidebar with checklist, office location, and schedule
- Full Tailwind CSS styling
- Interactive elements with hover states

Enjoy previewing your AI-generated React components! 🚀

