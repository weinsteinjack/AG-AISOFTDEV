# Student Guide: Previewing React Components Generated in the Labs

Day 8 focuses on AI-assisted front-end development. Most labs output `.jsx` snippets instead of a full React project so you can focus on reasoning about UI structure. This guide shows two lightweight ways to preview those components locally without scaffolding a large build system.

---

## Option 1 – Zero-Build HTML Shell (Recommended for Quick Checks)

This approach uses a single `index.html` file plus CDN-hosted React, ReactDOM, Babel, and optional Tailwind CSS. It is perfect for verifying layout or styles immediately after a lab.

### Step-by-Step

1. **Create a folder for previews** (for example, `frontend_preview/` at the repository root).
2. **Copy your generated component** from `Labs/Day_08_Vision_and_Evaluation` (or the matching file in `Solutions/`) into that folder. Example filename: `onboarding_dashboard.jsx`.
3. **Create `index.html`** alongside the component and paste the template below. Update the final `<script>` tag so `src` points to your component file.

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>React Component Preview</title>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
      body {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        background: #f4f5f7;
        margin: 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI";
      }
    </style>
  </head>
  <body>
    <div id="root"></div>
    <script type="text/babel" src="onboarding_dashboard.jsx"></script>
  </body>
</html>
```

4. **Start a static web server** from the preview folder:

   ```bash
   cd frontend_preview
   python -m http.server 8000
   ```

5. **Open** `http://localhost:8000` in your browser. Save the `.jsx` file and refresh to see updates.

> **Tip:** Keep multiple preview folders if you want to compare different component variations side-by-side.

### Troubleshooting

| Issue | Fix |
| --- | --- |
| Seeing raw JSX in the browser | Ensure the `<script>` tag that imports your component includes `type="text/babel"`. |
| Blank page with console errors | Check the browser dev tools (`F12`) for syntax errors or incorrect file paths. |
| Tailwind classes not applied | Confirm the Tailwind CDN script is present and that class names are correct. |

---

## Option 2 – Minimal Vite Project (Great for Iteration)

If you want hot reloading, TypeScript support, or integration tests, scaffold a lightweight Vite project dedicated to Day 8.

### Initialise

```bash
npm create vite@latest onboarding-ui -- --template react
cd onboarding-ui
npm install
```

Copy the generated components into `src/components/`. Update `src/App.jsx` to render the component you want to preview.

```jsx
import OnboardingDashboard from "./components/onboarding_dashboard.jsx";

function App() {
  return <OnboardingDashboard />;
}

export default App;
```

Start the dev server:

```bash
npm run dev
```

Open the URL shown in the terminal (typically `http://localhost:5173`).

### Integrating With the Course Backend

When connecting to the FastAPI backend you build in Day 3:

1. Create a `.env.local` file in the Vite project with `VITE_API_URL=http://localhost:8000`.
2. Read the variable in your component: `const api = import.meta.env.VITE_API_URL;`.
3. Use `fetch` or `axios` to call the FastAPI endpoints created during the backend labs.

This mirrors the wiring you will perform later in the deployment guide.

---

## Organizing Components

* Store raw lab outputs under version control (e.g., `Labs/Day_08_Vision_and_Evaluation/generated_components/`).
* Save polished versions to `frontend/src/components/` once you are ready to integrate them with the backend.
* Document each component’s expected props and dependencies so teammates can consume them easily.

---

## Frequently Asked Questions

**Do I need Node.js for Option 1?**  
No. Python’s built-in `http.server` works because React and Babel are loaded via CDN.

**How do I bundle multiple components on one page?**  
Create additional `<script type="text/babel">` blocks that `import` the components you need and render them into different DOM nodes, or use a single entry file that aggregates them.

**Can I use Tailwind or other CSS frameworks?**  
Yes. Include the appropriate CDN link (as shown above) or configure the framework within Vite if you need custom builds.

---

Previewing components early keeps your AI-generated UI grounded in reality and highlights accessibility or layout issues before you wire everything to the backend.
