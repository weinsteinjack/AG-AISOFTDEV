# Student Guide: How to View Your React (.jsx) Components Locally

This guide provides the simplest method to view and test your AI-generated React (`.jsx`) components on your local machine. This approach is perfect for the labs in this course as it requires **no complex build tools** like Vite, Webpack, or Create React App.

We will use a single HTML file to load the necessary libraries directly in the browser and a simple, built-in web server to display the page.

### Why Do We Need This Process?

Browsers don't understand JSX syntax out of the box. JSX (e.g., `<div>Hello</div>`) needs to be converted (transpiled) into regular JavaScript function calls (e.g., `React.createElement('div', null, 'Hello')`).

Instead of setting up a complex local development environment, we will load a tool called **Babel** directly in the browser. Babel will handle this transpilation for us automatically, on the fly.

---

## Step-by-Step Instructions

### Step 1: Create the `index.html` File

This file will be the container for your React component.

1.  Navigate to the **root directory** of your `AI_Driven_Software_Engineering` project.
2.  Create a new file and name it `index.html`.
3.  Copy and paste the following code into your new `index.html` file:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>React Component Test</title>
    
    <!-- 1. Load React libraries -->
    <script src="[https://unpkg.com/react@18/umd/react.development.js](https://unpkg.com/react@18/umd/react.development.js)"></script>
    <script src="[https://unpkg.com/react-dom@18/umd/react-dom.development.js](https://unpkg.com/react-dom@18/umd/react-dom.development.js)"></script>

    <!-- 2. Load Babel to transpile JSX -->
    <script src="[https://unpkg.com/@babel/standalone/babel.min.js](https://unpkg.com/@babel/standalone/babel.min.js)"></script>
    
    <!-- 3. (Optional) Load Tailwind CSS if your component uses it -->
    <script src="[https://cdn.tailwindcss.com](https://cdn.tailwindcss.com)"></script>

    <style>
        /* Basic styling to center the component on the page */
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            background-color: #f0f2f5;
            padding: 1rem;
            box-sizing: border-box;
        }
    </style>
</head>
<body>
    <!-- 4. This is the root element where your component will be rendered -->
    <div id="root"></div>

    <!-- 5. IMPORTANT: Link to your generated .jsx file -->
    <!-- The type="text/babel" attribute is essential! -->
    <script type="text/babel" src="app/day8_login_refactored.jsx"></script> 
    
</body>
</html>
```

#### **Action Required:** Update the File Path

In the code above, locate the final `<script>` tag. You **must** change the `src` attribute to point to the `.jsx` file you wish to view.

For example, if you are working on the weather app from the self-paced lab, you would change it to:
`<script type="text/babel" src="app/day8_sp_weather_ui.jsx"></script>`

### Step 2: Start a Local Web Server

You need to serve the `index.html` file from a local web server to view it correctly. Here are two easy options.

#### **Option 1: Python's Built-in `http.server` (Recommended)**

This is the recommended method as Python is a prerequisite for the course.

1.  Open your terminal or command prompt.
2.  Make sure you are in the **root directory** of your `AI_Driven_Software_Engineering` project (the same location as your `index.html` file).
3.  Run the following command:
    ```bash
    python -m http.server
    ```
4.  The server will start, and you will see a message like this:
    ```
    Serving HTTP on 0.0.0.0 port 8000 ([http://0.0.0.0:8000/](http://0.0.0.0:8000/)) ...
    ```

#### **Option 2: Node.js `npx serve` (Alternative)**

If you have Node.js installed, this is another excellent and simple option.

1.  Open your terminal in the root directory of your project.
2.  Run the following command:
    ```bash
    npx serve
    ```
    *(`npx` is a tool included with Node.js that runs packages without a global installation.)*
3.  The server will start and provide you with a local URL:
    ```
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │   Serving!                                       │
    │                                                  │
    │   - Local:    http://localhost:3000              │
    │                                                  │
    └──────────────────────────────────────────────────┘
    ```

### Step 3: View Your Component in the Browser

1.  Open your web browser (Chrome, Firefox, etc.).
2.  In the address bar, type the URL from the server you started:
    * If you used Python: `localhost:8000`
    * If you used `npx serve`: `localhost:3000`
3.  Press Enter. You should now see your React component rendered on the page!

If you make any changes to your `.jsx` file, simply save the file and refresh the browser page to see the updates.

---

### Which Option Should I Choose?

Both options accomplish the same goal: they start a simple, local web server from your project directory. For this course, either one is a great choice. Here’s a quick comparison to help you decide.

|              | **Python `http.server`** | **Node.js `npx serve`** |
| :----------- | :----------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- |
| **Pros** | ✅ Already installed (comes with Python).<br>✅ Zero configuration needed.<br>✅ Perfect for simple file serving. | ✅ More user-friendly output.<br>✅ Automatically copies the URL to your clipboard.<br>✅ Can be slightly faster with more features. |
| **Cons** | ❌ Very basic output.<br>❌ Doesn't automatically open the browser.                                      | ❌ Requires Node.js and npm to be installed.<br>❌ Downloads the `serve` package on first run.                          |

**Recommendation:** If you're looking for the absolute fastest way to get started, use the **Python `http.server`**. It's guaranteed to be available on your system. If you already have Node.js installed and prefer a slightly more polished experience, **`npx serve`** is an excellent alternative.

### Troubleshooting

* **"My component is not showing up / I see a blank page."**
    * Double-check the `src` path in your `index.html` file. Make sure it correctly points to your `.jsx` file.
    * Open the browser's developer console (usually by pressing `F12` or `Ctrl+Shift+I`). Look for any error messages in the "Console" tab. This will often tell you if there's a typo in your path or a syntax error in your JSX.
* **"I see my JSX code on the page instead of the component."**
    * Make sure you have included `type="text/babel"` in the script tag that links your `.jsx` file. This attribute is what tells the Babel library to transpile your code.
* **"The styling looks wrong."**
    * If your component uses Tailwind CSS, ensure you have included the Tailwind CDN script in the `<head>` of your `index.html` file (it's included in the template above).
