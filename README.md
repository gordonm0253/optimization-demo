# Optimizer Visualization Demo

A standalone HTML/CSS/JavaScript project for visualizing SGD, SGD with Momentum, and Adam on a 3D loss surface.

## Files

- `index.html` — page structure and controls
- `styles.css` — layout and visual styling
- `app.js` — Three.js scene, loss surface, optimizers, animation, and interactions

## Run locally

Open `index.html` in a browser. If your browser blocks external scripts from local files, run a local server:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

The project uses Three.js from a CDN.
