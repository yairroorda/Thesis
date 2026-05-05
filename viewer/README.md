# viewer_potree

Minimal COPC viewer that wraps Potree.

Potree upstream code is vendored in `third_party/potree`.
Custom project code lives only in `viewer_potree`.

Run

1. Start a static server from `Thesis/` with Range support (required for taking advantage of COPC):

For example:

```bash
cd /home/yair/Desktop/Thesis/Thesis
./.pixi/envs/thesis/bin/python -m RangeHTTPServer 8000
```

2. Open:
- `http://localhost:8000/`

The repository root now redirects to `viewer/`, so the viewer opens directly.

Use the demo dropdown to switch between Groningen and Delft. You can also open `http://localhost:8000/?demo=delft` directly.