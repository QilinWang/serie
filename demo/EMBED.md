# SERIE Demo Embed Contract

This folder builds the React/React Three Fiber attractor viewer as a static
artifact that another site can copy or iframe. Astro should treat the built demo
as an asset folder first, not as source to rewrite.

## Build

From `serie/demo/`:

```bash
bun run bake
bun run build
```

`bun run bake` writes:

```text
public/attractor_data.json
```

`bun run build` writes:

```text
dist/index.html
dist/attractor_data.json
dist/assets/*
```

The checked build on this machine produced a `dist/` folder around 24 MB. Most
of that is `attractor_data.json`; the JavaScript bundle is also large because it
contains React, Three.js, and React Three Fiber.

## Embed Path

The build is portable because `vite.config.ts` sets:

```ts
base: "./"
```

The emitted `index.html` loads assets through relative paths such as:

```html
<script type="module" crossorigin src="./assets/index-....js"></script>
<link rel="stylesheet" crossorigin href="./assets/index-....css">
```

The browser app also fetches:

```text
attractor_data.json
```

from the same directory as `index.html`.

## Astro Copy Contract

If Claude's Astro project lives at `site/`, copy the build output to:

```text
site/public/demo/
```

Then the public URL is:

```text
/demo/index.html
```

An Astro page can embed it with:

```astro
<iframe
  src="/demo/index.html"
  title="SERIE attractor demo"
  loading="lazy"
></iframe>
```

For the first integration pass, prefer this static-folder contract over
importing the React components directly into Astro. Direct React islands can
come later if the site needs tighter layout control.

## Smaller Payloads

For a lighter page, rebake fewer points or fewer noise draws:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with-editable .. \
  python bake_demo_data.py --n-points 2000 --n-draws 12
bun run build
```

Keep the output filename as `public/attractor_data.json`; the frontend expects
that name when built as a static folder.

## Two engines: Python `serie` vs `serie-rs`

The viewer is engine-agnostic — it only renders `noisy = clean + sigma * draw`
from `attractor_data.json`. Two engines bake that exact contract:

```bash
bun run bake      # Python `serie`  → public/attractor_data.json
bun run bake:rs   # Rust `serie-rs` → public/attractor_data.json (serie-rs demo shapes)
```

`bun run bake:rs` calls the `serie-rs` CLI (`cargo run -q --manifest-path
../../serie-rs/Cargo.toml -- demo shapes`). Bake with whichever engine, then
`bun run build`; the build is identical otherwise.

The Fern site ships **both** as separate pages:

- `/demo/serie`    — Python build, embedded from its standalone deployment
  (`site/src/pages/demo/serie.astro`, external iframe URL).
- `/demo/serie-rs` — Rust build, **self-hosted** under `site/public/demo-rs/`
  (`site/src/pages/demo/serie-rs.astro`, iframes `/demo-rs/index.html`).

To refresh the self-hosted serie-rs page:

```bash
cd serie/demo
bun run bake:rs && bun run build
rm -rf ../../site/public/demo-rs && mkdir -p ../../site/public/demo-rs
cp -r dist/* ../../site/public/demo-rs/
# (then restore the Python data file if you also serve /demo/serie locally)
```

Byte-parity with the Python bake is **not** required — the serie-rs page is a
valid Rust-generated attractor in its own right.
