# Experimental Lab Courses — GitHub Pages site

A polished, responsive course hub for the open-source labs in
[`karlb-dev/labs`](https://github.com/karlb-dev/labs). It includes:

- `/` — a data-driven catalog for every course
- `/mechanistic-interpretability/` — the 36-lab mechanistic interpretability course
- `/collective-communication/` — the 11-lab TPU collective communication course

The site is deliberately build-free. GitHub Pages serves the files in `site/`
directly, so there are no runtime dependencies or package updates to maintain.

## Preview locally

```bash
python3 -m http.server 8000 --directory site
```

Then open `http://localhost:8000`.

## Publish with GitHub Pages

Push this project to a GitHub repository. The included workflow publishes on
every push to `main`:

1. In the repository, open **Settings → Pages**.
2. Under **Build and deployment**, choose **GitHub Actions**.
3. Push to `main` or run **Deploy course site to GitHub Pages** manually.

All asset and page links are relative, so the site works as a user site, under
`/repository-name/`, or behind a custom domain.

## Add another course

1. Add one entry to `site/courses.js`.
2. Add a folder such as `site/new-course/` with its `index.html` and optional
   JavaScript.
3. Reference shared assets as `../styles.css` and `../favicon.svg`.

The landing-page card and total course count are generated from
`site/courses.js`; no landing-page HTML changes are needed.

## Content sources

- [Mechanistic interpretability labs](https://github.com/karlb-dev/labs/tree/main/interpretability)
- [Collective communication labs](https://github.com/karlb-dev/labs/tree/main/collective_communication)
