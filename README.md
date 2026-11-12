# sip

**S**weet maria's **I**nventory **P**icker

A command-line toolkit for exploring and selecting green coffee from
[Sweet Maria's](https://www.sweetmarias.com/). Scrapes the current catalog,
builds a local database of cupping scores and flavor profiles, then helps you
navigate the flavor space to help you decide what to buy.

## What it does

- **Scrape** the full Sweet Maria's green coffee catalog (scores, flavor notes,
  farm details, availability)
- **Recommend** coffees that maximize flavor variation from beans you've already
  tried
- **Profile** any coffee — see its dimension rankings, archetype mix, contrast
  pair (antipode), cluster placement, and nearest in-stock substitute
- **Explore** high/low pairs per flavor dimension to isolate specific taste
  characteristics
- **Cluster** the catalog into flavor families with k-means and silhouette
  analysis
- **HTML report** — a single interactive page with UMAP scatter (text-weight
  slider), archetype radars, correlation heatmap, score distributions, PCA
  loadings, processing profiles, explore pairs, and superlatives
  ([example](https://bsherrod.github.io/sip/examples/flavor-report.html))
- **Text embeddings** to incorporate cupping-note semantics into distance
  calculations and map layout
- **Factor analysis** to reveal latent flavor themes and gaps in your
  exploration
- **Archetype analysis** to identify extreme flavor styles and decompose coffees
  as mixtures

## Quick start

Needs Python 3.11 or newer. Dependencies are declared in `pyproject.toml`.

### With [uv](https://docs.astral.sh/uv/) (nothing to install by hand)

The scripts carry a `#!/usr/bin/env -S uv run` shebang, so uv creates the
virtualenv and installs dependencies on first run, from anywhere:

```bash
./scrape_sm.py                  # scrape the current catalog
./coffee.py recommend
```

The first run has to download the dependency set, and `sentence-transformers`
pulls in PyTorch, so expect it to take a while. Later runs add roughly a tenth of
a second of overhead.

### Without uv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python scrape_sm.py
python coffee.py recommend
```

Use `pip install -e .`, not `pip install .`. The database path is resolved
relative to the source file, so a non-editable install would try to keep
`coffees.db` inside `site-packages`.

The spaCy `en_core_web_sm` model installs automatically with everything else —
it is declared in `pyproject.toml` as a direct reference to its release wheel,
so there is no separate `python -m spacy download` step. If you ever bump the
spaCy minor version, bump the model URL to match.

### Then

```bash
./coffee.py compare "ethiopia natural"    # profile a specific coffee
./coffee.py html                          # full interactive report
./coffee.py html --slider-steps 11        # fewer UMAP frames (faster)
```

## Commands

| Command | Description |
|---------|-------------|
| `coffee.py recommend` | Suggest next beans (max variation from tried) |
| `coffee.py profile <query>` | Full coffee dossier: rankings, archetypes, antipode |
| `coffee.py compare <query>` | Alias for `profile` |
| `coffee.py explore` | Find contrast pairs to isolate dimensions |
| `coffee.py insights` | Outliers, clusters, superlatives |
| `coffee.py factors` | PCA factor analysis of latent themes |
| `coffee.py archetypes` | Extreme styles and mixture decomposition |
| `coffee.py html` | Interactive HTML report with all visualizations |
| `coffee.py map` | Alias for `html` |
| `embed.py build` | Compute sentence embeddings for cupping notes |
| `embed.py status` | Show embedding coverage |

All commands support `--help` for detailed options.

## Text embeddings

The `embed.py` tool uses sentence-transformers (`all-MiniLM-L6-v2`) to encode
each coffee's cupping notes into a 384-dimensional vector. These embeddings
capture semantic flavor similarity beyond what numeric scores alone convey.

Text embeddings are built automatically when needed — any `coffee.py` command
will detect missing embeddings and compute them on first run. You can also
manage them explicitly with `embed.py build` and `embed.py status`.

Once built, embeddings unlock two features:

- **`--text-weight W`** (global flag, 0.0–1.0) — blends text cosine similarity
  into the distance function used by `recommend` and `compare`.
- **UMAP slider** (in `html` report) — pre-computes UMAP layouts at multiple
  text-weight steps (0→1) with Procrustes alignment so you can smoothly see how
  text semantics reshape the flavor map.

## Database

The scraper populates a local SQLite database (`coffees.db`), kept next to the
source files. This file is gitignored — run `scrape_sm.py` to build your own
fresh copy from the current catalog.

## License

MIT
