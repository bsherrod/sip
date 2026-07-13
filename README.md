# sip

**S**weet maria's **I**nventory **P**icker

A command-line toolkit for exploring and selecting green coffee from
[Sweet Maria's](https://www.sweetmarias.com/). Scrapes the current catalog,
builds a local database of cupping scores and flavor profiles, then helps you
navigate the flavor space to find your next bag.

## What it does

- **Scrape** the full Sweet Maria's green coffee catalog (scores, flavor notes,
  farm details, availability)
- **Recommend** coffees that maximize flavor variation from beans you've already
  tried
- **Compare** any coffee against your collection — see where it sits in the
  flavor space
- **Explore** high/low pairs per flavor dimension to isolate specific taste
  characteristics
- **Cluster** the catalog into flavor families with k-means and silhouette
  analysis
- **Map** the entire flavor space onto a 2D UMAP visualization with archetype
  coloring, center/contrast markers, and an interactive text-weight slider
  ([example](https://htmlpreview.github.io/?https://github.com/bsherrod/sip/blob/main/examples/flavor-map-slider.html))
- **Text embeddings** to incorporate cupping-note semantics into distance
  calculations and map layout
- **Factor analysis** to reveal latent flavor themes and gaps in your
  exploration
- **Archetype analysis** to identify extreme flavor styles and decompose coffees
  as mixtures

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Scrape the current catalog
python scrape_sm.py

# Get recommendations
python coffee.py recommend

# Compare a specific coffee
python coffee.py compare "ethiopia natural"

# Generate flavor maps
python coffee.py map                          # numeric only (PNG + HTML)
python coffee.py map --text-mode blended      # hybrid distance matrix
python coffee.py map --text-mode concat       # concatenated PCA vectors
python coffee.py map --slider                 # interactive text-weight slider
```

## Commands

| Command | Description |
|---------|-------------|
| `coffee.py recommend` | Suggest next beans (max variation from tried) |
| `coffee.py compare <query>` | Place a coffee in your flavor landscape |
| `coffee.py explore` | Find contrast pairs to isolate dimensions |
| `coffee.py insights` | Outliers, clusters, superlatives |
| `coffee.py factors` | PCA factor analysis of latent themes |
| `coffee.py archetypes` | Extreme styles and mixture decomposition |
| `coffee.py map` | 2D UMAP flavor map as PNG + interactive HTML |
| `coffee.py map --slider` | Interactive HTML with text-weight slider |
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
- **`--text-mode MODE`** (for `map`) — controls how text influences the UMAP
  layout:
  - `none` — numeric PCA scores only (default, original behavior)
  - `blended` — precomputed distance matrix mixing numeric L2 + text cosine
  - `concat` — PCA-reduced text embeddings concatenated with numeric scores

## Database

The scraper populates a local SQLite database (`coffees.db`). This file is
gitignored — run `scrape_sm.py` to build your own fresh copy from the current
catalog.

## License

MIT
