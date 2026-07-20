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

# Generate interactive HTML report
python coffee.py html                         # full report with all visualizations
python coffee.py html --slider-steps 11       # fewer UMAP frames (faster)
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

The scraper populates a local SQLite database (`coffees.db`). This file is
gitignored — run `scrape_sm.py` to build your own fresh copy from the current
catalog.

## License

MIT
