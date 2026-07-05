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
- **Map** the entire flavor space onto a 2D UMAP visualization
- **Factor analysis** to reveal latent flavor themes and gaps in your
  exploration
- **Archetype analysis** to identify extreme flavor styles and decompose coffees
  as mixtures

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Scrape the current catalog
python scrape_sm.py

# Get recommendations
python coffee.py recommend

# Compare a specific coffee
python coffee.py compare "ethiopia natural"

# Generate a flavor map
python coffee.py map
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
| `coffee.py map` | 2D UMAP flavor map as PNG |

All commands support `--help` for detailed options.

## Database

The scraper populates a local SQLite database (`coffees.db`). This file is
gitignored — run `scrape_sm.py` to build your own fresh copy from the current
catalog.

## License

MIT
