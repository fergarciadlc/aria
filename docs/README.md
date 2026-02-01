# ARIA Documentation

## For Users

- [Main README](../README.md) — Installation, quick start, API reference

## For Dataset Maintainers

- [Packaging Guide](packaging.md) — Creating distribution variants for Zenodo
- [Scripts README](../scripts/README.md) — Dataset generation pipeline

## CLI Reference

```bash
# Dataset info
python -m aria info /path/to/aria --detailed

# Validation
python -m aria validate /path/to/aria

# Packaging
python -m aria package --source /full/aria --output /releases/tiny --tier tiny

# List tiers
python -m aria tiers
```


