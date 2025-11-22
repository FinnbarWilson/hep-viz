# Publishing hep-viz

## Prerequisites

- `build`
- `twine`

## Build

```bash
python -m build
```

## Upload to PyPI

```bash
twine upload dist/*
```

## Manual Verification

Before publishing, verify the package locally:

1.  Create a fresh virtual environment.
2.  Install the package: `pip install .`
3.  Run the CLI against a sample dataset: `hep-viz view /path/to/data`
