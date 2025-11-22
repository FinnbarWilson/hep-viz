# hep-viz

A local visualizer for High Energy Physics (HEP) data.

## Overview

`hep-viz` is a CLI tool that spins up a local web server to visualize high-dimensional particle tracking and calorimeter data. It bridges the gap between heavy local datasets and lightweight web visualization by streaming optimized data on demand.

## Installation

```bash
pip install .
```

## Usage

```bash
hep-viz view /path/to/data/directory
```

Options:
- `--port`: Specify port (default 8000).
- `--browser/--no-browser`: Auto-open browser.

## Data Format

The tool expects a directory containing Parquet files for:
- Particles (`*particles*.parquet`)
- Tracks (`*tracks*.parquet`)
- Tracker Hits (`*tracker_hits*.parquet`)
- Calorimeter Hits (`*calo_hits*.parquet`)

## Development

1.  Clone the repository.
2.  Install dependencies: `pip install -e .`
3.  Run tests: `pytest tests/`
