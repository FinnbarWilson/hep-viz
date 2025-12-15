# hep-viz

**hep-viz** is a lightweight, local 3D event visualizer designed specifically for the new [ColliderML](https://huggingface.co/datasets/CERN/ColliderML-Release-1) datasets. It provides an interactive interface to explore High Energy Physics (HEP) events, visualize particle tracks, calorimeter hits, and inspect Graph Neural Network (GNN) outputs such as edge scores and truth labels.

![hep-viz_event](Images/hep-viz_event.png)

It is built for researchers and developers working with ColliderML data, supporting both standard event visualization and graph-based GNN inspection.

## Installation

### pip
You can install `hep-viz` directly from PyPI:

```bash
pip install hep-viz
```

### From Source
```bash
git clone https://github.com/your-username/hep-viz.git
cd hep-viz
pip install .
```

### Conda

```bash
conda install finnbarwilson::hep-viz
```

## How to Use

`hep-viz` can be used either through its Command Line Interface (CLI) for quick file inspection or via its Python API for integration into notebooks and scripts.

### 1. Command Line Interface (CLI)

The CLI is perfect for viewing data stored in local directories (e.g., Parquet files).

```bash
hep-viz view /path/to/data/directory
```

**Options:**
- `--port <int>`: Specify the port to run the server on (default: `8000`).
- `--browser / --no-browser`: control whether the browser opens automatically.

#### SSH Port Forwarding
If you are running `hep-viz` on a remote server via SSH, the browser cannot open automatically. You will need to forward the port to your local machine.

**Standard SSH Forwarding:**
Run this in a **new terminal** on your local machine:
```bash
ssh -L 8000:localhost:8000 user@remote-host
```

**Nested SSH Forwarding (SSH inside SSH):**
If you are connected to a jump host (login node) and then to a compute node:

1.  **On your local machine:** Forward port 8000 to the login node.
    ```bash
    ssh -L 8000:localhost:8000 user@login-node
    ```
2.  **On the login node:** Forward port 8000 to the compute node where `hep-viz` is running.
    ```bash
    ssh -L 8000:localhost:8000 user@compute-node
    ```
3.  Open `http://localhost:8000` in your local browser.

### 2. Python API

You can use `hep-viz` directly within your Python scripts.

#### Standard Event Visualization (`view`)

Pass a dictionary where each key is a **list of dictionaries** (records) or a **Hugging Face Dataset**.

```python
import hep_viz

# Data is a dictionary of lists (records) or Datasets
event_data = {
    "particles": [
        {"event_id": 0, "particle_id": 1, "pdg_id": 11, ...}, # Event 0
        {"event_id": 1, "particle_id": 1, "pdg_id": 13, ...}  # Event 1
    ],
    "tracker_hits": [
        {"event_id": 0, "x": 10.1, "y": 20.2, "z": 30.3, "particle_id": 1},
        ...
    ],
    "calo_hits": [ ... ],
    
    # Standard Reconstruction Tracks
    "tracks": [
        {
            "event_id": 0,
            "track_id": 1,
            "majority_particle_id": 1,  # CRITICAL: Links track to Truth Particle
            "d0": 0.1, 
            "z0": 1.2
        },
        ...
    ]
}

# Launch the visualizer
hep_viz.view(event_data, port=8000)
```

#### Graph/GNN Visualization (`view_graph`)
Use this for visualizing GNN outputs (edges and scores).

```python
import hep_viz

hep_viz.view_graph(
    hits=hits_df,       # DataFrame/Dict with x, y, z
    edges=edges_array,  # 2xN array of indices
    scores=scores_array, # Score per edge
    truth=truth_array,   # (Optional) Truth label per edge
    port=8001
)
```

## Input Data Requirements

`hep-viz` expects key names matching the **ColliderML** schema.

### Standard Inputs
- `particles`: List of particle dictionaries. Must contain `event_id`, `particle_id`, `pdg_id`.
- `tracker_hits`: List of hit dictionaries. Must contain `event_id`, `x`, `y`, `z`, `particle_id`.
    - **Note:** `hep-viz` builds the 3D track visualization by grouping these hits by `particle_id`.
- `calo_hits`: List of calorimeter hit dictionaries. Must contain `event_id`, `energy`, `x`, `y`, `z`.

### Track Inputs (Reconstruction)
To visualize reconstruction performance, you provide a `tracks` list. `hep-viz` uses this to annotate the truth tracks (e.g., highlighting which particles were successfully reconstructed).

*   `event_id`: ID of the event this track belongs to.
*   `majority_particle_id`: **(Required)** The `particle_id` of the truth particle this track matches. This is used to link your reconstructed track to the visual truth track.
*   Optional metadata: `d0`, `z0`, `phi`, `theta`, `qop`.

### Comparing Reconstruction Models
`hep-viz` allows you to compare multiple reconstruction algorithms by seeing which truth particles they successfully recovered.

**Rule:** Any key in your data dictionary that starts with `tracks` will be treated as a reconstruction collection.

**Example:**

```python
data = {
    'particles': ...,
    'tracker_hits': ...,
    
    'tracks': standard_tracks_list,       # e.g. Baseline
    'tracks_gnn': my_gnn_tracks_list,     # Your Custom Model
    'tracks_ckf': ckf_tracks_list         # Another Model
}

hep_viz.view(data)
```

In the UI, a dropdown menu 'Reco Algorithm' will appear. Switching between options will update the visualization to show which truth particles are "Found" (colored) vs "Missed" (greyed out) by that specific algorithm.

### File Naming (CLI)

For the CLI to automatically detect files, they should contain the category name (e.g., `my_particles.parquet`). You can split data across multiple files using the pattern `events<start>-<end>` (e.g., `particles.events0-999.parquet`).

---
## License

This project is licensed under the [MIT License](https://github.com/FinnbarWilson/hep-viz/blob/main/LICENSE).

Copyright (c) 2025 Finnbar Wilson

Built for the ColliderML community.
