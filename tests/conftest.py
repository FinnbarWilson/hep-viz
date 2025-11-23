import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path

@pytest.fixture
def sample_memory_data():
    """
    Returns a dictionary mimicking a Hugging Face dataset structure (one row per event, list columns).
    """
    # Event 0: 2 particles (Electron, Muon)
    event0_particles = {
        'event_id': 0,
        'particle_id': [1, 2],
        'pdg_id': [11, 13],
        'px': [1.0, 0.0],
        'py': [0.0, 1.0],
        'vx': [0.0, 0.0],
        'vy': [0.0, 0.0]
    }
    
    event0_tracker = {
        'event_id': 0,
        'particle_id': [1, 1, 2, 2], # Added hit for particle 2
        'x': [10.0, 20.0, 0.0, 0.0],
        'y': [0.0, 1.0, 10.0, 11.0],
        'z': [0.0, 1.0, 0.0, 1.0],
        'volume_id': [7, 8, 7, 7]
    }
    
    event0_calo = {
        'event_id': 0,
        'cell_id': [100],
        'x': [100.0], 'y': [0.0], 'z': [0.0],
        'total_energy': [10.0],
        'contrib_particle_ids': [[1]], # List of lists
        'contrib_energies': [[10.0]],
        'contrib_times': [[0.1]]
    }

    # Event 1: 1 particle (Photon)
    event1_particles = {
        'event_id': 1,
        'particle_id': [1],
        'pdg_id': [22],
        'px': [0.5], 'py': [0.5], 'vx': [0.0], 'vy': [0.0]
    }
    
    event1_tracker = {
        'event_id': 1,
        'particle_id': [1],
        'x': [5.0], 'y': [5.0], 'z': [5.0],
        'volume_id': [7]
    }
    
    event1_calo = {
        'event_id': 1,
        'cell_id': [101],
        'x': [50.0], 'y': [50.0], 'z': [50.0],
        'total_energy': [5.0],
        'contrib_particle_ids': [[1]],
        'contrib_energies': [[5.0]],
        'contrib_times': [[0.2]]
    }

    return {
        'particles': [event0_particles, event1_particles],
        'tracker_hits': [event0_tracker, event1_tracker],
        'calo_hits': [event0_calo, event1_calo]
    }

@pytest.fixture
def mock_parquet_dir(tmp_path):
    """
    Creates a temporary directory with mock Parquet files.
    Uses list columns to match real data structure.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Particles: List columns
    p_data = {
        'event_id': [0, 1, 2],
        'particle_id': [[1, 2], [1], [1]],
        'pdg_id': [[11, 13], [22], [11]],
        'px': [[1.0, 0.0], [0.5], [0.5]],
        'py': [[0.0, 1.0], [0.5], [0.5]],
        'vx': [[0.0, 0.0], [0.0], [0.0]],
        'vy': [[0.0, 0.0], [0.0], [0.0]]
    }
    particles_df = pd.DataFrame(p_data)
    
    # Split particles
    p_file1 = data_dir / "particles.events0-1.parquet"
    particles_df[particles_df['event_id'] <= 1].to_parquet(p_file1)
    
    p_file2 = data_dir / "particles.events2-2.parquet"
    particles_df[particles_df['event_id'] == 2].to_parquet(p_file2)
    
    # Tracker Hits
    t_data = {
        'event_id': [0],
        'particle_id': [[1, 1]], # 2 hits for particle 1
        'x': [[10.0, 20.0]], 'y': [[0.0, 1.0]], 'z': [[0.0, 1.0]],
        'volume_id': [[7, 8]]
    }
    pd.DataFrame(t_data).to_parquet(data_dir / "tracker_hits.parquet")
    
    # Calo Hits
    c_data = {
        'event_id': [0],
        'cell_id': [[100]],
        'x': [[100.0]], 'y': [[0.0]], 'z': [[0.0]],
        'total_energy': [[10.0]],
        'contrib_particle_ids': [[[1]]], # List of lists of lists (Event -> Hit -> Contributions)
        'contrib_energies': [[[10.0]]],
        'contrib_times': [[[0.1]]]
    }
    pd.DataFrame(c_data).to_parquet(data_dir / "calo_hits.parquet")
    
    return data_dir
