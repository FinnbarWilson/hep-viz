import pytest
import pandas as pd
import numpy as np
from hep_viz.data_processor import DataProcessor

def test_init_from_memory(sample_memory_data):
    """
    Test initialization from an in-memory dictionary.
    """
    processor = DataProcessor(sample_memory_data)
    
    assert processor.memory_data == sample_memory_data
    assert len(processor.event_index_map) == 2
    assert 0 in processor.event_index_map
    assert 1 in processor.event_index_map
    
    # Check index mapping
    assert processor.event_index_map[0]['particles'] == 0
    assert processor.event_index_map[0]['tracker_hits'] == 0 # First hit for event 0
    
def test_get_event_list_memory(sample_memory_data):
    """
    Test retrieving the event list from memory data.
    """
    processor = DataProcessor(sample_memory_data)
    event_list = processor.get_event_list()
    
    assert "events" in event_list
    assert event_list["events"] == [0, 1]

def test_process_event_memory(sample_memory_data):
    """
    Test processing a specific event from memory.
    """
    processor = DataProcessor(sample_memory_data)
    
    # Process Event 0
    data = processor.process_event("0")
    
    assert "error" not in data
    assert "tracks" in data
    assert "calo_hits" in data
    assert "all_tracker_hits" in data
    
    # Check Particles/Tracks
    # Event 0 has 2 particles: ID 1 (Electron) and ID 2 (Muon)
    # Both should have tracks (tracker hits exist for both)
    assert len(data['tracks']) == 2
    
    track1 = next(t for t in data['tracks'] if t['particle_id'] == 1)
    assert track1['pdg_id'] == 11
    assert len(track1['points']) == 2
    
    # Check Calo Hits
    # Event 0 has 1 calo hit linked to particle 1
    assert len(data['calo_hits']) == 1
    assert data['calo_hits'][0]['particle_id'] == 1
    assert data['calo_hits'][0]['energy'] == 10.0

def test_load_data_files(mock_parquet_dir):
    """
    Test scanning and loading files from a directory.
    """
    processor = DataProcessor(mock_parquet_dir)
    
    # Check if files were found
    assert 'particles' in processor.files
    assert len(processor.files['particles']) == 2 # We created 2 split files
    
    # Check ranges
    f1 = processor.files['particles'][0]
    assert f1['start'] == 0
    assert f1['end'] == 1
    
    f2 = processor.files['particles'][1]
    assert f2['start'] == 2
    assert f2['end'] == 2

def test_get_event_list_files(mock_parquet_dir):
    """
    Test retrieving event list from files.
    """
    processor = DataProcessor(mock_parquet_dir)
    event_list = processor.get_event_list()
    
    assert "events" in event_list
    # Should find events 0, 1, 2 based on ranges
    assert sorted(event_list["events"]) == [0, 1, 2]

def test_process_event_files(mock_parquet_dir):
    """
    Test processing an event from files.
    """
    processor = DataProcessor(mock_parquet_dir)
    
    # Process Event 0 (Should be in first file)
    data = processor.process_event("0")
    
    assert "error" not in data
    # We added 2 hits for particle 1 in conftest, so it should form a track
    assert len(data['tracks']) == 1 
    assert data['tracks'][0]['particle_id'] == 1
    
    # Process Event 2 (Should be in second file)
    data2 = processor.process_event("2")
    assert "error" not in data2
    # Event 2 has particle 1 but no hits in our mock data
    assert len(data2['tracks']) == 0 

def test_missing_event(sample_memory_data):
    """
    Test handling of non-existent events.
    """
    processor = DataProcessor(sample_memory_data)
    
    result = processor.process_event("999")
    assert "error" in result
    assert "not found" in result["error"]

def test_invalid_event_id(sample_memory_data):
    """
    Test handling of invalid event ID format.
    """
    processor = DataProcessor(sample_memory_data)
    
    with pytest.raises(ValueError):
        processor.process_event("abc")
