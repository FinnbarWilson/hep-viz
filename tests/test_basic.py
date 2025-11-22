import sys
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add package to path
sys.path.append(str(Path(__file__).parent.parent))

from hep_viz.server import app
from hep_viz.data_processor import DataProcessor

# Mock data path
# We need to point to the reference data for testing
DATA_PATH = "/Users/finn/Documents/Code/Large_Datasets"

def test_data_processor_init():
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Data path {DATA_PATH} does not exist")
    
    dp = DataProcessor(DATA_PATH)
    assert dp.particles_df is not None
    assert dp.tracks_df is not None

def test_get_event_list():
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Data path {DATA_PATH} does not exist")
        
    dp = DataProcessor(DATA_PATH)
    events = dp.get_event_list()
    assert "events" in events
    assert isinstance(events["events"], list)
    assert len(events["events"]) > 0

def test_process_event():
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Data path {DATA_PATH} does not exist")
        
    dp = DataProcessor(DATA_PATH)
    event_data = dp.process_event("0")
    
    assert "tracks" in event_data
    assert "calo_hits" in event_data
    assert "metadata" in event_data
    assert event_data["metadata"]["has_pdg"] is True

def test_api_root():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "hep-viz" in response.text

def test_api_events():
    # We need to initialize the processor in the app
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Data path {DATA_PATH} does not exist")
        
    # Manually init processor for the app
    from hep_viz import server
    server.processor = DataProcessor(DATA_PATH)
    
    client = TestClient(app)
    response = client.get("/api/events")
    assert response.status_code == 200
    data = response.json()
    assert "events" in data
