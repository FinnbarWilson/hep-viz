import pytest
from fastapi.testclient import TestClient
from hep_viz.server import app, set_processor
from hep_viz.data_processor import DataProcessor

@pytest.fixture
def client(sample_memory_data):
    """
    Create a TestClient with a pre-initialized DataProcessor.
    """
    processor = DataProcessor(sample_memory_data)
    set_processor(processor)
    return TestClient(app)

def test_read_root(client):
    """
    Test the root endpoint returns HTML.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_get_events(client):
    """
    Test /api/events endpoint.
    """
    response = client.get("/api/events")
    assert response.status_code == 200
    data = response.json()
    assert "events" in data
    assert data["events"] == [0, 1]

def test_get_event_detail(client):
    """
    Test /api/event/{id} endpoint.
    """
    response = client.get("/api/event/0")
    assert response.status_code == 200
    data = response.json()
    
    assert "tracks" in data
    assert len(data["tracks"]) == 2
    assert data["tracks"][0]["particle_id"] == 1

def test_get_event_not_found(client):
    """
    Test requesting a non-existent event.
    """
    response = client.get("/api/event/999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]

def test_server_shutdown(client):
    """
    Test the shutdown endpoint.
    Note: We can't fully test the signal killing in this environment, 
    but we can check the response.
    """
    response = client.post("/shutdown")
    assert response.status_code == 200
    assert "shutting down" in response.json()["message"]
