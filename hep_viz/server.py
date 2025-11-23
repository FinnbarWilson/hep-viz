from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from pathlib import Path
from .data_processor import DataProcessor

app = FastAPI()

# Global processor instance.
# This will be initialized either via environment variable (CLI mode)
# or directly via set_processor (Python API mode).
processor = None

@app.on_event("startup")
async def startup_event():
    """
    Initialize the DataProcessor on server startup.
    Checks for the HEP_VIZ_DATA_PATH environment variable if the processor is not already set.
    """
    global processor
    if processor is None: 
        # Re-fetch env var inside startup event to ensure it's captured
        env_path = os.environ.get("HEP_VIZ_DATA_PATH")
        if env_path:
            print(f"Initializing DataProcessor with path: {env_path}")
            processor = DataProcessor(env_path)
        else:
            print("Warning: HEP_VIZ_DATA_PATH not set and no processor provided.")

def set_processor(proc):
    """
    Manually set the DataProcessor instance.
    Used when running from the Python API.
    """
    global processor
    processor = proc

def run_server(host="127.0.0.1", port=8000):
    """
    Run the Uvicorn server programmatically.
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serve the main application page (index.html).
    """
    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        return template_path.read_text()
    return "<h1>hep-viz: Template not found</h1>"

@app.get("/api/events")
async def get_events():
    """
    Get the list of available event IDs.
    """
    if not processor:
        raise HTTPException(status_code=500, detail="DataProcessor not initialized")
    return processor.get_event_list()

@app.get("/api/event/{event_id}")
async def get_event(event_id: str):
    """
    Get processed data for a specific event ID.
    Returns JSON containing particles, tracks, and hits.
    """
    if not processor:
        raise HTTPException(status_code=500, detail="DataProcessor not initialized")
    try:
        data = processor.process_event(event_id)
        if "error" in data:
             raise HTTPException(status_code=404, detail=data["error"])
        return data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing event {event_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/shutdown")
async def shutdown():
    """
    Gracefully shutdown the server.
    """
    import os
    import signal
    import threading
    import time

    def kill_server():
        time.sleep(1) # Give time for the response to be sent
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=kill_server).start()
    return {"message": "Server shutting down..."}
