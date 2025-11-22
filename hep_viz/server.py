from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from pathlib import Path
from .data_processor import DataProcessor

app = FastAPI()

# Get data path from env var set by CLI
DATA_PATH = os.environ.get("HEP_VIZ_DATA_PATH")
processor = None

@app.on_event("startup")
async def startup_event():
    global processor
    if DATA_PATH:
        print(f"Initializing DataProcessor with path: {DATA_PATH}")
        processor = DataProcessor(DATA_PATH)
    else:
        print("Warning: HEP_VIZ_DATA_PATH not set.")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Serve the index.html
    # In a real package, we might want to use Jinja2Templates or just read the file
    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        return template_path.read_text()
    return "<h1>hep-viz: Template not found</h1>"

@app.get("/api/events")
async def get_events():
    if not processor:
        raise HTTPException(status_code=500, detail="DataProcessor not initialized")
    return processor.get_event_list()

@app.get("/api/event/{event_id}")
async def get_event(event_id: str):
    """
    Get processed data for a specific event.
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
    except Exception as e:
        print(f"Error processing event {event_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/shutdown")
async def shutdown():
    """
    Shutdown the server.
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
