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
    if not processor:
        raise HTTPException(status_code=500, detail="DataProcessor not initialized")
    try:
        data = processor.process_event(event_id)
        return data
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
