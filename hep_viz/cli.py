import typer
import uvicorn
import os
from typing import Optional
from pathlib import Path

app = typer.Typer()

@app.command()
def view(
    path: Path = typer.Argument(..., help="Path to the data file or directory containing Parquet files."),
    port: int = typer.Option(8000, help="Port to run the server on."),
    browser: bool = typer.Option(True, help="Automatically open the default web browser."),
):
    """
    Visualize HEP data from a local directory.

    This command starts a local web server to visualize High Energy Physics (HEP) data
    stored in Parquet files. It expects a directory containing subfolders or files
    for 'particles', 'tracks', 'tracker_hits', and 'calo_hits'.
    """
    if not path.exists():
        typer.echo(f"Error: Path '{path}' does not exist.")
        raise typer.Exit(code=1)

    typer.echo(f"Starting hep-viz server for data at: {path}")
    
    # Pass the data path to the server process via an environment variable.
    # The server module will read this variable during startup.
    os.environ["HEP_VIZ_DATA_PATH"] = str(path.absolute())

    if browser:
        import webbrowser
        import threading
        import time
        import socket

        def open_browser():
            url = f"http://127.0.0.1:{port}"
            # Poll for server availability before opening browser
            for _ in range(30): # Try for 30 seconds
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except (OSError, ConnectionRefusedError):
                    time.sleep(0.5)
            else:
                typer.echo("Server did not start in time, not opening browser.")
                return

            typer.echo(f"Server started. Opening browser at {url}")
            webbrowser.open(url)

        # Run browser opener in a background thread so it doesn't block server startup
        threading.Thread(target=open_browser, daemon=True).start()

    # Start the Uvicorn server
    uvicorn.run("hep_viz.server:app", host="127.0.0.1", port=port, reload=False)

@app.command()
def version():
    """
    Show the current version of hep-viz.
    """
    typer.echo("hep-viz version 0.1.3")

if __name__ == "__main__":
    app()
