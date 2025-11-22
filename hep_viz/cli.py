import typer
import uvicorn
import os
from typing import Optional
from pathlib import Path

app = typer.Typer()

@app.command()
def view(
    path: Path = typer.Argument(..., help="Path to the data file or directory"),
    port: int = typer.Option(8000, help="Port to run the server on"),
    browser: bool = typer.Option(True, help="Automatically open the browser"),
):
    """
    Visualize HEP data from a local file or directory.
    """
    if not path.exists():
        typer.echo(f"Error: Path '{path}' does not exist.")
        raise typer.Exit(code=1)

    typer.echo(f"Starting hep-viz server for data at: {path}")
    
    # We will pass the data path to the server via an environment variable or a global config
    # For now, let's set an env var
    os.environ["HEP_VIZ_DATA_PATH"] = str(path.absolute())

    if browser:
        import webbrowser
        import threading
        import time
        import socket

        def open_browser():
            url = f"http://127.0.0.1:{port}"
            # Poll for server availability
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

        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("hep_viz.server:app", host="127.0.0.1", port=port, reload=False)

@app.command()
def version():
    """
    Show the version of hep-viz.
    """
    typer.echo("hep-viz version 0.1.0")

if __name__ == "__main__":
    app()
