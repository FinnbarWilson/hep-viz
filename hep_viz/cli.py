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

    uvicorn.run("hep_viz.server:app", host="127.0.0.1", port=port, reload=False)

@app.command()
def version():
    """
    Show the version of hep-viz.
    """
    typer.echo("hep-viz version 0.1.0")

if __name__ == "__main__":
    app()
