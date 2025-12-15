import os
import threading
import webbrowser
import time
import socket
import numpy as np
import pandas as pd
from .data_processor import DataProcessor
from .server import set_data_processor, run_server, app, set_server_mode
import uvicorn

def view(data, port=8000, host="127.0.0.1", open_browser=None):
    """
    Launch the hep-viz 3D event visualizer.

    Initializes the data processor with the provided data, starts a local
    web server in a background thread, and opens the visualization interface.

    Args:
        data (dict): Event data structure. Expected to match Hugging Face datasets
            structure or a dictionary of lists. Required keys: 'particles',
            'tracks', 'tracker_hits', 'calo_hits'.
        port (int): Port to run the local web server on. Default is 8000.
        host (str): Host to bind the server to. Default is "127.0.0.1".
        open_browser (bool, optional): Whether to automatically open the browser.
            If None, detection is attempted based on SSH session status.
    """
    # Auto-detect SSH session
    is_ssh = "SSH_CONNECTION" in os.environ
    
    if open_browser is None:
        open_browser = not is_ssh

    print("Initializing hep-viz...")
    
    # Initialize the DataProcessor with the provided in-memory data
    processor = DataProcessor(data)
    set_data_processor(processor)
    set_server_mode("default")
    
    # Start the FastAPI server in a separate thread.
    server_thread = threading.Thread(
        target=run_server, 
        kwargs={"host": host, "port": port}, 
        daemon=False
    )
    server_thread.start()
    
    print(f"Starting hep-viz server at http://{host}:{port}...")
    
    # Helper function to poll the server until it is ready
    def wait_for_server():
        for _ in range(60): # Wait up to 30 seconds
            try:
                with socket.create_connection((host, port), timeout=1):
                    return True
            except (OSError, ConnectionRefusedError):
                time.sleep(0.5)
        return False

    if wait_for_server():
        url = f"http://{host}:{port}"
        
        if open_browser:
            print(f"Server started! Opening {url}")
            webbrowser.open(url)
        else:
            print(f"\n✅ Server running at {url}")
            if is_ssh:
                print("\n⚠️  SSH session detected. The browser cannot open automatically.")
                print(f"To view this on your local machine, forward the port by running this in a NEW local terminal:")
                print(f"\n    ssh -L {port}:localhost:{port} <your-user>@<remote-host>\n")
                print(f"Then open this URL in your local browser:\n")
                print(f"    http://localhost:{port}\n")
    else:
        print("Error: Server failed to start within timeout.")

def view_graph(hits, edges=None, scores=None, truth=None, masks=None, port=8001, open_browser=True):
    """
    Visualizes event data as a graph.

    Args:
        hits: DataFrame or dict containing hit data (x, y, z).
        edges: (Optional) 2xN array of edge indices.
        scores: (Optional) Array of scores for each edge.
        truth: (Optional) Array of truth labels for each edge.
        masks: (Optional) MxN matrix (Flows x Hits). If provided, edges be generated from flows.
        port: Port to run the server on.
        open_browser: Whether to open the browser automatically.
    """
    global processor

    # --- HANDLE MASKS (Convert to Edges) ---
    if masks is not None and edges is None:
        print("Converting Masks to Edges...")
        
        # Extract hit coordinates
        if isinstance(hits, pd.DataFrame):
            h_x, h_y, h_z = hits['x'].values, hits['y'].values, hits['z'].values
        else:
            h_x, h_y, h_z = hits['x'], hits['y'], hits['z']

        # Calculate R for sorting (Inner -> Outer)
        h_r = np.sqrt(h_x**2 + h_y**2 + h_z**2)

        all_edges_list = []
        all_scores_list = []

        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        
        num_flows, num_hits = masks.shape
        
        for i in range(num_flows):
            flow = masks[i]
            
            # Select indices (assuming binary-like behavior > 0.0)
            hit_indices = np.where(flow > 0.0)[0]
            
            if len(hit_indices) < 2:
                continue

            # Sort by Radius (r) to ensure sequential connections
            r_vals = h_r[hit_indices]
            sorted_order = np.argsort(r_vals)
            sorted_indices = hit_indices[sorted_order]

            # Create sequential edges: (0->1), (1->2), etc.
            sources = sorted_indices[:-1]
            targets = sorted_indices[1:]
            
            flow_edges = np.stack([sources, targets], axis=0)
            all_edges_list.append(flow_edges)
            
            # Default scores to 1.0 for binary masks
            flow_scores = np.ones(len(sources), dtype=np.float32)
            all_scores_list.append(flow_scores)

        if all_edges_list:
            edges = np.concatenate(all_edges_list, axis=1)
            scores = np.concatenate(all_scores_list)
        else:
            edges = np.zeros((2, 0), dtype=int)
            scores = np.zeros(0)
            
        print(f"Generated {edges.shape[1]} edges from {num_flows} flows.")

    # Standard data wrapping for DataProcessor
    data = {'hits': hits}
    
    if edges is not None: data['edges'] = edges
    if scores is not None: data['scores'] = scores
    if truth is not None: data['truth'] = truth
    
    # Ensure edges exist
    if 'edges' not in data:
        data['edges'] = np.zeros((2, 0), dtype=int)
        data['scores'] = np.zeros(0)

    print("Initializing hep-viz in Graph Mode...")
        
    processor = DataProcessor(data)
    set_data_processor(processor)
    set_server_mode("graph")
    
    start_server(port, open_browser)

def start_server(port, open_browser):
    """Helper to start the server and browser thread."""
    url = f"http://127.0.0.1:{port}"
    print(f"Starting Graph Mode Server at {url}")
    
    def launch_browser():
        if open_browser:
            time.sleep(1)
            webbrowser.open(url)
        
    threading.Thread(target=launch_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
