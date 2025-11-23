import threading
import webbrowser
import time
import socket
from .data_processor import DataProcessor
from .server import set_processor, run_server

def view(data, port=8000, host="127.0.0.1"):
    """
    Launch the hep-viz 3D event visualizer in the default web browser.
    
    This function initializes the data processor with the provided data,
    starts a local web server in a background thread, and opens the
    visualization interface.

    Args:
        data (dict): A dictionary containing the event data. It is expected to match
                     the structure of Hugging Face datasets or a dictionary of lists.
                     Required keys: 'particles', 'tracks', 'tracker_hits', 'calo_hits'.
        port (int): The port to run the local web server on. Default is 8000.
        host (str): The host to bind the server to. Default is "127.0.0.1".
    """
    print("Initializing hep-viz...")
    
    # Initialize the DataProcessor with the provided in-memory data
    processor = DataProcessor(data)
    set_processor(processor)
    
    # Start the FastAPI server in a separate thread.
    # daemon=False ensures the script keeps running while the server is active,
    # preventing the program from exiting immediately after this call.
    server_thread = threading.Thread(target=run_server, kwargs={"host": host, "port": port}, daemon=False)
    server_thread.start()
    
    print(f"Starting hep-viz server at http://{host}:{port}...")
    
    # Helper function to poll the server until it is ready
    def wait_for_server():
        for _ in range(60): # Wait up to 30 seconds (60 * 0.5s)
            try:
                with socket.create_connection((host, port), timeout=1):
                    return True
            except (OSError, ConnectionRefusedError):
                time.sleep(0.5)
        return False

    if wait_for_server():
        url = f"http://{host}:{port}"
        print(f"Server started! Opening {url}")
        webbrowser.open(url)
    else:
        print("Error: Server failed to start within timeout.")
