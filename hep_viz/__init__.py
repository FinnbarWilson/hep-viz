import threading
import webbrowser
import time
import socket
from .data_processor import DataProcessor
from .server import set_processor, run_server

def view(data, port=8000, host="127.0.0.1"):
    """
    View the provided data in a web browser.
    
    Args:
        data (dict): Dictionary containing the data (e.g. from Hugging Face datasets).
                     Expected keys: 'particles', 'tracks', 'tracker_hits', 'calo_hits'.
        port (int): Port to run the server on.
        host (str): Host to run the server on.
    """
    print("Initializing hep-viz...")
    
    # 1. Initialize Processor
    processor = DataProcessor(data)
    set_processor(processor)
    
    # 2. Start Server in Thread
    # daemon=False ensures the script keeps running while the server is active
    server_thread = threading.Thread(target=run_server, kwargs={"host": host, "port": port}, daemon=False)
    server_thread.start()
    
    # 3. Wait for server to start (simple poll)
    print(f"Starting hep-viz server at http://{host}:{port}...")
    
    def wait_for_server():
        for _ in range(60): # 30 seconds timeout
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
