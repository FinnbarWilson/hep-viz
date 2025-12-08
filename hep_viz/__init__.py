import os
import threading
import webbrowser
import time
import socket
from .data_processor import DataProcessor
from .server import set_processor, run_server

def view(data, port=8000, host="127.0.0.1", open_browser=None):
    """
    Launch the hep-viz 3D event visualizer.
    
    This function initializes the data processor with the provided data,
    starts a local web server in a background thread, and opens the
    visualization interface.

    Args:
        data (dict): A dictionary containing the event data. It is expected to match
                     the structure of Hugging Face datasets or a dictionary of lists.
                     Required keys: 'particles', 'tracks', 'tracker_hits', 'calo_hits'.
        port (int): The port to run the local web server on. Default is 8000.
        host (str): The host to bind the server to. Default is "127.0.0.1".
        open_browser (bool, optional): Whether to automatically open the browser. 
                                       If None, it detects if running in an SSH session.
    """
    # Auto-detect SSH session
    is_ssh = "SSH_CONNECTION" in os.environ
    
    if open_browser is None:
        open_browser = not is_ssh

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
