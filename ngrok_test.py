import subprocess
import os
import logging
import time
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_serveo_tunnel(port, subdomain=None):
    """
    Create a Serveo tunnel for a specified local port.
    
    Args:
        port (int): Local port to expose
        subdomain (str, optional): Custom subdomain for the tunnel
    
    Returns:
        str: Serveo tunnel URL or None if tunnel creation fails
    """
    try:
        # Construct SSH command with optional subdomain
        subdomain_arg = f"{subdomain}:" if subdomain else ""
        serveo_command = f"ssh -R {subdomain_arg}80:localhost:{port} serveo.net"
        
        logger.info(f"Initiating Serveo tunnel on port {port}")
        
        # Use a pipe to capture real-time output
        process = subprocess.Popen(
            serveo_command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        # Wait and capture output
        start_time = time.time()
        serveo_url = None
        
        while time.time() - start_time < 30:  # 30-second timeout
            # Read output lines
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            # Extract Serveo URL using regex
            url_match = re.search(r'https?://([^\s]+\.serveo\.net)', line)
            if url_match:
                serveo_url = url_match.group(0)
                logger.info(f"Serveo URL found: {serveo_url}")
                break
            
            time.sleep(0.5)
        
        if not serveo_url:
            logger.error("Failed to obtain Serveo URL")
            return None
        
        return serveo_url
    
    except Exception as e:
        logger.error(f"Tunnel creation error: {e}")
        return None

def main():
    try:
        # Example usage with port 9090
        port = 9090
        
        # Optional: specify a custom subdomain
        # subdomain = "myapp"
        # serveo_url = create_serveo_tunnel(port, subdomain)
        
        serveo_url = create_serveo_tunnel(port)
        
        if serveo_url:
            print(f"Serveo Tunnel URL: {serveo_url}")
        else:
            print("Failed to create Serveo tunnel")
    
    except KeyboardInterrupt:
        print("\nTunnel creation interrupted")

if __name__ == "__main__":
    main()