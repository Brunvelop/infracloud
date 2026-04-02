"""
infracloud — Launch and manage GPU servers on Vast.ai for AI model inference.

Typical usage:

    from infracloud import InfraCloud

    cloud = InfraCloud()
    server = cloud.up("ltx-video")   # blocks until ready
    print(server.url)
    server.down()

Or via CLI:

    infracloud up ltx-video
    infracloud url
    infracloud down
"""

__version__ = "0.1.0"

from infracloud.stack import Stack
from infracloud.state import save_state, load_state, clear_state

# Populated in later commits:
# from infracloud.cloud import InfraCloud, Server

__all__ = ["Stack", "save_state", "load_state", "clear_state", "__version__"]
