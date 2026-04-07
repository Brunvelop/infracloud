"""
infracloud — Launch and manage GPU servers on Vast.ai for AI model inference.

Typical usage:

    from infracloud import InfraCloud

    cloud = InfraCloud()
    server = cloud.up("ltx-2.3-fp8-distilled")   # blocks until ready
    print(server.url)
    server.down()

Or via CLI:

    infracloud up ltx-2.3-fp8-distilled
    infracloud url
    infracloud down
"""

__version__ = "0.1.0"

from infracloud.stack import Stack
from infracloud.cloud import InfraCloud, Server

__all__ = ["Stack", "InfraCloud", "Server", "__version__"]
