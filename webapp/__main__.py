"""
Allow running webapp as a module: python -m webapp
"""

import uvicorn
from .config import get_config

if __name__ == "__main__":
    config = get_config()
    uvicorn.run(
        "webapp.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug
    )



