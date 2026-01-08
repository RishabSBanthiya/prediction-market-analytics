"""
Web app configuration.
"""

import os
from pathlib import Path


class WebAppConfig:
    """Configuration for web application"""
    
    def __init__(self):
        # Server settings
        self.host = os.getenv("WEBAPP_HOST", "0.0.0.0")
        self.port = int(os.getenv("WEBAPP_PORT", "8000"))
        self.debug = os.getenv("WEBAPP_DEBUG", "false").lower() == "true"
        
        # Database path
        db_path = os.getenv("RISK_DB_PATH", "data/risk_state.db")
        self.db_path = Path(db_path).resolve()
        
        # Static and template paths
        self.static_dir = Path(__file__).parent / "static"
        self.template_dir = Path(__file__).parent / "templates"
        
        # CORS settings
        self.cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")


def get_config() -> WebAppConfig:
    """Get web app configuration"""
    return WebAppConfig()



