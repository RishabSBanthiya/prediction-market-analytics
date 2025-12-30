"""
FastAPI web application entry point.
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
import uvicorn

from .config import get_config
from .api import trades, positions, alerts, performance, risk

app = FastAPI(title="Polymarket Analytics", version="1.0.0")

config = get_config()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(config.static_dir)), name="static")

# Templates
templates = Jinja2Templates(directory=str(config.template_dir))

# Include routers
app.include_router(trades.router, prefix="/api/trades", tags=["trades"])
app.include_router(positions.router, prefix="/api/positions", tags=["positions"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])
app.include_router(performance.router, prefix="/api/performance", tags=["performance"])
app.include_router(risk.router, prefix="/api/risk", tags=["risk"])


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/trades", response_class=HTMLResponse)
async def trades_page(request: Request):
    """Trade history page"""
    return templates.TemplateResponse("trades.html", {"request": request})


@app.get("/alerts", response_class=HTMLResponse)
async def alerts_page(request: Request):
    """Unusual trades page"""
    return templates.TemplateResponse("alerts.html", {"request": request})


@app.get("/risk", response_class=HTMLResponse)
async def risk_page(request: Request):
    """Risk monitoring page"""
    return templates.TemplateResponse("risk.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run(
        "webapp.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug
    )

