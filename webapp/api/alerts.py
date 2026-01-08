"""
Flow alerts API endpoints.
"""

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from typing import Optional, List
from datetime import datetime
import json
import asyncio

from ..models.schemas import AlertResponse, AlertStats
from ..services.alert_service import AlertService

router = APIRouter()
service = AlertService()


@router.get("/", response_model=List[AlertResponse])
async def list_alerts(
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    alert_type: Optional[str] = Query(None, description="Filter by alert type"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: Optional[int] = Query(100, description="Limit results")
):
    """List all alerts with optional filters"""
    alerts = service.get_alerts(
        start_time=start_time,
        end_time=end_time,
        alert_type=alert_type,
        severity=severity,
        category=category,
        limit=limit
    )
    return alerts


@router.get("/recent")
async def get_recent_alerts(
    hours: int = Query(24, description="Hours to look back"),
    limit: int = Query(100, description="Limit results")
):
    """Get recent alerts"""
    return service.get_recent_alerts(hours=hours, limit=limit)


@router.get("/stats", response_model=AlertStats)
async def get_alert_stats():
    """Get alert statistics"""
    return service.get_stats()


@router.websocket("/realtime")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alerts"""
    await websocket.accept()
    
    last_alert_id = 0
    
    try:
        while True:
            # Get new alerts since last check
            alerts = service.get_alerts(limit=100)
            new_alerts = [a for a in alerts if a["id"] > last_alert_id]
            
            if new_alerts:
                # Update last alert ID
                last_alert_id = max(a["id"] for a in new_alerts)
                
                # Send new alerts
                for alert in new_alerts:
                    await websocket.send_json({
                        "type": "alert",
                        "data": alert
                    })
            
            # Wait before next check
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        pass



