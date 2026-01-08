"""
Alert service for managing flow alerts.
"""

from typing import Optional, List, Dict
from datetime import datetime, timedelta

from ..storage.alert_storage import AlertStorage


class AlertService:
    """Service for alert operations"""
    
    def __init__(self, storage: Optional[AlertStorage] = None):
        self.storage = storage or AlertStorage()
    
    def get_alerts(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[dict]:
        """Get alerts with filters"""
        return self.storage.get_alerts(
            start_time=start_time,
            end_time=end_time,
            alert_type=alert_type,
            severity=severity,
            category=category,
            limit=limit
        )
    
    def get_recent_alerts(self, hours: int = 24, limit: int = 100) -> List[dict]:
        """Get recent alerts"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        return self.get_alerts(
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
    def get_stats(self) -> Dict:
        """Get alert statistics"""
        return self.storage.get_stats()



