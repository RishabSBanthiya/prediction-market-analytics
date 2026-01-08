"""
Alert storage wrapper for web app.
"""

from typing import Optional, List
from datetime import datetime

from polymarket.trading.storage.sqlite import SQLiteStorage
from polymarket.core.config import get_config


class AlertStorage:
    """Wrapper for flow alert storage"""
    
    def __init__(self, db_path: Optional[str] = None):
        config = get_config()
        self.storage = SQLiteStorage(db_path or config.db_path)
    
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
        with self.storage.transaction() as txn:
            alerts = txn.get_alerts(
                start_time=start_time,
                end_time=end_time,
                alert_type=alert_type,
                severity=severity,
                category=category,
                limit=limit
            )
        return alerts
    
    def get_alert(self, alert_id: int) -> Optional[dict]:
        """Get single alert by ID"""
        alerts = self.get_alerts()
        for alert in alerts:
            if alert["id"] == alert_id:
                return alert
        return None
    
    def get_stats(self) -> dict:
        """Get alert statistics"""
        with self.storage.transaction() as txn:
            return txn.get_alert_stats()



