"""
Application updates and changelog system
"""
from typing import List, Dict
from datetime import datetime

# Version tracking
CURRENT_VERSION = "2.1.0"

# Updates history (most recent first)
UPDATES = [
    {
        "version": "2.1.0",
        "date": "2025-01-27",
        "title": "Loading States & Improved UX",
        "type": "feature",  # feature, fix, improvement
        "description": "Přidány loading indikátory do všech sekcí aplikace",
        "details": [
            "Univerzální loading komponenty pro všechny sekce",
            "Progress indicator pro ML training (30-60 sekund)",
            "Loading states pro Prediction History, Score History, Watchlist Sentiment",
            "Vylepšená UX s vizuálními indikátory načítání"
        ],
        "icon": "⚡"
    },
    # Add new updates here (most recent first)
]

def get_latest_updates(limit: int = 5) -> List[Dict]:
    """Get latest N updates"""
    return UPDATES[:limit]

def get_current_version() -> str:
    """Get current application version"""
    return CURRENT_VERSION

def get_updates_since_version(version: str) -> List[Dict]:
    """Get all updates since a specific version"""
    try:
        version_index = next(i for i, u in enumerate(UPDATES) if u["version"] == version)
        return UPDATES[:version_index]
    except StopIteration:
        return UPDATES

