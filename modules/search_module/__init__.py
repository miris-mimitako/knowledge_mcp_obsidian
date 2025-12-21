"""
Search Module
全文検索エンジンを提供するモジュール
SQLite (FTS5) を使用した日本語対応の全文検索機能
"""
from .router import router, task_router

__all__ = ["router", "task_router"]

