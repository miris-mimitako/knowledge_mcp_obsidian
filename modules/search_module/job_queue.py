"""
ジョブキュー管理モジュール
汎用的なキュー管理システム（インデックス作成、エクスポート、バックアップなど）
"""
import sqlite3
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """ジョブステータス"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """ジョブタイプ"""
    INDEX = "index"
    VECTORIZE = "vectorize"
    EXPORT = "export"
    BACKUP = "backup"
    # 将来的に他のタイプを追加可能


class JobQueueManager:
    """ジョブキュー管理クラス"""
    
    def __init__(self, db_conn: sqlite3.Connection):
        """
        ジョブキュー管理を初期化
        
        Args:
            db_conn: SQLiteデータベース接続
        """
        self.conn = db_conn
        self._init_table()
    
    def _init_table(self):
        """ジョブキューテーブルを初期化"""
        cursor = self.conn.cursor()
        
        # job_queueテーブル（汎用的なキュー管理）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                parameters TEXT,  -- JSON形式でジョブのパラメータを保存
                progress TEXT,    -- JSON形式で進捗情報を保存 {"current": 10, "total": 100, "message": "処理中..."}
                result TEXT,      -- JSON形式で結果を保存
                error_message TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                started_at DATETIME,
                completed_at DATETIME,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # インデックスを作成（ステータスとジョブタイプで検索を高速化）
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_job_queue_status 
            ON job_queue(status)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_job_queue_type 
            ON job_queue(job_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_job_queue_created 
            ON job_queue(created_at DESC)
        """)
        
        self.conn.commit()
    
    def create_job(
        self,
        job_type: str,
        parameters: Dict[str, Any],
        initial_progress: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        新しいジョブを作成
        
        Args:
            job_type: ジョブタイプ（'index', 'export', 'backup'など）
            parameters: ジョブのパラメータ（辞書形式）
            initial_progress: 初期進捗情報（辞書形式）
        
        Returns:
            作成されたジョブのID
        """
        cursor = self.conn.cursor()
        
        now = datetime.now().isoformat()
        progress = initial_progress or {"current": 0, "total": 0, "message": "待機中"}
        
        cursor.execute("""
            INSERT INTO job_queue (
                job_type, status, parameters, progress, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            job_type,
            JobStatus.PENDING.value,
            json.dumps(parameters, ensure_ascii=False),
            json.dumps(progress, ensure_ascii=False),
            now,
            now
        ))
        
        job_id = cursor.lastrowid
        self.conn.commit()
        return job_id
    
    def update_job_status(
        self,
        job_id: int,
        status: JobStatus,
        error_message: Optional[str] = None
    ):
        """
        ジョブのステータスを更新
        
        Args:
            job_id: ジョブID
            status: 新しいステータス
            error_message: エラーメッセージ（失敗時のみ）
        """
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        
        updates = ["status = ?", "updated_at = ?"]
        values = [status.value, now]
        
        if status == JobStatus.PROCESSING:
            updates.append("started_at = COALESCE(started_at, ?)")
            values.append(now)
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            updates.append("completed_at = ?")
            values.append(now)
        
        if error_message:
            updates.append("error_message = ?")
            values.append(error_message)
        
        values.append(job_id)
        
        cursor.execute(f"""
            UPDATE job_queue
            SET {', '.join(updates)}
            WHERE id = ?
        """, values)
        
        self.conn.commit()
    
    def update_job_progress(
        self,
        job_id: int,
        current: int,
        total: int,
        message: Optional[str] = None
    ):
        """
        ジョブの進捗を更新
        
        Args:
            job_id: ジョブID
            current: 現在の進捗値
            total: 総数
            message: 進捗メッセージ
        """
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        
        progress = {
            "current": current,
            "total": total,
            "percentage": round((current / total * 100), 2) if total > 0 else 0,
            "message": message or f"処理中: {current}/{total}"
        }
        
        cursor.execute("""
            UPDATE job_queue
            SET progress = ?, updated_at = ?
            WHERE id = ?
        """, (json.dumps(progress, ensure_ascii=False), now, job_id))
        
        self.conn.commit()
    
    def update_job_result(self, job_id: int, result: Dict[str, Any]):
        """
        ジョブの結果を更新
        
        Args:
            job_id: ジョブID
            result: 結果データ（辞書形式）
        """
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute("""
            UPDATE job_queue
            SET result = ?, updated_at = ?
            WHERE id = ?
        """, (json.dumps(result, ensure_ascii=False), now, job_id))
        
        self.conn.commit()
    
    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        """
        ジョブ情報を取得
        
        Args:
            job_id: ジョブID
        
        Returns:
            ジョブ情報（辞書形式）、存在しない場合はNone
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM job_queue WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return self._row_to_dict(row)
    
    def get_jobs(
        self,
        job_type: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        ジョブ一覧を取得
        
        Args:
            job_type: ジョブタイプでフィルタ（Noneの場合はすべて）
            status: ステータスでフィルタ（Noneの場合はすべて）
            limit: 取得件数の上限
        
        Returns:
            ジョブ情報のリスト
        """
        cursor = self.conn.cursor()
        
        conditions = []
        values = []
        
        if job_type:
            conditions.append("job_type = ?")
            values.append(job_type)
        
        if status:
            conditions.append("status = ?")
            values.append(status.value)
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        values.append(limit)
        
        cursor.execute(f"""
            SELECT * FROM job_queue
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """, values)
        
        return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def cancel_job(self, job_id: int) -> bool:
        """
        ジョブをキャンセル
        
        Args:
            job_id: ジョブID
        
        Returns:
            キャンセルに成功した場合はTrue
        """
        cursor = self.conn.cursor()
        
        # 処理中または待機中のジョブのみキャンセル可能
        cursor.execute("""
            UPDATE job_queue
            SET status = ?, completed_at = ?, updated_at = ?
            WHERE id = ? AND status IN ('pending', 'processing')
        """, (JobStatus.CANCELLED.value, datetime.now().isoformat(), datetime.now().isoformat(), job_id))
        
        self.conn.commit()
        return cursor.rowcount > 0
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """SQLiteのRowオブジェクトを辞書に変換"""
        result = dict(row)
        
        # JSON形式のフィールドをパース
        for key in ['parameters', 'progress', 'result']:
            if result[key]:
                try:
                    result[key] = json.loads(result[key])
                except (json.JSONDecodeError, TypeError):
                    result[key] = {}
        
        return result

