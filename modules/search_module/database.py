"""
データベース管理モジュール
SQLite (FTS5) を使用した全文検索データベースの管理
"""
import sqlite3
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from .job_queue import JobQueueManager


class SearchDatabase:
    """全文検索データベース管理クラス"""
    
    def __init__(self, db_path: str = "search_index.db"):
        """
        データベースを初期化
        
        Args:
            db_path: SQLiteデータベースファイルのパス
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.job_queue: Optional[JobQueueManager] = None
        self._init_database()
    
    def _init_database(self):
        """データベースとテーブルを初期化"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        cursor = self.conn.cursor()
        
        # documentsテーブル（メタデータ管理）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                file_type TEXT,
                location_info TEXT,
                file_modified_time REAL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # マイグレーション: 既存のテーブルにfile_modified_timeカラムを追加（存在しない場合）
        try:
            cursor.execute("""
                ALTER TABLE documents 
                ADD COLUMN file_modified_time REAL
            """)
            self.conn.commit()
        except sqlite3.OperationalError:
            # カラムが既に存在する場合はエラーを無視
            pass
        
        # documents_ftsテーブル（FTS5仮想テーブル）
        # rowidでdocumentsテーブルのidと紐付け
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                content,
                content_rowid='id'
            )
        """)
        
        # データ整合性を保つためのトリガー（documents削除時にFTSも削除）
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ad 
            AFTER DELETE ON documents 
            BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, content) 
                VALUES('delete', old.id, old.content);
            END
        """)
        
        self.conn.commit()
        
        # ジョブキュー管理を初期化
        self.job_queue = JobQueueManager(self.conn)
    
    def add_document(
        self, 
        file_path: str, 
        file_type: str, 
        location_info: str, 
        content: str,
        file_modified_time: Optional[float] = None
    ) -> int:
        """
        ドキュメントをデータベースに追加
        
        Args:
            file_path: ファイルのフルパス
            file_type: ファイルタイプ（pdf, docx, xlsx, txt など）
            location_info: 位置情報（ページ番号、シート名など）
            content: 分かち書き済みのテキスト内容
            file_modified_time: ファイルの最終更新日時（Unix timestamp、秒単位）
        
        Returns:
            追加されたドキュメントのID
        """
        cursor = self.conn.cursor()
        
        try:
            # contentが空の場合は空文字列に設定
            if content is None:
                content = ""
            
            # documentsテーブルにメタデータを挿入
            cursor.execute("""
                INSERT INTO documents (file_path, file_type, location_info, file_modified_time, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (file_path, file_type, location_info, file_modified_time, datetime.now().isoformat()))
            
            doc_id = cursor.lastrowid
            
            # documents_ftsテーブルに検索用テキストを挿入
            # contentが空文字列でも挿入可能にする
            cursor.execute("""
                INSERT INTO documents_fts (rowid, content)
                VALUES (?, ?)
            """, (doc_id, content))
            
            self.conn.commit()
            return doc_id
        except sqlite3.Error as e:
            self.conn.rollback()
            error_msg = f"SQLエラー (add_document): {str(e)}, file_path: {file_path[:100]}..., location_info: {location_info}"
            raise sqlite3.Error(error_msg) from e
    
    def update_document(
        self,
        file_path: str,
        file_type: str,
        location_info: str,
        content: str,
        file_modified_time: Optional[float] = None
    ) -> Optional[int]:
        """
        既存のドキュメントを更新（ファイルパスと位置情報で検索）
        ファイルの編集日時が変更されていない場合はスキップ
        
        Args:
            file_path: ファイルのフルパス
            file_type: ファイルタイプ
            location_info: 位置情報
            content: 分かち書き済みのテキスト内容
            file_modified_time: ファイルの最終更新日時（Unix timestamp、秒単位）
        
        Returns:
            更新されたドキュメントのID、スキップされた場合は-1、存在しない場合は新規追加されたID
        """
        cursor = self.conn.cursor()
        
        try:
            # contentが空の場合は空文字列に設定
            if content is None:
                content = ""
            
            # 既存のドキュメントを検索
            cursor.execute("""
                SELECT id, file_modified_time FROM documents
                WHERE file_path = ? AND location_info = ?
            """, (file_path, location_info))
            
            row = cursor.fetchone()
            
            if row:
                doc_id = row['id']
                existing_modified_time = row['file_modified_time']
                
                # ファイルの編集日時を比較
                # 同じ場合（または両方Noneの場合）はスキップ
                if file_modified_time is not None and existing_modified_time is not None:
                    # 浮動小数点数の比較（1秒以内の誤差は許容）
                    if abs(file_modified_time - existing_modified_time) < 1.0:
                        # 編集日時が同じ場合はスキップ
                        return -1
                
                # 編集日時が異なる、またはファイルの編集日時が取得できない場合は更新
                # メタデータを更新
                cursor.execute("""
                    UPDATE documents
                    SET file_type = ?, file_modified_time = ?, updated_at = ?
                    WHERE id = ?
                """, (file_type, file_modified_time, datetime.now().isoformat(), doc_id))
                
                # FTSテーブルを更新（削除して再挿入）
                # FTS5の削除構文を使用して既存のエントリを削除
                try:
                    cursor.execute("""
                        INSERT INTO documents_fts(documents_fts, rowid, content)
                        VALUES('delete', ?, '')
                    """, (doc_id,))
                except sqlite3.Error as fts_error:
                    # FTS削除でエラーが発生した場合も続行（既に存在しない可能性）
                    print(f"FTS削除エラー（無視）: {str(fts_error)}")
                
                # 新しいコンテンツを挿入
                cursor.execute("""
                    INSERT INTO documents_fts (rowid, content)
                    VALUES (?, ?)
                """, (doc_id, content))
                
                self.conn.commit()
                return doc_id
            else:
                # 存在しない場合は新規追加
                return self.add_document(file_path, file_type, location_info, content, file_modified_time)
        except sqlite3.Error as e:
            self.conn.rollback()
            error_msg = f"SQLエラー (update_document): {str(e)}, file_path: {file_path[:100]}..., location_info: {location_info}"
            raise sqlite3.Error(error_msg) from e
    
    def delete_documents_by_path(self, file_path: str):
        """
        指定されたファイルパスのすべてのドキュメントを削除
        
        Args:
            file_path: 削除するファイルのパス
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM documents WHERE file_path = ?", (file_path,))
        self.conn.commit()
    
    def search(
        self, 
        query: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        全文検索を実行
        
        Args:
            query: 検索クエリ（FTS5構文をサポート）
            limit: 返却する結果の最大数
        
        Returns:
            検索結果のリスト（file_path, location_info, snippetを含む）
        """
        cursor = self.conn.cursor()
        
        # FTS5のMATCH構文を使用して検索
        # snippet関数で検索結果の前後テキストを取得
        cursor.execute("""
            SELECT 
                d.file_path,
                d.file_type,
                d.location_info,
                snippet(documents_fts, 0, '【', '】', '...', 20) as snippet,
                rank
            FROM 
                documents d
            JOIN 
                documents_fts f ON d.id = f.rowid
            WHERE 
                f.content MATCH ?
            ORDER BY 
                rank
            LIMIT ?
        """, (query, limit))
        
        results = []
        for row in cursor.fetchall():
            # doc_idを生成
            doc_id = f"{row['file_path']}|{row['location_info'] or ''}"
            results.append({
                "doc_id": doc_id,
                "file_path": row['file_path'],
                "file_type": row['file_type'],
                "location_info": row['location_info'],
                "snippet": row['snippet'] or "",
                "rank": row['rank']
            })
        
        return results
    
    def search_by_keywords(
        self,
        keywords: List[str],
        limit_per_keyword: int = 10,
        max_total: int = 50
    ) -> List[Dict[str, Any]]:
        """
        各キーワードごとに検索を実行し、結果をマージ
        
        Args:
            keywords: 検索キーワードのリスト
            limit_per_keyword: 各キーワードあたりの取得件数
            max_total: マージ後の最大取得件数
        
        Returns:
            マージされた検索結果のリスト（重複除去、スコア順）
        """
        if not keywords:
            return []
        
        # 各キーワードごとに検索を実行
        all_results = []
        seen_doc_ids = set()
        
        for keyword in keywords:
            keyword_results = self.search(query=keyword, limit=limit_per_keyword)
            
            for result in keyword_results:
                doc_id = result.get('doc_id', f"{result['file_path']}|{result['location_info'] or ''}")
                
                # 重複チェック（同じdoc_idが既にある場合はスキップ）
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    all_results.append(result)
        
        # スコア（rankの逆数）でソート
        # rankが小さいほどスコアが高い
        all_results.sort(key=lambda x: x.get('rank', 999999))
        
        return all_results[:max_total]
    
    def get_document_count(self) -> int:
        """データベース内のドキュメント数を取得"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM documents")
        row = cursor.fetchone()
        return row['count'] if row else 0
    
    def get_documents_by_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        指定されたディレクトリパスに含まれるドキュメントを取得
        
        Args:
            directory_path: 対象ディレクトリのパス（前方一致で検索）
        
        Returns:
            ドキュメント情報のリスト（id, file_path, file_type, location_info, content, updated_at）
        """
        cursor = self.conn.cursor()
        
        # パスを正規化（絶対パスに変換し、区切り文字を統一）
        try:
            normalized_dir = os.path.abspath(directory_path)
        except Exception:
            normalized_dir = directory_path
        
        # WindowsとUnixの両方に対応したパス正規化
        # バックスラッシュをスラッシュに変換
        search_path_forward = normalized_dir.replace('\\', '/')
        # スラッシュをバックスラッシュに変換（Windows用）
        search_path_backward = normalized_dir.replace('/', '\\')
        
        # 末尾にスラッシュ/バックスラッシュを追加（サブディレクトリの検索用）
        search_patterns = []
        for path_variant in [search_path_forward, search_path_backward, normalized_dir]:
            if not path_variant.endswith('/') and not path_variant.endswith('\\'):
                search_patterns.append(path_variant + '/')
                search_patterns.append(path_variant + '\\')
            search_patterns.append(path_variant)
        
        # 重複を除去
        search_patterns = list(set(search_patterns))
        
        # SQLのLIKEパターンを生成
        like_patterns = [f"{pattern}%" for pattern in search_patterns]
        
        # file_pathが指定ディレクトリ配下にあるドキュメントを取得
        # 複数のパターンで検索（大文字小文字を無視するため、LIKEを複数回使用）
        placeholders = ' OR '.join(['d.file_path LIKE ?' for _ in like_patterns])
        placeholders += ' OR ' + ' OR '.join(['d.file_path = ?' for _ in search_patterns])
        
        query = f"""
            SELECT 
                d.id,
                d.file_path,
                d.file_type,
                d.location_info,
                d.updated_at,
                d.file_modified_time,
                f.content
            FROM 
                documents d
            LEFT JOIN 
                documents_fts f ON d.id = f.rowid
            WHERE 
                {placeholders}
            ORDER BY 
                d.file_path, d.location_info
        """
        
        # パラメータを結合（LIKEパターン + 完全一致パターン）
        params = like_patterns + search_patterns
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row['id'],
                "file_path": row['file_path'],
                "file_type": row['file_type'],
                "location_info": row['location_info'],
                "content": row['content'] or "",
                "updated_at": row['updated_at'],
                "file_modified_time": row['file_modified_time']
            })
        
        return results
    
    def clear_all(self):
        """データベース内のすべてのドキュメントを削除"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM documents")
        self.conn.commit()
    
    def close(self):
        """データベース接続を閉じる"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

