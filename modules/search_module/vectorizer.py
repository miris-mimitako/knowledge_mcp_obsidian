"""
ベクトル化パイプラインモジュール
データベースからドキュメントを取得し、ベクトル化してChromaDBに保存
"""
import os
from typing import List, Dict, Any, Optional, Callable
from .database import SearchDatabase
from .chunker import TextChunker
from .embedding_providers import EmbeddingProvider, EmbeddingProviderType, create_embedding_provider
from .vector_store import VectorStore


def get_file_modified_time(file_path: str) -> Optional[float]:
    """
    ファイルの最終更新日時を取得（Unix timestamp）
    
    Args:
        file_path: ファイルのパス
    
    Returns:
        ファイルの最終更新日時（Unix timestamp、秒単位）、取得できない場合はNone
    """
    try:
        if os.path.exists(file_path):
            # os.path.getmtime() は秒単位のUnix timestampを返す
            return os.path.getmtime(file_path)
        return None
    except Exception:
        return None


class DocumentVectorizer:
    """ドキュメントをベクトル化するクラス"""
    
    def __init__(
        self,
        db: SearchDatabase,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        chunker: TextChunker
    ):
        """
        ベクトル化パイプラインを初期化
        
        Args:
            db: データベースインスタンス
            vector_store: ベクトルストアインスタンス
            embedding_provider: Embeddingプロバイダー
            chunker: チャンカー
        """
        self.db = db
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.chunker = chunker
    
    def vectorize_directory(
        self,
        directory_path: str,
        batch_size: int = 10,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        指定されたディレクトリ内のドキュメントをベクトル化
        
        Args:
            directory_path: 対象ディレクトリのパス
            batch_size: バッチ処理サイズ（一度に処理するチャンク数）
            progress_callback: 進捗コールバック関数（current, total, messageを受け取る）
        
        Returns:
            処理結果（処理したファイル数、チャンク数など）
        """
        # データベースからドキュメントを取得
        try:
            documents = self.db.get_documents_by_directory(directory_path)
        except Exception as e:
            raise Exception(f"データベースからのドキュメント取得に失敗しました: {str(e)}") from e
        
        if not documents:
            return {
                "message": "対象ディレクトリにドキュメントが見つかりませんでした（データベースにインデックスが存在しない可能性があります）",
                "processed_files": 0,
                "total_chunks": 0,
                "failed_files": 0,
                "total_files": 0
            }
        
        total_files = len(documents)
        processed_files = 0
        skipped_files = 0
        failed_files = 0
        total_chunks = 0
        
        # ファイルごとに処理
        for doc_idx, doc in enumerate(documents):
            try:
                if progress_callback:
                    progress_callback(
                        current=doc_idx + 1,
                        total=total_files,
                        message=f"処理中: {doc['file_path']}"
                    )
                
                # コンテンツが空の場合はスキップ
                if not doc.get("content") or not doc["content"].strip():
                    continue
                
                # ファイルの更新日時をチェック
                file_path = doc.get("file_path")
                if file_path:
                    current_modified_time = get_file_modified_time(file_path)
                    stored_modified_time = doc.get("file_modified_time")
                    
                    # ファイルの更新日時が変更されていない場合はスキップ
                    if current_modified_time is not None and stored_modified_time is not None:
                        # 浮動小数点数の比較（1秒以内の誤差は許容）
                        if abs(current_modified_time - stored_modified_time) < 1.0:
                            skipped_files += 1
                            continue
                
                # ファイルの更新日時を取得（チャンクのメタデータに含めるため）
                file_path = doc.get("file_path")
                current_modified_time = None
                if file_path:
                    current_modified_time = get_file_modified_time(file_path)
                
                # チャンキング
                chunks = self.chunker.chunk_document(
                    content=doc["content"],
                    file_path=doc["file_path"],
                    location_info=doc.get("location_info")
                )
                
                if not chunks:
                    continue
                
                # 各チャンクにfile_modified_timeを追加
                for chunk in chunks:
                    if current_modified_time is not None:
                        chunk["file_modified_time"] = current_modified_time
                
                # バッチ処理でEmbeddingを取得
                chunk_texts = [chunk["text"] for chunk in chunks]
                embeddings = []
                
                for i in range(0, len(chunk_texts), batch_size):
                    batch_texts = chunk_texts[i:i + batch_size]
                    batch_embeddings = self.embedding_provider.get_embeddings(batch_texts)
                    embeddings.extend(batch_embeddings)
                
                # ChromaDBに保存
                provider_name = f"{self.embedding_provider.__class__.__name__}"
                self.vector_store.add_chunks(
                    chunks=chunks,
                    embeddings=embeddings,
                    provider=provider_name
                )
                
                processed_files += 1
                total_chunks += len(chunks)
                
            except Exception as e:
                failed_files += 1
                error_msg = str(e)
                
                # APIキー関連のエラーの場合は詳細な情報を表示
                if "401" in error_msg or "Unauthorized" in error_msg or "API" in error_msg:
                    print(f"⚠️ 認証エラー: {error_msg}")
                    print(f"   ファイル: {doc['file_path']}")
                    print(f"   ※ このエラーはすべてのファイルで発生する可能性があります。")
                    print(f"   ※ APIキーの設定を確認してから再試行してください。\n")
                else:
                    print(f"ファイルのベクトル化に失敗しました: {doc['file_path']}")
                    print(f"  エラー: {error_msg}")
                continue
        
        return {
            "message": "ベクトル化が完了しました",
            "processed_files": processed_files,
            "skipped_files": skipped_files,
            "failed_files": failed_files,
            "total_files": total_files,
            "total_chunks": total_chunks
        }
    
    def vectorize_file(
        self,
        file_path: str,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        指定されたファイルをベクトル化
        
        Args:
            file_path: 対象ファイルのパス
            batch_size: バッチ処理サイズ
        
        Returns:
            処理結果
        """
        # データベースからファイルのドキュメントを取得
        documents = self.db.get_documents_by_directory(file_path)
        
        if not documents:
            return {
                "message": "ファイルが見つかりませんでした",
                "processed_chunks": 0
            }
        
        total_chunks = 0
        
        for doc in documents:
            if not doc.get("content") or not doc["content"].strip():
                continue
            
            # ファイルの更新日時をチェック
            file_path = doc.get("file_path")
            if file_path:
                current_modified_time = get_file_modified_time(file_path)
                stored_modified_time = doc.get("file_modified_time")
                
                # ファイルの更新日時が変更されていない場合はスキップ
                if current_modified_time is not None and stored_modified_time is not None:
                    # 浮動小数点数の比較（1秒以内の誤差は許容）
                    if abs(current_modified_time - stored_modified_time) < 1.0:
                        continue
            
            # ファイルの更新日時を取得（チャンクのメタデータに含めるため）
            current_modified_time = None
            if file_path:
                current_modified_time = get_file_modified_time(file_path)
            
            # チャンキング
            chunks = self.chunker.chunk_document(
                content=doc["content"],
                file_path=doc["file_path"],
                location_info=doc.get("location_info")
            )
            
            if not chunks:
                continue
            
            # 各チャンクにfile_modified_timeを追加
            for chunk in chunks:
                if current_modified_time is not None:
                    chunk["file_modified_time"] = current_modified_time
            
            # バッチ処理でEmbeddingを取得
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = []
            
            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i:i + batch_size]
                batch_embeddings = self.embedding_provider.get_embeddings(batch_texts)
                embeddings.extend(batch_embeddings)
            
            # ChromaDBに保存
            provider_name = f"{self.embedding_provider.__class__.__name__}"
            self.vector_store.add_chunks(
                chunks=chunks,
                embeddings=embeddings,
                provider=provider_name
            )
            
            total_chunks += len(chunks)
        
        return {
            "message": "ファイルのベクトル化が完了しました",
            "processed_chunks": total_chunks
        }

