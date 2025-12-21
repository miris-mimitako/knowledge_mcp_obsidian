"""
ChromaDBベクトルストア管理モジュール
"""
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError("chromadbがインストールされていません。pip install chromadb を実行してください。")


class VectorStore:
    """ChromaDBベクトルストア管理クラス"""
    
    def __init__(
        self,
        collection_name: str = "obsidian_knowledge_base",
        persist_directory: str = "./chroma_db"
    ):
        """
        ベクトルストアを初期化
        
        Args:
            collection_name: コレクション名
            persist_directory: 永続化ディレクトリのパス
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # 永続化ディレクトリを作成
        os.makedirs(persist_directory, exist_ok=True)
        
        # ChromaDBクライアントを作成
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # コレクションを取得または作成
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Obsidian knowledge base embeddings"}
        )
    
    def _generate_id(self, file_path: str, chunk_index: int) -> str:
        """
        チャンクIDを生成
        
        Args:
            file_path: ファイルパス
            chunk_index: チャンクインデックス
        
        Returns:
            一意のID
        """
        # ファイルパスをハッシュ化してIDを生成
        path_hash = hashlib.md5(file_path.encode('utf-8')).hexdigest()
        return f"{path_hash}_{chunk_index}"
    
    def _extract_directory(self, file_path: str) -> str:
        """
        ファイルパスからディレクトリ部分を抽出
        
        Args:
            file_path: ファイルパス
        
        Returns:
            ディレクトリパス
        """
        return str(Path(file_path).parent)
    
    def _extract_filename(self, file_path: str) -> str:
        """
        ファイルパスからファイル名を抽出
        
        Args:
            file_path: ファイルパス
        
        Returns:
            ファイル名
        """
        return Path(file_path).name
    
    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        provider: str = "openrouter/qwen"
    ):
        """
        チャンクとベクトルをChromaDBに追加
        
        Args:
            chunks: チャンク情報のリスト（各チャンクにtext, file_path, location_info, chunk_index, total_chunksを含む）
            embeddings: ベクトルのリスト
            provider: 使用したEmbeddingプロバイダー
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"チャンク数({len(chunks)})とベクトル数({len(embeddings)})が一致しません")
        
        ids = []
        documents = []
        metadatas = []
        embedding_list = []
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = self._generate_id(chunk["file_path"], chunk["chunk_index"])
            
            # 既存のチャンクを削除（更新のため）
            try:
                self.collection.delete(ids=[chunk_id])
            except Exception:
                pass  # 存在しない場合は無視
            
            ids.append(chunk_id)
            documents.append(chunk["text"])
            embedding_list.append(embedding)
            
            # メタデータを構築
            metadata = {
                "source": "obsidian",
                "file_path": chunk["file_path"],
                "file_name": self._extract_filename(chunk["file_path"]),
                "directory": self._extract_directory(chunk["file_path"]),
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "location_info": chunk.get("location_info", ""),
                "created_at": datetime.now().isoformat(),
                "provider": provider
            }
            # file_modified_timeをメタデータに追加（存在する場合）
            if "file_modified_time" in chunk:
                metadata["file_modified_time"] = chunk["file_modified_time"]
            metadatas.append(metadata)
        
        # バッチで追加
        self.collection.add(
            ids=ids,
            embeddings=embedding_list,
            documents=documents,
            metadatas=metadatas
        )
    
    def delete_by_file_path(self, file_path: str):
        """
        指定されたファイルパスのすべてのチャンクを削除
        
        Args:
            file_path: 削除するファイルのパス
        """
        # ファイルパスに基づいてチャンクを検索
        results = self.collection.get(
            where={"file_path": file_path}
        )
        
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
    
    def delete_by_directory(self, directory_path: str):
        """
        指定されたディレクトリパスのすべてのチャンクを削除
        
        Args:
            directory_path: 削除するディレクトリのパス
        """
        # ディレクトリパスに基づいてチャンクを検索
        # ChromaDBのwhere句では前方一致が直接サポートされていないため、
        # すべてのチャンクを取得してフィルタリング
        results = self.collection.get()
        
        # ディレクトリパスでフィルタリング
        directory_path_normalized = directory_path.replace('\\', '/')
        if not directory_path_normalized.endswith('/'):
            directory_path_normalized += '/'
        
        ids_to_delete = []
        for idx, metadata in enumerate(results["metadatas"]):
            file_path = metadata.get("file_path", "")
            if file_path.startswith(directory_path_normalized) or file_path == directory_path:
                ids_to_delete.append(results["ids"][idx])
        
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        directory_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        ベクトル検索を実行
        
        Args:
            query_embedding: クエリベクトル
            n_results: 返却する結果の最大数
            directory_filter: ディレクトリでフィルタリング（オプション）
        
        Returns:
            検索結果のリスト
        """
        where_clause = None
        if directory_filter:
            # ディレクトリフィルタリング（簡易実装）
            # 注意: ChromaDBのwhere句では前方一致が直接サポートされていないため、
            # すべての結果を取得してフィルタリングする必要がある場合がある
            where_clause = {"directory": directory_filter}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause
        )
        
        # 結果を整形
        search_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                search_results.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
        
        return search_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        コレクションの統計情報を取得
        
        Returns:
            統計情報（ドキュメント数など）
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_directory": self.persist_directory
        }
    
    def clear_collection(self):
        """コレクション内のすべてのデータを削除"""
        self.client.delete_collection(name=self.collection_name)
        # コレクションを再作成
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Obsidian knowledge base embeddings"}
        )

