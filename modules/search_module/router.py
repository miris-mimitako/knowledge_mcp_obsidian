"""
Search Module Router
全文検索エンジンのAPIエンドポイントを提供するルーター
"""
import os
import threading
from pathlib import Path
from typing import List, Optional, Callable
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, Response
from pydantic import BaseModel, Field

from .database import SearchDatabase
from .tokenizer import JapaneseTokenizer
from .parsers import get_parser
from .job_queue import JobStatus, JobType
from .vectorizer import DocumentVectorizer
from .chunker import TextChunker
from .vector_store import VectorStore
from .embedding_providers import EmbeddingProviderType, create_embedding_provider
from .query_expansion import QueryExpansion
from .hybrid_search import reciprocal_rank_fusion, normalize_doc_id, HybridSearchResult
from .llm_providers import LLMProviderType, create_llm_provider


router = APIRouter(
    prefix="/search",
    tags=["search"],
    responses={404: {"description": "Not found"}},
)


# リクエスト/レスポンスモデル
class IndexRequest(BaseModel):
    """インデックス作成リクエスト"""
    directory_path: str = Field(..., description="インデックスを作成するディレクトリのパス")
    clear_existing: bool = Field(default=False, description="既存のインデックスをクリアするか")


class WatchedDirectoryRequest(BaseModel):
    """監視対象ディレクトリ追加リクエスト"""
    directory_path: str = Field(..., description="監視対象ディレクトリのパス")
    scan_interval_minutes: int = Field(default=60, ge=1, description="スキャン間隔（分）")
    enabled: bool = Field(default=True, description="有効/無効")


class WatchedDirectoryUpdateRequest(BaseModel):
    """監視対象ディレクトリ更新リクエスト"""
    scan_interval_minutes: Optional[int] = Field(None, ge=1, description="スキャン間隔（分）")
    enabled: Optional[bool] = Field(None, description="有効/無効")


class SearchRequest(BaseModel):
    """検索リクエスト"""
    query: str = Field(..., description="検索キーワード")
    limit: int = Field(default=50, ge=1, le=100, description="返却する結果の最大数")


class HybridSearchRequest(BaseModel):
    """ハイブリッド検索リクエスト"""
    query: str = Field(..., description="検索キーワード")
    limit: int = Field(default=20, ge=1, le=100, description="返却する結果の最大数")
    hybrid_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="ベクトル検索の重み（0.0=全文検索のみ、1.0=ベクトル検索のみ、0.5=等価）")
    keyword_limit: int = Field(default=10, ge=1, le=50, description="各キーワードあたりの全文検索取得件数")
    vector_limit: int = Field(default=20, ge=1, le=100, description="ベクトル検索の取得件数")
    expand_synonyms: bool = Field(default=False, description="類義語展開を使用するかどうか")


class SearchResult(BaseModel):
    """検索結果"""
    file_path: str
    file_type: Optional[str]
    location_info: Optional[str]
    snippet: str


class SearchResponse(BaseModel):
    """検索レスポンス"""
    query: str
    results: List[SearchResult]
    total: int


class IndexResponse(BaseModel):
    """インデックス作成レスポンス"""
    message: str
    job_id: int
    directory_path: str


class VectorizeRequest(BaseModel):
    """ベクトル化リクエスト"""
    directory_path: str = Field(..., description="ベクトル化するディレクトリのパス")
    provider: Optional[str] = Field(default=None, description="Embeddingプロバイダー（openrouter, aws_bedrock, litellm）")
    model: Optional[str] = Field(default=None, description="使用するモデル名またはARN（ARNは推論エンドポイントの場合に使用）")
    api_base: Optional[str] = Field(default=None, description="LiteLLMのカスタムエンドポイントURL（litellmプロバイダーの場合のみ）")
    aws_region_name: Optional[str] = Field(default=None, description="AWSリージョン名（aws_bedrockプロバイダーの場合のみ）")
    aws_access_key_id: Optional[str] = Field(default=None, description="AWSアクセスキーID（aws_bedrockプロバイダーの場合のみ）")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWSシークレットキー（aws_bedrockプロバイダーの場合のみ）")
    chunk_size: int = Field(default=1024, description="チャンクサイズ（トークン数）")
    chunk_overlap: int = Field(default=50, description="オーバーラップサイズ（トークン数）")
    force_revectorize: bool = Field(default=False, description="強制再ベクトル化（更新日時チェックをスキップ）")


class VectorizeResponse(BaseModel):
    """ベクトル化レスポンス"""
    message: str
    job_id: int
    directory_path: str


class JobResponse(BaseModel):
    """ジョブ情報レスポンス"""
    id: int
    job_type: str
    status: str
    parameters: dict
    progress: dict
    result: Optional[dict] = None
    error_message: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: str


class RAGRequest(BaseModel):
    """RAG回答生成リクエスト"""
    query: str = Field(..., description="ユーザーの質問")
    limit: int = Field(default=20, ge=1, le=100, description="検索結果の最大数")
    hybrid_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="ベクトル検索の重み")
    keyword_limit: int = Field(default=10, ge=1, le=50, description="各キーワードあたりの全文検索取得件数")
    vector_limit: int = Field(default=20, ge=1, le=100, description="ベクトル検索の取得件数")
    expand_synonyms: bool = Field(default=False, description="類義語展開を使用するかどうか")
    llm_provider: Optional[str] = Field(default=None, description="LLMプロバイダー（openrouter, litellm, aws_bedrock）")
    model: Optional[str] = Field(default=None, description="使用するLLMモデル名またはARN（ARNは推論エンドポイントの場合に使用）")
    api_base: Optional[str] = Field(default=None, description="LiteLLMのカスタムエンドポイントURL（litellmプロバイダーの場合のみ）")
    aws_region_name: Optional[str] = Field(default=None, description="AWSリージョン名（aws_bedrockプロバイダーの場合のみ）")
    aws_access_key_id: Optional[str] = Field(default=None, description="AWSアクセスキーID（aws_bedrockプロバイダーの場合のみ）")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWSシークレットキー（aws_bedrockプロバイダーの場合のみ）")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度パラメータ")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="最大トークン数")


class RAGResponse(BaseModel):
    """RAG回答生成レスポンス"""
    query: str
    answer: str
    sources: List[SearchResult]
    model_used: str
    provider_used: str


# グローバル変数（シングルトン）
db = SearchDatabase()
tokenizer = JapaneseTokenizer()

# 監視スケジューラー（グローバル）
_watcher_thread: Optional[threading.Thread] = None
_watcher_running = False
_watcher_lock = threading.Lock()

# ベクトルストアとベクトライザーの初期化（遅延初期化）
_vector_store: Optional[VectorStore] = None
_vectorizer: Optional[DocumentVectorizer] = None


def get_vectorizer() -> DocumentVectorizer:
    """ベクトライザーを取得（シングルトン）"""
    global _vector_store, _vectorizer
    
    if _vectorizer is None:
        # Embeddingプロバイダーを環境変数から取得（デフォルトはOpenRouter）
        provider_type_str = os.environ.get("EMBEDDING_PROVIDER", "openrouter")
        try:
            provider_type = EmbeddingProviderType(provider_type_str)
        except ValueError:
            provider_type = EmbeddingProviderType.OPENROUTER
        
        # プロバイダーを作成
        embedding_provider = create_embedding_provider(provider_type)
        
        # ベクトルストアを作成
        _vector_store = VectorStore()
        
        # チャンカーを作成
        chunker = TextChunker(
            chunk_size=1024,
            chunk_overlap=50,
            tokenizer=tokenizer
        )
        
        # ベクトライザーを作成
        _vectorizer = DocumentVectorizer(
            db=db,
            vector_store=_vector_store,
            embedding_provider=embedding_provider,
            chunker=chunker
        )
    
    return _vectorizer


def scan_directory(directory_path: str) -> List[str]:
    """
    ディレクトリ内のファイルを再帰的にスキャン
    
    Args:
        directory_path: スキャンするディレクトリのパス
    
    Returns:
        ファイルパスのリスト
    """
    files = []
    path = Path(directory_path)
    
    if not path.exists():
        raise ValueError(f"ディレクトリが存在しません: {directory_path}")
    
    if not path.is_dir():
        raise ValueError(f"ディレクトリではありません: {directory_path}")
    
    # サポートされている拡張子
    supported_extensions = {
        '.pdf', '.docx', '.pptx', '.xlsx',
        '.txt', '.md', '.markdown',
        '.py', '.js', '.ts', '.jsx', '.tsx',
        '.json', '.xml', '.html', '.css',
        '.yaml', '.yml', '.csv'
    }
    
    # 除外する拡張子（データベースファイルなど）
    excluded_extensions = {
        '.db', '.sqlite', '.sqlite3', '.db3',
        '.mdb', '.accdb',  # Access
        '.fdb', '.gdb',  # Firebird
        '.sql',  # SQLスクリプト（通常はインデックス対象外）
        '.dbf',  # dBASE
        '.ldb', '.laccdb'  # Access ロックファイル
    }
    
    # 除外するディレクトリ名
    excluded_dirs = {
        '.git', '.svn', '.hg',  # バージョン管理
        'node_modules', '__pycache__', '.pytest_cache',  # キャッシュ
        'venv', 'env', '.venv',  # 仮想環境
        'chroma_db',  # ChromaDBのデータディレクトリ
        '.obsidian',  # Obsidianの設定ディレクトリ
        'dist', 'build', '.next', '.nuxt',  # ビルド成果物
    }
    
    # データベースファイルのパスを取得（除外用）
    db_path = os.path.abspath(db.db_path) if hasattr(db, 'db_path') else None
    
    for root, dirs, filenames in os.walk(directory_path):
        # 除外するディレクトリをスキップ
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith('.')]
        
        for filename in filenames:
            file_path = os.path.join(root, filename)
            file_path_abs = os.path.abspath(file_path)
            
            # データベースファイル自体を除外
            if db_path and file_path_abs == db_path:
                continue
            
            # 除外する拡張子をスキップ
            ext = Path(filename).suffix.lower()
            if ext in excluded_extensions:
                continue
            
            # サポートされている拡張子のみを追加
            if ext in supported_extensions:
                files.append(file_path)
    
    return files


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


def index_file(
    file_path: str, 
    db: SearchDatabase, 
    tokenizer: JapaneseTokenizer,
    job_id: Optional[int] = None
) -> bool:
    """
    1つのファイルをインデックスに追加
    ファイルの編集日時が変更されていない場合はスキップ
    コードファイルの場合はコードパーサーを使用
    
    Args:
        file_path: インデックスに追加するファイルのパス
        db: データベースインスタンス
        tokenizer: トークナイザーインスタンス
        job_id: ジョブID（進捗更新用）
    
    Returns:
        成功した場合、またはスキップされた場合はTrue、失敗した場合はFalse
    """
    from .parsers import CodeParser, get_code_parser
    
    # コードファイルの場合はコードパーサーを使用
    code_parser = get_code_parser(file_path)
    if code_parser:
        return index_code_file(file_path, db, code_parser)
    
    # 通常のファイルの場合は既存の処理
    parser = get_parser(file_path)
    if not parser:
        return False
    
    try:
        # ファイルの最終更新日時を取得
        file_modified_time = get_file_modified_time(file_path)
        
        # ファイルを解析
        parsed_data = parser.parse(file_path)
        
        file_type = Path(file_path).suffix.lower().lstrip('.')
        
        # 更新されたかどうかを追跡
        updated = False
        skipped = True
        
        for data in parsed_data:
            # テキストを分かち書き
            tokenized_content = tokenizer.tokenize(data['content'])
            
            # tokenized_contentが空文字列でもデータベースに保存（空文字列は許可）
            try:
                # データベースに追加または更新
                result_id = db.update_document(
                    file_path=file_path,
                    file_type=file_type,
                    location_info=data['location_info'],
                    content=tokenized_content if tokenized_content else "",
                    file_modified_time=file_modified_time
                )
                
                # result_idが-1の場合はスキップされた
                if result_id == -1:
                    skipped = True
                else:
                    updated = True
                    skipped = False
            except Exception as db_error:
                # データベースエラーの詳細をログに出力
                import traceback
                error_details = traceback.format_exc()
                print(f"データベースエラーが発生しました: {file_path}")
                print(f"  エラー: {str(db_error)}")
                print(f"  location_info: {data.get('location_info', 'N/A')}")
                print(f"  詳細: {error_details}")
                # データベースエラーが発生した場合は、このファイルの処理を失敗として扱う
                raise
        
        # 更新されたか、スキップされた場合は成功として扱う
        return updated or skipped
    except Exception as e:
        # エラーが発生したファイルはログに記録してスキップ
        import traceback
        error_details = traceback.format_exc()
        print(f"ファイルのインデックス作成に失敗しました: {file_path}")
        print(f"  エラー: {str(e)}")
        print(f"  詳細: {error_details}")
        return False


def index_code_file(
    file_path: str,
    db: SearchDatabase,
    code_parser
) -> bool:
    """
    コードファイルをインデックスに追加
    
    Args:
        file_path: コードファイルのパス
        db: データベースインスタンス
        code_parser: コードパーサーインスタンス
    
    Returns:
        成功した場合、またはスキップされた場合はTrue、失敗した場合はFalse
    """
    import hashlib
    
    try:
        # ファイルの最終更新日時を取得
        file_modified_time = get_file_modified_time(file_path)
        
        # 既存のファイル情報を取得
        existing_file = db.get_code_file(file_path)
        
        # ファイルの内容を読み込んでハッシュを計算
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        except Exception:
            return False
        
        # 既存のファイルがあり、ハッシュと更新日時が同じ場合はスキップ
        if existing_file:
            if (existing_file.get('content_hash') == content_hash and 
                existing_file.get('file_modified_time') == file_modified_time):
                return True  # スキップ
        
        # ファイルを解析してトークンを抽出
        tokens = code_parser.parse(file_path)
        
        if not tokens:
            return False
        
        # 言語を判定
        ext = Path(file_path).suffix.lower().lstrip('.')
        language_map = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'jsx': 'javascript',
            'tsx': 'typescript',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'h': 'c',
            'hpp': 'cpp',
            'cs': 'csharp',
            'go': 'go',
            'rs': 'rust',
            'rb': 'ruby',
            'php': 'php',
            'swift': 'swift',
            'kt': 'kotlin',
            'scala': 'scala',
            'r': 'r',
            'm': 'objective-c',
            'mm': 'objective-cpp',
            'sh': 'bash',
            'bash': 'bash',
            'zsh': 'zsh',
            'fish': 'fish',
            'ps1': 'powershell',
            'bat': 'batch',
            'cmd': 'batch'
        }
        language = language_map.get(ext, ext)
        
        # コードファイルをデータベースに追加または更新
        file_id = db.add_code_file(
            file_path=file_path,
            language=language,
            content_hash=content_hash,
            file_modified_time=file_modified_time
        )
        
        # コードインデックスを追加
        db.add_code_indices(file_id, tokens)
        
        return True
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"コードファイルのインデックス作成に失敗しました: {file_path}")
        print(f"  エラー: {str(e)}")
        print(f"  詳細: {error_details}")
        return False


def process_watched_directory_scan(dir_id: int, directory_path: str):
    """
    監視対象ディレクトリのスキャンを実行（監視用、clear_existing=False）
    
    Args:
        dir_id: 監視対象ディレクトリID
        directory_path: スキャンするディレクトリのパス
    """
    import time
    start_time = time.time()
    
    try:
        # ディレクトリ内のファイルをスキャン
        files = scan_directory(directory_path)
        total_files = len(files)
        
        if total_files == 0:
            scan_duration = time.time() - start_time
            db.update_watched_directory_scan_info(dir_id, scan_duration)
            return
        
        # 各ファイルをインデックスに追加（clear_existing=Falseなので既存は更新のみ）
        indexed_files = 0
        failed_files = 0
        
        for idx, file_path in enumerate(files):
            # ファイルをインデックスに追加
            # index_fileは、更新またはスキップされた場合はTrueを返す
            # ファイルの編集日時をチェックして、変更がない場合は自動的にスキップされる
            if index_file(file_path, db, tokenizer, None):
                indexed_files += 1
            else:
                failed_files += 1
        
        # スキャン時間を記録
        scan_duration = time.time() - start_time
        db.update_watched_directory_scan_info(dir_id, scan_duration)
        
        print(f"監視スキャン完了: {directory_path} ({indexed_files}ファイル処理, {scan_duration:.2f}秒)")
        
    except Exception as e:
        scan_duration = time.time() - start_time
        db.update_watched_directory_scan_info(dir_id, scan_duration)
        error_msg = str(e)
        print(f"監視スキャンエラー: {directory_path} - {error_msg}")


def watcher_worker():
    """
    監視対象ディレクトリを定期的にスキャンするワーカースレッド
    """
    import time
    from datetime import datetime, timedelta
    
    global _watcher_running
    
    while _watcher_running:
        try:
            # 有効な監視対象ディレクトリを取得
            watched_dirs = db.get_watched_directories(enabled_only=True)
            
            for dir_info in watched_dirs:
                if not _watcher_running:
                    break
                
                dir_id = dir_info['id']
                directory_path = dir_info['directory_path']
                scan_interval_minutes = dir_info['scan_interval_minutes']
                last_scan_at = dir_info['last_scan_at']
                last_scan_duration_seconds = dir_info['last_scan_duration_seconds'] or 0
                
                # 次回スキャン時刻を計算
                if last_scan_at:
                    try:
                        last_scan_time = datetime.fromisoformat(last_scan_at.replace('Z', '+00:00'))
                        next_scan_time = last_scan_time + timedelta(minutes=scan_interval_minutes)
                    except:
                        next_scan_time = datetime.now()
                else:
                    # 初回スキャン
                    next_scan_time = datetime.now()
                
                # スキャン間隔の自動調整
                # スキャン時間が指定間隔を超えている場合は、スキャン時間+1分に設定
                if last_scan_duration_seconds > 0:
                    scan_duration_minutes = last_scan_duration_seconds / 60.0
                    if scan_duration_minutes >= scan_interval_minutes:
                        # スキャン時間+1分に間隔を調整
                        new_interval = int(scan_duration_minutes) + 1
                        db.update_watched_directory(dir_id, scan_interval_minutes=new_interval)
                        print(f"スキャン間隔を自動調整: {directory_path} -> {new_interval}分（スキャン時間: {scan_duration_minutes:.2f}分）")
                        scan_interval_minutes = new_interval
                        next_scan_time = datetime.now() + timedelta(minutes=scan_interval_minutes)
                
                # スキャンが必要かチェック
                now = datetime.now()
                if now >= next_scan_time:
                    print(f"監視スキャン開始: {directory_path}")
                    process_watched_directory_scan(dir_id, directory_path)
            
            # 1分ごとにチェック
            time.sleep(60)
            
        except Exception as e:
            print(f"監視ワーカーエラー: {str(e)}")
            time.sleep(60)


def start_watcher():
    """監視スケジューラーを開始"""
    global _watcher_thread, _watcher_running
    
    with _watcher_lock:
        if _watcher_running:
            return
        
        _watcher_running = True
        _watcher_thread = threading.Thread(target=watcher_worker, daemon=True)
        _watcher_thread.start()
        print("監視スケジューラーを開始しました")


def stop_watcher():
    """監視スケジューラーを停止"""
    global _watcher_running
    
    with _watcher_lock:
        _watcher_running = False


def process_index_job(job_id: int, directory_path: str, clear_existing: bool):
    """
    インデックス作成ジョブをバックグラウンドで実行
    
    Args:
        job_id: ジョブID
        directory_path: インデックスを作成するディレクトリのパス
        clear_existing: 既存のインデックスをクリアするか
    """
    try:
        # ジョブステータスを処理中に更新
        db.job_queue.update_job_status(job_id, JobStatus.PROCESSING)
        
        # 既存のインデックスをクリアする場合
        if clear_existing:
            db.clear_all()
            db.job_queue.update_job_progress(job_id, 0, 0, "既存インデックスをクリアしました")
        
        # ディレクトリ内のファイルをスキャン
        files = scan_directory(directory_path)
        total_files = len(files)
        
        if total_files == 0:
            db.job_queue.update_job_status(job_id, JobStatus.COMPLETED)
            db.job_queue.update_job_result(job_id, {
                "message": "インデックス対象のファイルが見つかりませんでした",
                "indexed_files": 0,
                "indexed_documents": db.get_document_count()
            })
            return
        
        # 各ファイルをインデックスに追加
        indexed_files = 0
        failed_files = 0
        
        for idx, file_path in enumerate(files):
            # 進捗を更新
            db.job_queue.update_job_progress(
                job_id,
                current=idx + 1,
                total=total_files,
                message=f"処理中: {os.path.basename(file_path)}"
            )
            
            # ファイルをインデックスに追加
            # index_fileは、更新またはスキップされた場合はTrueを返す
            # ファイルの編集日時をチェックして、変更がない場合は自動的にスキップされる
            if index_file(file_path, db, tokenizer, job_id):
                indexed_files += 1
            else:
                failed_files += 1
        
        # 完了
        final_doc_count = db.get_document_count()
        db.job_queue.update_job_status(job_id, JobStatus.COMPLETED)
        db.job_queue.update_job_result(job_id, {
            "message": "インデックス作成が完了しました",
            "indexed_files": indexed_files,
            "failed_files": failed_files,
            "total_files": total_files,
            "indexed_documents": final_doc_count
        })
        
    except Exception as e:
        # エラーを記録
        error_msg = str(e)
        db.job_queue.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
        print(f"インデックス作成ジョブ {job_id} が失敗しました: {error_msg}")


@router.post("/index", response_model=IndexResponse)
async def create_index(request: IndexRequest):
    """
    指定されたディレクトリ内のファイルをスキャンしてインデックスを作成（バックグラウンド処理）
    
    Args:
        request: インデックス作成リクエスト
    
    Returns:
        ジョブIDを含むレスポンス
    """
    try:
        directory_path = os.path.abspath(request.directory_path)
        
        # ジョブを作成
        job_id = db.job_queue.create_job(
            job_type=JobType.INDEX.value,
            parameters={
                "directory_path": directory_path,
                "clear_existing": request.clear_existing
            },
            initial_progress={
                "current": 0,
                "total": 0,
                "percentage": 0,
                "message": "ジョブを開始しています..."
            }
        )
        
        # バックグラウンドスレッドで処理を開始
        thread = threading.Thread(
            target=process_index_job,
            args=(job_id, directory_path, request.clear_existing),
            daemon=True
        )
        thread.start()
        
        return IndexResponse(
            message="インデックス作成ジョブを開始しました",
            job_id=job_id,
            directory_path=directory_path
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"インデックス作成ジョブの作成中にエラーが発生しました: {str(e)}")


@router.get("/file-content")
async def get_file_content(file_path: str):
    """
    ファイルの内容を取得
    
    Args:
        file_path: ファイルパス（URLエンコード済み）
    
    Returns:
        ファイルの内容とメタデータ
    """
    if not file_path:
        raise HTTPException(status_code=400, detail="ファイルパスが指定されていません")
    
    # URLデコード（FastAPIは自動的にデコードするが、念のため）
    import urllib.parse
    # バックスラッシュが失われている可能性があるため、スラッシュをバックスラッシュに変換
    # Windowsパスの場合、URLエンコード時にバックスラッシュが失われる可能性がある
    decoded_path = urllib.parse.unquote(file_path)
    # スラッシュをバックスラッシュに変換（Windowsパスの場合）
    if os.name == 'nt':  # Windows
        decoded_path = decoded_path.replace('/', '\\')
    
    # ファイルパスの安全性チェック
    if not os.path.exists(decoded_path):
        raise HTTPException(status_code=404, detail=f"ファイルが見つかりません: {decoded_path}")
    
    try:
        file_path_obj = Path(decoded_path)
        ext = file_path_obj.suffix.lower()
        
        # PDFファイルの場合
        if ext == '.pdf':
            # URLエンコード（バックスラッシュをスラッシュに変換してからエンコード）
            encoded_path = urllib.parse.quote(decoded_path.replace('\\', '/'), safe='')
            return {
                "file_path": decoded_path,
                "file_type": "pdf",
                "content": None,  # PDFは埋め込み表示のため、内容は不要
                "can_embed": True,
                "file_url": f"/search/file?file_path={encoded_path}"
            }
        
        # コードファイルの場合
        code_extensions = {'.py', '.ts', '.tsx', '.js', '.jsx', '.java', '.cpp', '.c', '.h', '.hpp', 
                          '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r', 
                          '.m', '.mm', '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
                          '.json', '.xml', '.html', '.css', '.yaml', '.yml'}
        if ext in code_extensions:
            # コードファイルとして読み込み
            content = ""
            encodings = ['utf-8', 'shift-jis', 'cp932', 'euc-jp', 'iso-2022-jp']
            for encoding in encodings:
                try:
                    with open(decoded_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        break
                except (UnicodeDecodeError, LookupError):
                    continue
            
            if not content:
                try:
                    with open(decoded_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except Exception:
                    raise HTTPException(status_code=500, detail="ファイルの読み込みに失敗しました")
            
            # 言語を判定
            language_map = {
                '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                '.jsx': 'javascript', '.tsx': 'typescript', '.java': 'java',
                '.cpp': 'cpp', '.c': 'c', '.h': 'c', '.hpp': 'cpp',
                '.cs': 'csharp', '.go': 'go', '.rs': 'rust', '.rb': 'ruby',
                '.php': 'php', '.swift': 'swift', '.kt': 'kotlin', '.scala': 'scala',
                '.r': 'r', '.m': 'objective-c', '.mm': 'objective-cpp',
                '.sh': 'bash', '.bash': 'bash', '.zsh': 'zsh', '.fish': 'fish',
                '.ps1': 'powershell', '.bat': 'batch', '.cmd': 'batch',
                '.json': 'json', '.xml': 'xml', '.html': 'html', '.css': 'css',
                '.yaml': 'yaml', '.yml': 'yaml'
            }
            language = language_map.get(ext, 'text')
            
            return {
                "file_path": decoded_path,
                "file_type": "code",
                "language": language,
                "content": content,
                "can_embed": False
            }
        
        # Markdownファイルの場合
        if ext in {'.md', '.markdown'}:
            parser = get_parser(decoded_path)
            if parser:
                parsed = parser.parse(decoded_path)
                content = parsed[0]['content'] if parsed else ""
                return {
                    "file_path": decoded_path,
                    "file_type": "markdown",
                    "content": content,
                    "can_embed": False
                }
        
        # テキストファイルの場合
        if ext == '.txt':
            parser = get_parser(decoded_path)
            if parser:
                parsed = parser.parse(decoded_path)
                content = parsed[0]['content'] if parsed else ""
                return {
                    "file_path": decoded_path,
                    "file_type": "text",
                    "content": content,
                    "can_embed": False
                }
        
        # その他のファイルタイプ
        raise HTTPException(status_code=400, detail=f"サポートされていないファイルタイプ: {ext}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ファイルの読み込み中にエラーが発生しました: {str(e)}")


@router.get("/file")
async def serve_file(file_path: str):
    """
    ファイルを直接返す（PDFなど埋め込み表示用）
    
    Args:
        file_path: ファイルパス（URLエンコード済み）
    
    Returns:
        ファイルの内容
    """
    if not file_path:
        raise HTTPException(status_code=400, detail="ファイルパスが指定されていません")
    
    # URLデコード
    import urllib.parse
    decoded_path = urllib.parse.unquote(file_path)
    # スラッシュをバックスラッシュに変換（Windowsパスの場合）
    if os.name == 'nt':  # Windows
        decoded_path = decoded_path.replace('/', '\\')
    
    # ファイルパスの安全性チェック
    if not os.path.exists(decoded_path):
        raise HTTPException(status_code=404, detail=f"ファイルが見つかりません: {decoded_path}")
    
    file_path_obj = Path(decoded_path)
    ext = file_path_obj.suffix.lower()
    
    # PDFファイルの場合
    if ext == '.pdf':
        return FileResponse(
            decoded_path,
            media_type='application/pdf',
            filename=file_path_obj.name
        )
    
    # その他のファイルタイプはエラー
    raise HTTPException(status_code=400, detail=f"サポートされていないファイルタイプ: {ext}")


@router.get("/query", response_model=SearchResponse)
async def search(query: str, limit: int = 50):
    """
    キーワードで全文検索を実行
    
    Args:
        query: 検索キーワード
        limit: 返却する結果の最大数（1-100）
    
    Returns:
        検索結果
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="検索キーワードが指定されていません")
    
    if limit < 1 or limit > 100:
        limit = 50
    
    try:
        # クエリを分かち書き（検索の精度を向上）
        tokenized_query = tokenizer.tokenize(query)
        
        # 分かち書きされたクエリで検索を実行
        search_query = tokenized_query if tokenized_query else query
        
        # データベースで検索
        results = db.search(search_query, limit=limit)
        
        # レスポンスモデルに変換
        search_results = [
            SearchResult(
                file_path=result['file_path'],
                file_type=result['file_type'],
                location_info=result['location_info'],
                snippet=result['snippet']
            )
            for result in results
        ]
        
        return SearchResponse(
            query=query,
            results=search_results,
            total=len(search_results)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"検索中にエラーが発生しました: {str(e)}")


@router.post("/query", response_model=SearchResponse)
async def search_post(request: SearchRequest):
    """
    キーワードで全文検索を実行（POST版）
    
    Args:
        request: 検索リクエスト
    
    Returns:
        検索結果
    """
    return await search(query=request.query, limit=request.limit)


# コード検索用のリクエスト/レスポンスモデル
class CodeSearchRequest(BaseModel):
    """コード検索リクエスト"""
    query: str = Field(..., description="検索キーワード（トークン）")
    limit: int = Field(default=50, ge=1, le=100, description="返却する結果の最大数")
    language: Optional[str] = Field(default=None, description="プログラミング言語でフィルタリング（例: python, javascript）")


class CodeSearchResult(BaseModel):
    """コード検索結果"""
    file_path: str
    language: Optional[str]
    line_number: int
    column_number: Optional[int]
    token_type: Optional[str]
    token: str


class CodeSearchResponse(BaseModel):
    """コード検索レスポンス"""
    query: str
    results: List[CodeSearchResult]
    total: int


@router.get("/code", response_model=CodeSearchResponse)
async def search_code(
    query: str,
    limit: int = 50,
    language: Optional[str] = None
):
    """
    コード検索を実行（GET版）
    
    Args:
        query: 検索キーワード（トークン）
        limit: 返却する結果の最大数（1-100）
        language: プログラミング言語でフィルタリング（オプション）
    
    Returns:
        コード検索結果
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="検索キーワードが指定されていません")
    
    if limit < 1 or limit > 100:
        limit = 50
    
    try:
        # コード検索を実行
        results = db.search_code(query=query, limit=limit, language=language)
        
        # レスポンスモデルに変換
        search_results = [
            CodeSearchResult(
                file_path=result['file_path'],
                language=result['language'],
                line_number=result['line_number'],
                column_number=result['column_number'],
                token_type=result['token_type'],
                token=result['token']
            )
            for result in results
        ]
        
        return CodeSearchResponse(
            query=query,
            results=search_results,
            total=len(search_results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"コード検索中にエラーが発生しました: {str(e)}")


@router.post("/code", response_model=CodeSearchResponse)
async def search_code_post(request: CodeSearchRequest):
    """
    コード検索を実行（POST版）
    
    Args:
        request: コード検索リクエスト
    
    Returns:
        コード検索結果
    """
    return await search_code(query=request.query, limit=request.limit, language=request.language)


@router.get("/stats")
async def get_stats():
    """
    インデックス統計情報を取得
    
    Returns:
        統計情報（ドキュメント数など）
    """
    try:
        count = db.get_document_count()
        return {
            "total_documents": count,
            "database_path": db.db_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"統計情報の取得中にエラーが発生しました: {str(e)}")


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: int):
    """
    ジョブ情報を取得
    
    Args:
        job_id: ジョブID
    
    Returns:
        ジョブ情報
    """
    job = db.job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"ジョブ {job_id} が見つかりません")
    
    return JobResponse(**job)


@router.get("/jobs", response_model=List[JobResponse])
async def list_jobs(
    job_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """
    ジョブ一覧を取得
    
    Args:
        job_type: ジョブタイプでフィルタ
        status: ステータスでフィルタ（pending, processing, completed, failed, cancelled）
        limit: 取得件数の上限
    
    Returns:
        ジョブ情報のリスト
    """
    try:
        status_enum = None
        if status:
            try:
                status_enum = JobStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"無効なステータス: {status}")
        
        jobs = db.job_queue.get_jobs(job_type=job_type, status=status_enum, limit=limit)
        return [JobResponse(**job) for job in jobs]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ジョブ一覧の取得中にエラーが発生しました: {str(e)}")


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: int):
    """
    ジョブをキャンセル
    
    Args:
        job_id: ジョブID
    
    Returns:
        キャンセル結果
    """
    success = db.job_queue.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=400, 
            detail=f"ジョブ {job_id} をキャンセルできませんでした（存在しないか、既に完了/失敗/キャンセル済みです）"
        )
    
    return {
        "message": f"ジョブ {job_id} をキャンセルしました",
        "job_id": job_id
    }


@router.delete("/index")
async def clear_index():
    """
    すべてのインデックスをクリア
    
    Returns:
        クリア結果
    """
    try:
        db.clear_all()
        return {
            "message": "インデックスがクリアされました",
            "total_documents": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"インデックスクリア中にエラーが発生しました: {str(e)}")


def load_html_template(template_name: str = "index_status.html") -> str:
    """HTMLテンプレートを読み込む"""
    template_path = Path(__file__).parent / "templates" / template_name
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <title>エラー</title>
        </head>
        <body>
            <h1>テンプレートファイルが見つかりません: {template_name}</h1>
        </body>
        </html>
        """


# タスク管理用のルーター（/taskパス用）
task_router = APIRouter(
    prefix="/task",
    tags=["task"],
    responses={404: {"description": "Not found"}},
)


@task_router.get("/search", response_class=HTMLResponse)
async def search_page():
    """
    キーワード検索ページを表示
    """
    html_content = load_html_template("search.html")
    return HTMLResponse(content=html_content, status_code=200)


@task_router.get("/", response_class=HTMLResponse)
async def task_index():
    """
    タスク管理のメインページ
    
    Returns:
        HTMLページ
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>タスク管理 - Obsidian MCP Server</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 40px 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                margin-bottom: 30px;
            }
            .task-list {
                display: grid;
                gap: 20px;
            }
            .task-card {
                background: white;
                border-radius: 8px;
                padding: 24px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .task-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }
            .task-card h2 {
                margin: 0 0 12px 0;
                color: #2c3e50;
            }
            .task-card p {
                color: #666;
                margin: 0 0 16px 0;
            }
            .task-card a {
                display: inline-block;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                padding: 10px 20px;
                border-radius: 4px;
                transition: background-color 0.2s;
            }
            .task-card a:hover {
                background-color: #0056b3;
            }
        </style>
    </head>
    <body>
        <h1>タスク管理</h1>
        <div class="task-list">
            <div class="task-card">
                <h2>インデックス作成</h2>
                <p>ディレクトリをスキャンして全文検索用のインデックスを作成します。</p>
                <a href="/task/create_index">インデックス作成ページへ</a>
            </div>
            <div class="task-card">
                <h2>ベクトル化</h2>
                <p>ドキュメントをベクトル化してベクトル検索用のデータを作成します。</p>
                <a href="/task/create_vector">ベクトル化ページへ</a>
            </div>
            <div class="task-card">
                <h2>インデックスリスト</h2>
                <p>インデックスされたファイルの一覧を確認できます。</p>
                <a href="/task/index_lists/">インデックスリストページへ</a>
            </div>
            <div class="task-card">
                <h2>監視対象ディレクトリ</h2>
                <p>監視対象ディレクトリを設定し、変更を自動的にインデックスに反映します。</p>
                <a href="/task/target_index_lists">監視対象ディレクトリ管理ページへ</a>
            </div>
            <div class="task-card">
                <h2>🔍 キーワード検索</h2>
                <p>インデックスされたドキュメントからキーワードを検索し、ファイル内容を表示します。</p>
                <a href="/task/search">キーワード検索ページへ</a>
            </div>
            <div class="task-card">
                <h2>RAG質問応答</h2>
                <p>知識ベースに対して質問をして、AIが回答を生成します。</p>
                <a href="/task/rag">RAG質問応答ページへ</a>
            </div>
            <div class="task-card">
                <h2>LLM設定</h2>
                <p>LLMとEmbeddingモデルの設定を行います。</p>
                <a href="/task/llm-settings/">LLM設定ページへ</a>
            </div>
            <div class="task-card">
                <h2>💾 データベースサイズ管理</h2>
                <p>データベースの保存サイズと統計情報を確認できます。</p>
                <a href="/task/db_status">データベースサイズ管理ページへ</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@task_router.get("/create_index", response_class=HTMLResponse)
async def create_index_page():
    """
    インデックス作成状況を確認するWebページ
    
    Returns:
        HTMLページ
    """
    html_content = load_html_template("index_status.html")
    return HTMLResponse(content=html_content, status_code=200)


@task_router.get("/create_vector", response_class=HTMLResponse)
async def create_vector_page():
    """
    ベクトル化状況を確認するWebページ
    
    Returns:
        HTMLページ
    """
    html_content = load_html_template("vector_status.html")
    return HTMLResponse(content=html_content, status_code=200)


@task_router.get("/db_status", response_class=HTMLResponse)
async def db_status_page():
    """
    データベースサイズ管理ページ
    
    Returns:
        HTMLページ
    """
    html_content = load_html_template("db_status.html")
    return HTMLResponse(content=html_content, status_code=200)


@task_router.get("/index_lists/", response_class=HTMLResponse)
async def index_lists_page(
    search: Optional[str] = None,
    page: int = 1,
    per_page: int = 50
):
    """
    インデックスされたファイルのリストを表示するWebページ
    
    Args:
        search: 検索クエリ（ファイルパス、ファイルタイプ、位置情報で検索）
        page: ページ番号（1から開始）
        per_page: 1ページあたりの表示件数
    
    Returns:
        HTMLページ
    """
    try:
        # パラメータの検証
        if page < 1:
            page = 1
        if per_page < 1:
            per_page = 50
        if per_page > 500:
            per_page = 500
        
        # 検索クエリを正規化（空文字列はNoneに）
        search_query = search.strip() if search and search.strip() else None
        
        # 総件数を取得（検索条件付き）
        total_count = db.count_documents(search_query)
        
        # オフセットを計算
        offset = (page - 1) * per_page
        
        # データベースからドキュメントを取得（検索条件とページネーション付き）
        documents = db.search_documents(search_query=search_query, limit=per_page, offset=offset)
        
        # ファイルパスでグループ化（同じファイルの異なるlocation_infoをまとめる）
        files_dict = {}
        for doc in documents:
            file_path = doc['file_path']
            if file_path not in files_dict:
                files_dict[file_path] = {
                    'file_path': file_path,
                    'file_type': doc['file_type'],
                    'locations': [],
                    'updated_at': doc['updated_at'],
                    'file_modified_time': doc['file_modified_time']
                }
            if doc['location_info']:
                files_dict[file_path]['locations'].append(doc['location_info'])
        
        # ファイルリストをソート
        files_list = sorted(files_dict.values(), key=lambda x: x['file_path'])
        
        # ページネーション情報を計算
        total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1
        if page > total_pages:
            page = total_pages
        
        # ページネーションHTMLを生成（HTMLテンプレートより前に生成）
        def generate_pagination_html(current_page, total_pages, search_query, per_page):
            if total_pages <= 1:
                return ""
            
            from urllib.parse import quote
            
            pagination_html = '<div class="pagination">'
            
            # 前のページ
            if current_page > 1:
                prev_params = f"?page={current_page - 1}&per_page={per_page}"
                if search_query:
                    prev_params += f"&search={quote(search_query)}"
                pagination_html += f'<a href="/task/index_lists/{prev_params}">« 前</a>'
            else:
                pagination_html += '<span class="disabled">« 前</span>'
            
            # ページ番号
            start_page = max(1, current_page - 2)
            end_page = min(total_pages, current_page + 2)
            
            if start_page > 1:
                params = f"?page=1&per_page={per_page}"
                if search_query:
                    params += f"&search={quote(search_query)}"
                pagination_html += f'<a href="/task/index_lists/{params}">1</a>'
                if start_page > 2:
                    pagination_html += '<span>...</span>'
            
            for p in range(start_page, end_page + 1):
                if p == current_page:
                    pagination_html += f'<span class="current">{p}</span>'
                else:
                    params = f"?page={p}&per_page={per_page}"
                    if search_query:
                        params += f"&search={quote(search_query)}"
                    pagination_html += f'<a href="/task/index_lists/{params}">{p}</a>'
            
            if end_page < total_pages:
                if end_page < total_pages - 1:
                    pagination_html += '<span>...</span>'
                params = f"?page={total_pages}&per_page={per_page}"
                if search_query:
                    params += f"&search={quote(search_query)}"
                pagination_html += f'<a href="/task/index_lists/{params}">{total_pages}</a>'
            
            # 次のページ
            if current_page < total_pages:
                next_params = f"?page={current_page + 1}&per_page={per_page}"
                if search_query:
                    next_params += f"&search={quote(search_query)}"
                pagination_html += f'<a href="/task/index_lists/{next_params}">次 »</a>'
            else:
                pagination_html += '<span class="disabled">次 »</span>'
            
            pagination_html += '</div>'
            return pagination_html
        
        pagination_html = generate_pagination_html(page, total_pages, search_query, per_page)
        
        # HTMLを生成
        from datetime import datetime
        
        def format_datetime(dt_str):
            """日時をフォーマット"""
            if not dt_str:
                return "-"
            try:
                dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                return dt_str
        
        def format_timestamp(ts):
            """タイムスタンプをフォーマット"""
            if not ts:
                return "-"
            try:
                dt = datetime.fromtimestamp(ts)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                return "-"
        
        files_html = ""
        for file_info in files_list:
            file_path = file_info['file_path']
            file_type = file_info['file_type'] or "-"
            locations = file_info['locations']
            updated_at = format_datetime(file_info['updated_at'])
            file_modified_time = format_timestamp(file_info['file_modified_time'])
            
            locations_html = ""
            if locations:
                locations_html = f"<div class='locations'>{', '.join(locations)}</div>"
            
            files_html += f"""
            <tr>
                <td class="file-path">{file_path}</td>
                <td class="file-type">{file_type}</td>
                <td class="locations-cell">{locations_html if locations_html else '-'}</td>
                <td class="updated-at">{updated_at}</td>
                <td class="file-modified">{file_modified_time}</td>
            </tr>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>インデックスリスト - Obsidian MCP Server</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 30px;
                }}
                h1 {{
                    color: #333;
                    margin-bottom: 10px;
                }}
                .stats {{
                    color: #666;
                    margin-bottom: 20px;
                    font-size: 14px;
                }}
                .back-link {{
                    display: inline-block;
                    margin-bottom: 20px;
                    color: #007bff;
                    text-decoration: none;
                    font-size: 14px;
                }}
                .back-link:hover {{
                    text-decoration: underline;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th {{
                    background-color: #f8f9fa;
                    padding: 12px;
                    text-align: left;
                    font-weight: 600;
                    color: #333;
                    border-bottom: 2px solid #dee2e6;
                    position: sticky;
                    top: 0;
                }}
                td {{
                    padding: 12px;
                    border-bottom: 1px solid #dee2e6;
                    vertical-align: top;
                }}
                tr:hover {{
                    background-color: #f8f9fa;
                }}
                .file-path {{
                    font-family: 'Courier New', monospace;
                    font-size: 13px;
                    word-break: break-all;
                    max-width: 500px;
                }}
                .file-type {{
                    text-align: center;
                    min-width: 80px;
                }}
                .locations-cell {{
                    max-width: 300px;
                }}
                .locations {{
                    font-size: 12px;
                    color: #666;
                    word-break: break-word;
                }}
                .updated-at, .file-modified {{
                    font-size: 12px;
                    color: #666;
                    white-space: nowrap;
                }}
                .no-data {{
                    text-align: center;
                    padding: 40px;
                    color: #999;
                }}
                .search-form {{
                    margin-bottom: 20px;
                    display: flex;
                    gap: 10px;
                    align-items: center;
                }}
                .search-form input {{
                    flex: 1;
                    padding: 10px;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    font-size: 14px;
                }}
                .search-form button {{
                    padding: 10px 20px;
                    background-color: #007bff;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 14px;
                }}
                .search-form button:hover {{
                    background-color: #0056b3;
                }}
                .pagination {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    gap: 5px;
                    margin-top: 20px;
                    flex-wrap: wrap;
                }}
                .pagination a, .pagination span {{
                    padding: 8px 12px;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    text-decoration: none;
                    color: #007bff;
                    background-color: white;
                    font-size: 14px;
                }}
                .pagination a:hover {{
                    background-color: #f8f9fa;
                }}
                .pagination .current {{
                    background-color: #007bff;
                    color: white;
                    border-color: #007bff;
                }}
                .pagination .disabled {{
                    color: #6c757d;
                    cursor: not-allowed;
                    opacity: 0.6;
                }}
                .pagination-info {{
                    text-align: center;
                    margin-top: 10px;
                    color: #666;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <a href="/task/" class="back-link">← タスク管理に戻る</a>
                <h1>インデックスリスト</h1>
                <form method="get" action="/task/index_lists/" class="search-form">
                    <input type="text" name="search" placeholder="ファイルパス、ファイルタイプ、位置情報で検索..." value="{search_query or ''}">
                    <button type="submit">検索</button>
                    {f'<a href="/task/index_lists/" style="padding: 10px 15px; background-color: #6c757d; color: white; text-decoration: none; border-radius: 4px; font-size: 14px;">クリア</a>' if search_query else ''}
                </form>
                <div class="stats">
                    表示中: {len(files_list)} ファイル | 総ドキュメント数: {total_count} | ページ {page} / {total_pages}
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>ファイルパス</th>
                            <th>ファイルタイプ</th>
                            <th>位置情報</th>
                            <th>更新日時</th>
                            <th>ファイル更新日時</th>
                        </tr>
                    </thead>
                    <tbody>
                        {files_html if files_html else '<tr><td colspan="5" class="no-data">インデックスされたファイルがありません</td></tr>'}
                    </tbody>
                </table>
                {pagination_html}
                <div class="pagination-info">
                    {offset + 1 if total_count > 0 else 0} - {min(offset + per_page, total_count)} 件を表示（全 {total_count} 件）
                </div>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        error_html = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <title>エラー - Obsidian MCP Server</title>
        </head>
        <body>
            <h1>エラーが発生しました</h1>
            <p>{str(e)}</p>
            <a href="/task/">タスク管理に戻る</a>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=500)


@task_router.get("/target_index_lists", response_class=HTMLResponse)
async def target_index_lists_page():
    """
    監視対象ディレクトリの管理ページ
    
    Returns:
        HTMLページ
    """
    try:
        # 監視対象ディレクトリ一覧を取得
        watched_dirs = db.get_watched_directories()
        
        from datetime import datetime
        
        def format_datetime(dt_str):
            """日時をフォーマット"""
            if not dt_str:
                return "-"
            try:
                dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                return dt_str
        
        def format_duration(seconds):
            """時間をフォーマット"""
            if not seconds:
                return "-"
            try:
                sec = float(seconds)
                if sec < 60:
                    return f"{sec:.1f}秒"
                elif sec < 3600:
                    return f"{sec/60:.1f}分"
                else:
                    return f"{sec/3600:.1f}時間"
            except:
                return "-"
        
        dirs_html = ""
        for dir_info in watched_dirs:
            dir_id = dir_info['id']
            directory_path = dir_info['directory_path']
            scan_interval = dir_info['scan_interval_minutes']
            enabled = dir_info['enabled']
            last_scan_at = format_datetime(dir_info['last_scan_at'])
            last_scan_duration = format_duration(dir_info['last_scan_duration_seconds'])
            
            status_badge = '<span style="color: #28a745; font-weight: bold;">有効</span>' if enabled else '<span style="color: #dc3545; font-weight: bold;">無効</span>'
            
            # ディレクトリパスをエスケープ（JavaScript用）
            directory_path_escaped = directory_path.replace("'", "\\'").replace("\\", "\\\\")
            
            dirs_html += f"""
            <tr>
                <td class="directory-path">{directory_path}</td>
                <td class="scan-interval">{scan_interval}分</td>
                <td class="status">{status_badge}</td>
                <td class="last-scan">{last_scan_at}</td>
                <td class="scan-duration">{last_scan_duration}</td>
                <td class="actions">
                    <button onclick="vectorizeDirectory('{directory_path_escaped}')" class="btn-vectorize">ベクトル化</button>
                    <button onclick="toggleEnabled({dir_id}, {not enabled})" class="btn-toggle">{'無効化' if enabled else '有効化'}</button>
                    <button onclick="editDirectory({dir_id}, '{directory_path_escaped}', {scan_interval})" class="btn-edit">編集</button>
                    <button onclick="deleteDirectory({dir_id})" class="btn-delete">削除</button>
                </td>
            </tr>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>監視対象ディレクトリ管理 - Obsidian MCP Server</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 30px;
                }}
                h1 {{
                    color: #333;
                    margin-bottom: 10px;
                }}
                .back-link {{
                    display: inline-block;
                    margin-bottom: 20px;
                    color: #007bff;
                    text-decoration: none;
                    font-size: 14px;
                }}
                .back-link:hover {{
                    text-decoration: underline;
                }}
                .add-form {{
                    margin-bottom: 20px;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                }}
                .add-form h2 {{
                    margin-top: 0;
                    font-size: 18px;
                    color: #333;
                }}
                .form-group {{
                    margin-bottom: 15px;
                }}
                .form-group label {{
                    display: block;
                    margin-bottom: 5px;
                    font-weight: 600;
                    color: #333;
                }}
                .form-group input {{
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    font-size: 14px;
                }}
                .form-group input[type="number"] {{
                    width: 150px;
                }}
                .form-group input[type="checkbox"] {{
                    width: auto;
                    margin-right: 5px;
                }}
                .form-actions {{
                    display: flex;
                    gap: 10px;
                }}
                .btn {{
                    padding: 10px 20px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 14px;
                    font-weight: 600;
                }}
                .btn-primary {{
                    background-color: #007bff;
                    color: white;
                }}
                .btn-primary:hover {{
                    background-color: #0056b3;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th {{
                    background-color: #f8f9fa;
                    padding: 12px;
                    text-align: left;
                    font-weight: 600;
                    color: #333;
                    border-bottom: 2px solid #dee2e6;
                }}
                td {{
                    padding: 12px;
                    border-bottom: 1px solid #dee2e6;
                    vertical-align: middle;
                }}
                tr:hover {{
                    background-color: #f8f9fa;
                }}
                .directory-path {{
                    font-family: 'Courier New', monospace;
                    font-size: 13px;
                    word-break: break-all;
                }}
                .scan-interval, .status, .last-scan, .scan-duration {{
                    text-align: center;
                }}
                .actions {{
                    text-align: center;
                }}
                .actions button {{
                    padding: 5px 10px;
                    margin: 0 2px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                }}
                .btn-toggle {{
                    background-color: #ffc107;
                    color: #333;
                }}
                .btn-edit {{
                    background-color: #17a2b8;
                    color: white;
                }}
                .btn-delete {{
                    background-color: #dc3545;
                    color: white;
                }}
                .btn-vectorize {{
                    background-color: #28a745;
                    color: white;
                }}
                .no-data {{
                    text-align: center;
                    padding: 40px;
                    color: #999;
                }}
                .modal {{
                    display: none;
                    position: fixed;
                    z-index: 1000;
                    left: 0;
                    top: 0;
                    width: 100%;
                    height: 100%;
                    background-color: rgba(0,0,0,0.5);
                }}
                .modal-content {{
                    background-color: white;
                    margin: 15% auto;
                    padding: 20px;
                    border-radius: 8px;
                    width: 500px;
                }}
                .modal-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }}
                .close {{
                    font-size: 28px;
                    font-weight: bold;
                    cursor: pointer;
                    color: #aaa;
                }}
                .close:hover {{
                    color: #000;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <a href="/task/" class="back-link">← タスク管理に戻る</a>
                <h1>監視対象ディレクトリ管理</h1>
                
                <div class="add-form">
                    <h2>新しい監視対象を追加</h2>
                    <form id="addForm" onsubmit="addDirectory(event)">
                        <div class="form-group">
                            <label for="directory_path">ディレクトリパス:</label>
                            <input type="text" id="directory_path" name="directory_path" required placeholder="C:\\path\\to\\directory">
                        </div>
                        <div class="form-group">
                            <label for="scan_interval">スキャン間隔（分）:</label>
                            <input type="number" id="scan_interval" name="scan_interval" value="60" min="1" required>
                        </div>
                        <div class="form-group">
                            <label>
                                <input type="checkbox" id="enabled" name="enabled" checked>
                                有効にする
                            </label>
                        </div>
                        <div class="form-actions">
                            <button type="submit" class="btn btn-primary">追加</button>
                        </div>
                    </form>
                </div>
                
                <table>
                    <thead>
                        <tr>
                            <th>ディレクトリパス</th>
                            <th>スキャン間隔</th>
                            <th>ステータス</th>
                            <th>最終スキャン</th>
                            <th>スキャン時間</th>
                            <th>操作</th>
                        </tr>
                    </thead>
                    <tbody>
                        {dirs_html if dirs_html else '<tr><td colspan="6" class="no-data">監視対象ディレクトリがありません</td></tr>'}
                    </tbody>
                </table>
            </div>
            
            <!-- 編集モーダル -->
            <div id="editModal" class="modal">
                <div class="modal-content">
                    <div class="modal-header">
                        <h2>監視設定を編集</h2>
                        <span class="close" onclick="closeEditModal()">&times;</span>
                    </div>
                    <form id="editForm" onsubmit="updateDirectory(event)">
                        <input type="hidden" id="edit_id" name="id">
                        <div class="form-group">
                            <label for="edit_directory_path">ディレクトリパス:</label>
                            <input type="text" id="edit_directory_path" name="directory_path" readonly>
                        </div>
                        <div class="form-group">
                            <label for="edit_scan_interval">スキャン間隔（分）:</label>
                            <input type="number" id="edit_scan_interval" name="scan_interval" min="1" required>
                        </div>
                        <div class="form-group">
                            <label>
                                <input type="checkbox" id="edit_enabled" name="enabled">
                                有効にする
                            </label>
                        </div>
                        <div class="form-actions">
                            <button type="submit" class="btn btn-primary">更新</button>
                            <button type="button" class="btn" onclick="closeEditModal()">キャンセル</button>
                        </div>
                    </form>
                </div>
            </div>
            
            <script>
                async function addDirectory(event) {{
                    event.preventDefault();
                    const formData = new FormData(event.target);
                    const data = {{
                        directory_path: formData.get('directory_path'),
                        scan_interval_minutes: parseInt(formData.get('scan_interval')),
                        enabled: formData.get('enabled') === 'on'
                    }};
                    
                    try {{
                        const response = await fetch('/task/watched_directories', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify(data)
                        }});
                        
                        if (response.ok) {{
                            location.reload();
                        }} else {{
                            alert('エラーが発生しました: ' + await response.text());
                        }}
                    }} catch (error) {{
                        alert('エラーが発生しました: ' + error.message);
                    }}
                }}
                
                async function toggleEnabled(dirId, enabled) {{
                    try {{
                        const response = await fetch(`/task/watched_directories/${{dirId}}`, {{
                            method: 'PATCH',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{ enabled: enabled }})
                        }});
                        
                        if (response.ok) {{
                            location.reload();
                        }} else {{
                            alert('エラーが発生しました: ' + await response.text());
                        }}
                    }} catch (error) {{
                        alert('エラーが発生しました: ' + error.message);
                    }}
                }}
                
                function editDirectory(dirId, directoryPath, scanInterval) {{
                    document.getElementById('edit_id').value = dirId;
                    document.getElementById('edit_directory_path').value = directoryPath;
                    document.getElementById('edit_scan_interval').value = scanInterval;
                    document.getElementById('editModal').style.display = 'block';
                }}
                
                function closeEditModal() {{
                    document.getElementById('editModal').style.display = 'none';
                }}
                
                async function updateDirectory(event) {{
                    event.preventDefault();
                    const formData = new FormData(event.target);
                    const dirId = formData.get('id');
                    const data = {{
                        scan_interval_minutes: parseInt(formData.get('scan_interval')),
                        enabled: formData.get('enabled') === 'on'
                    }};
                    
                    try {{
                        const response = await fetch(`/task/watched_directories/${{dirId}}`, {{
                            method: 'PATCH',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify(data)
                        }});
                        
                        if (response.ok) {{
                            closeEditModal();
                            location.reload();
                        }} else {{
                            alert('エラーが発生しました: ' + await response.text());
                        }}
                    }} catch (error) {{
                        alert('エラーが発生しました: ' + error.message);
                    }}
                }}
                
                async function deleteDirectory(dirId) {{
                    if (!confirm('この監視対象を削除しますか？')) {{
                        return;
                    }}
                    
                    try {{
                        const response = await fetch(`/task/watched_directories/${{dirId}}`, {{
                            method: 'DELETE'
                        }});
                        
                        if (response.ok) {{
                            location.reload();
                        }} else {{
                            alert('エラーが発生しました: ' + await response.text());
                        }}
                    }} catch (error) {{
                        alert('エラーが発生しました: ' + error.message);
                    }}
                }}
                
                async function vectorizeDirectory(directoryPath) {{
                    // 強制再ベクトル化の確認
                    const forceRevectorize = confirm(
                        `ディレクトリ「${{directoryPath}}」をベクトル化しますか？\\n\\n` +
                        `「OK」: 強制再ベクトル化（全ファイルを再処理）\\n` +
                        `「キャンセル」: 通常ベクトル化（更新がないファイルはスキップ）`
                    );
                    
                    if (!forceRevectorize && !confirm(`通常ベクトル化を実行しますか？\\n更新がないファイルは自動的にスキップされます。`)) {{
                        return;
                    }}
                    
                    try {{
                        const response = await fetch('/search/vectorize', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{
                                directory_path: directoryPath,
                                chunk_size: 1024,
                                chunk_overlap: 50,
                                force_revectorize: forceRevectorize
                            }})
                        }});
                        
                        if (response.ok) {{
                            const result = await response.json();
                            const mode = forceRevectorize ? '強制再ベクトル化' : '通常ベクトル化';
                            alert(`ベクトル化ジョブを開始しました（${{mode}}）。\\nジョブID: ${{result.job_id}}\\n\\n進捗は「ベクトル化ページ」で確認できます。`);
                            // ベクトル化ページにリダイレクト
                            window.location.href = '/task/create_vector';
                        }} else {{
                            const errorText = await response.text();
                            alert('エラーが発生しました: ' + errorText);
                        }}
                    }} catch (error) {{
                        alert('エラーが発生しました: ' + error.message);
                    }}
                }}
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        error_html = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <title>エラー - Obsidian MCP Server</title>
        </head>
        <body>
            <h1>エラーが発生しました</h1>
            <p>{str(e)}</p>
            <a href="/task/">タスク管理に戻る</a>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=500)


@task_router.get("/rag", response_class=HTMLResponse)
async def rag_page():
    """
    RAG質問応答ページ
    
    Returns:
        HTMLページ
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG質問応答 - Obsidian MCP Server</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 30px;
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
            }
            .back-link {
                display: inline-block;
                margin-bottom: 20px;
                color: #007bff;
                text-decoration: none;
                font-size: 14px;
            }
            .back-link:hover {
                text-decoration: underline;
            }
            .query-form {
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
            }
            .form-group {
                margin-bottom: 15px;
            }
            .form-group label {
                display: block;
                margin-bottom: 5px;
                font-weight: 600;
                color: #333;
            }
            .form-group input,
            .form-group textarea,
            .form-group select {
                width: 100%;
                padding: 10px;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-size: 14px;
                font-family: inherit;
            }
            .form-group textarea {
                min-height: 100px;
                resize: vertical;
            }
            .form-group input[type="number"],
            .form-group input[type="range"] {
                width: 150px;
            }
            .form-group input[type="checkbox"] {
                width: auto;
                margin-right: 5px;
            }
            .form-row {
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
            }
            .form-row .form-group {
                flex: 1;
                min-width: 200px;
            }
            .btn {
                padding: 12px 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: background-color 0.2s;
            }
            .btn-primary {
                background-color: #007bff;
                color: white;
            }
            .btn-primary:hover {
                background-color: #0056b3;
            }
            .btn-primary:disabled {
                background-color: #6c757d;
                cursor: not-allowed;
            }
            .result-container {
                margin-top: 30px;
                display: none;
            }
            .result-container.show {
                display: block;
            }
            .answer-section {
                background-color: #e7f3ff;
                border-left: 4px solid #007bff;
                padding: 20px;
                border-radius: 4px;
                margin-bottom: 20px;
            }
            .answer-section h2 {
                margin-top: 0;
                color: #007bff;
                font-size: 18px;
            }
            .answer-content {
                line-height: 1.8;
                color: #333;
            }
            .answer-content h1,
            .answer-content h2,
            .answer-content h3,
            .answer-content h4 {
                margin-top: 20px;
                margin-bottom: 10px;
                color: #333;
                font-weight: 600;
            }
            .answer-content h1 {
                font-size: 24px;
                border-bottom: 2px solid #dee2e6;
                padding-bottom: 8px;
            }
            .answer-content h2 {
                font-size: 20px;
                border-bottom: 1px solid #dee2e6;
                padding-bottom: 6px;
            }
            .answer-content h3 {
                font-size: 18px;
            }
            .answer-content h4 {
                font-size: 16px;
            }
            .answer-content p {
                margin-bottom: 12px;
            }
            .answer-content ul,
            .answer-content ol {
                margin-bottom: 12px;
                padding-left: 25px;
            }
            .answer-content li {
                margin-bottom: 6px;
            }
            .answer-content code {
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 13px;
            }
            .answer-content pre {
                background-color: #f4f4f4;
                padding: 15px;
                border-radius: 4px;
                overflow-x: auto;
                margin-bottom: 12px;
            }
            .answer-content pre code {
                background-color: transparent;
                padding: 0;
            }
            .answer-content blockquote {
                border-left: 4px solid #007bff;
                padding-left: 15px;
                margin-left: 0;
                margin-bottom: 12px;
                color: #666;
                font-style: italic;
            }
            .answer-content strong {
                font-weight: 600;
                color: #333;
            }
            .answer-content em {
                font-style: italic;
            }
            .answer-content table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 12px;
            }
            .answer-content table th,
            .answer-content table td {
                border: 1px solid #dee2e6;
                padding: 8px;
                text-align: left;
            }
            .answer-content table th {
                background-color: #f8f9fa;
                font-weight: 600;
            }
            .answer-content hr {
                border: none;
                border-top: 1px solid #dee2e6;
                margin: 20px 0;
            }
            .sources-section {
                margin-top: 20px;
            }
            .sources-section h3 {
                color: #333;
                font-size: 16px;
                margin-bottom: 15px;
            }
            .source-item {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 15px;
                margin-bottom: 10px;
            }
            .source-item-header {
                font-weight: 600;
                color: #007bff;
                margin-bottom: 8px;
                font-size: 14px;
            }
            .source-item-path {
                font-family: 'Courier New', monospace;
                font-size: 12px;
                color: #666;
                margin-bottom: 5px;
            }
            .source-item-snippet {
                font-size: 13px;
                color: #333;
                line-height: 1.5;
                margin-top: 8px;
                padding: 10px;
                background-color: white;
                border-radius: 4px;
            }
            .loading {
                text-align: center;
                padding: 40px;
                color: #666;
            }
            .loading-spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #007bff;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .error {
                background-color: #f8d7da;
                border-left: 4px solid #dc3545;
                color: #721c24;
                padding: 15px;
                border-radius: 4px;
                margin-top: 20px;
            }
            .model-info {
                font-size: 12px;
                color: #666;
                margin-top: 10px;
                padding-top: 10px;
                border-top: 1px solid #dee2e6;
            }
            .advanced-options {
                margin-top: 15px;
                padding: 15px;
                background-color: white;
                border-radius: 4px;
                border: 1px solid #dee2e6;
            }
            .advanced-options summary {
                cursor: pointer;
                font-weight: 600;
                color: #007bff;
                margin-bottom: 10px;
            }
            .advanced-options summary:hover {
                color: #0056b3;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/task/" class="back-link">← タスク管理に戻る</a>
            <h1>RAG質問応答</h1>
            
            <form id="ragForm" class="query-form" onsubmit="submitRAGQuery(event)">
                <div class="form-group">
                    <label for="query">質問:</label>
                    <textarea id="query" name="query" required placeholder="知識ベースに対して質問を入力してください..."></textarea>
                </div>
                
                <details class="advanced-options">
                    <summary>詳細設定</summary>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="limit">検索結果数:</label>
                            <input type="number" id="limit" name="limit" value="20" min="1" max="100">
                        </div>
                        <div class="form-group">
                            <label for="hybrid_weight">ベクトル検索の重み:</label>
                            <input type="range" id="hybrid_weight" name="hybrid_weight" min="0" max="1" step="0.1" value="0.5">
                            <span id="hybrid_weight_value">0.5</span>
                        </div>
                        <div class="form-group">
                            <label for="temperature">温度パラメータ:</label>
                            <input type="range" id="temperature" name="temperature" min="0" max="2" step="0.1" value="0.7">
                            <span id="temperature_value">0.7</span>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>
                                <input type="checkbox" id="expand_synonyms" name="expand_synonyms">
                                類義語展開を使用
                            </label>
                        </div>
                    </div>
                </details>
                
                <button type="submit" class="btn btn-primary" id="submitBtn">質問を送信</button>
            </form>
            
            <div id="loading" class="loading" style="display: none;">
                <div class="loading-spinner"></div>
                <p>回答を生成中...</p>
            </div>
            
            <div id="resultContainer" class="result-container">
                <div class="answer-section">
                    <h2>回答</h2>
                    <div id="answerContent" class="answer-content"></div>
                    <div id="modelInfo" class="model-info"></div>
                </div>
                
                <div class="sources-section">
                    <h3>参照元</h3>
                    <div id="sourcesList"></div>
                </div>
            </div>
            
            <div id="errorContainer" class="error" style="display: none;"></div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <script>
            // スライダーの値を表示
            document.getElementById('hybrid_weight').addEventListener('input', function(e) {
                document.getElementById('hybrid_weight_value').textContent = e.target.value;
            });
            
            document.getElementById('temperature').addEventListener('input', function(e) {
                document.getElementById('temperature_value').textContent = e.target.value;
            });
            
            // MarkdownをHTMLに変換する関数
            function markdownToHtml(markdown) {
                if (typeof marked !== 'undefined') {
                    // marked.jsを使用
                    marked.setOptions({
                        breaks: true,
                        gfm: true
                    });
                    return marked.parse(markdown);
                } else {
                    // フォールバック: シンプルなMarkdownパーサー
                    return simpleMarkdownParser(markdown);
                }
            }
            
            // シンプルなMarkdownパーサー（フォールバック）
            function simpleMarkdownParser(markdown) {
                let html = markdown;
                
                // 見出し
                html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
                html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
                html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
                
                // 太字
                html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');
                
                // 斜体
                html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
                html = html.replace(/_(.*?)_/g, '<em>$1</em>');
                
                // コードブロック
                html = html.replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>');
                html = html.replace(/`(.*?)`/g, '<code>$1</code>');
                
                // リスト
                html = html.replace(/^\\* (.*$)/gim, '<li>$1</li>');
                html = html.replace(/^- (.*$)/gim, '<li>$1</li>');
                html = html.replace(/^\\d+\\. (.*$)/gim, '<li>$1</li>');
                
                // 段落
                html = html.replace(/\\n\\n/g, '</p><p>');
                html = '<p>' + html + '</p>';
                
                // リストのラップ
                html = html.replace(/(<li>.*?<\\/li>)/g, '<ul>$1</ul>');
                
                return html;
            }
            
            async function submitRAGQuery(event) {
                event.preventDefault();
                
                const form = event.target;
                const submitBtn = document.getElementById('submitBtn');
                const loading = document.getElementById('loading');
                const resultContainer = document.getElementById('resultContainer');
                const errorContainer = document.getElementById('errorContainer');
                
                // UIをリセット
                submitBtn.disabled = true;
                loading.style.display = 'block';
                resultContainer.classList.remove('show');
                errorContainer.style.display = 'none';
                
                // フォームデータを取得
                const formData = new FormData(form);
                const data = {
                    query: formData.get('query'),
                    limit: parseInt(formData.get('limit') || '20'),
                    hybrid_weight: parseFloat(formData.get('hybrid_weight') || '0.5'),
                    keyword_limit: 10,
                    vector_limit: 20,
                    expand_synonyms: formData.get('expand_synonyms') === 'on',
                    temperature: parseFloat(formData.get('temperature') || '0.7')
                };
                
                try {
                    const response = await fetch('/search/rag', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    
                    if (!response.ok) {
                        const errorText = await response.text();
                        throw new Error(errorText);
                    }
                    
                    const result = await response.json();
                    
                    // 回答をMarkdownとして表示
                    const answerContent = document.getElementById('answerContent');
                    answerContent.innerHTML = markdownToHtml(result.answer);
                    
                    document.getElementById('modelInfo').textContent = 
                        `使用モデル: ${result.model_used} (プロバイダー: ${result.provider_used})`;
                    
                    // ソースを表示
                    const sourcesList = document.getElementById('sourcesList');
                    sourcesList.innerHTML = '';
                    
                    if (result.sources && result.sources.length > 0) {
                        result.sources.forEach((source, index) => {
                            const sourceItem = document.createElement('div');
                            sourceItem.className = 'source-item';
                            sourceItem.innerHTML = `
                                <div class="source-item-header">資料 ${index + 1}</div>
                                <div class="source-item-path">${source.file_path}${source.location_info ? ' (' + source.location_info + ')' : ''}</div>
                                <div class="source-item-snippet">${source.snippet || ''}</div>
                            `;
                            sourcesList.appendChild(sourceItem);
                        });
                    } else {
                        sourcesList.innerHTML = '<p>参照元がありません</p>';
                    }
                    
                    // 結果を表示
                    resultContainer.classList.add('show');
                    
                } catch (error) {
                    errorContainer.textContent = 'エラーが発生しました: ' + error.message;
                    errorContainer.style.display = 'block';
                } finally {
                    submitBtn.disabled = false;
                    loading.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content, status_code=200)


# APIエンドポイント
@task_router.post("/watched_directories")
async def add_watched_directory(request: WatchedDirectoryRequest):
    """監視対象ディレクトリを追加"""
    try:
        dir_id = db.add_watched_directory(
            directory_path=request.directory_path,
            scan_interval_minutes=request.scan_interval_minutes,
            enabled=request.enabled
        )
        # 監視スケジューラーを開始（まだ開始されていない場合）
        start_watcher()
        return {"id": dir_id, "message": "監視対象ディレクトリを追加しました"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@task_router.get("/watched_directories")
async def get_watched_directories():
    """監視対象ディレクトリ一覧を取得"""
    try:
        dirs = db.get_watched_directories()
        return {"directories": dirs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@task_router.patch("/watched_directories/{dir_id}")
async def update_watched_directory(dir_id: int, request: WatchedDirectoryUpdateRequest):
    """監視対象ディレクトリの設定を更新"""
    try:
        success = db.update_watched_directory(
            dir_id=dir_id,
            scan_interval_minutes=request.scan_interval_minutes,
            enabled=request.enabled
        )
        if not success:
            raise HTTPException(status_code=404, detail="監視対象ディレクトリが見つかりません")
        return {"message": "監視対象ディレクトリを更新しました"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@task_router.delete("/watched_directories/{dir_id}")
async def delete_watched_directory(dir_id: int):
    """監視対象ディレクトリを削除"""
    try:
        success = db.delete_watched_directory(dir_id)
        if not success:
            raise HTTPException(status_code=404, detail="監視対象ディレクトリが見つかりません")
        return {"message": "監視対象ディレクトリを削除しました"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def process_vectorize_job(
    job_id: int,
    directory_path: str,
    provider_type_str: Optional[str] = None,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    aws_region_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    chunk_size: int = 1024,
    chunk_overlap: int = 50,
    force_revectorize: bool = False
):
    """
    ベクトル化ジョブをバックグラウンドで実行
    
    Args:
        job_id: ジョブID
        directory_path: ベクトル化するディレクトリのパス
        provider_type_str: Embeddingプロバイダーのタイプ
        model: 使用するモデル名またはARN（オプション）
        api_base: LiteLLMのカスタムエンドポイントURL（litellmプロバイダーの場合のみ）
        aws_region_name: AWSリージョン名（aws_bedrockプロバイダーの場合のみ）
        aws_access_key_id: AWSアクセスキーID（aws_bedrockプロバイダーの場合のみ）
        aws_secret_access_key: AWSシークレットキー（aws_bedrockプロバイダーの場合のみ）
        chunk_size: チャンクサイズ
        chunk_overlap: オーバーラップサイズ
    """
    try:
        # ジョブステータスを処理中に更新
        db.job_queue.update_job_status(job_id, JobStatus.PROCESSING)
        
        # ディレクトリの存在確認
        if not os.path.exists(directory_path):
            error_msg = f"ディレクトリが存在しません: {directory_path}"
            db.job_queue.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
            print(f"ベクトル化ジョブ {job_id} が失敗しました: {error_msg}")
            return
        
        if not os.path.isdir(directory_path):
            error_msg = f"指定されたパスはディレクトリではありません: {directory_path}"
            db.job_queue.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
            print(f"ベクトル化ジョブ {job_id} が失敗しました: {error_msg}")
            return
        
        # データベース内のドキュメント数を確認
        total_docs_in_db = db.get_document_count()
        db.job_queue.update_job_progress(
            job_id, 0, 0, 
            f"ディレクトリを確認中... (データベース内の総ドキュメント数: {total_docs_in_db})"
        )
        
        # 指定ディレクトリのドキュメントを確認
        documents = db.get_documents_by_directory(directory_path)
        if not documents:
            # データベース内の実際のパスのサンプルを取得（デバッグ用）
            cursor = db.conn.cursor()
            cursor.execute("SELECT DISTINCT file_path FROM documents LIMIT 5")
            sample_paths = [row['file_path'] for row in cursor.fetchall()]
            
            # より詳細なエラーメッセージを提供
            error_msg = (
                f"データベースにドキュメントが見つかりませんでした。\n\n"
                f"検索したディレクトリパス: {directory_path}\n"
                f"正規化後のパス: {os.path.abspath(directory_path)}\n"
                f"データベース内の総ドキュメント数: {total_docs_in_db}\n\n"
                f"データベース内のパスの例（最初の5件）:\n"
                + "\n".join([f"  - {path}" for path in sample_paths])
                + "\n\n"
                f"※ パスが一致していない可能性があります。\n"
                f"   - インデックス作成時に使用したディレクトリパスと同じパスを使用してください。\n"
                f"   - または、親ディレクトリ（例: G:\\マイドライブ\\Obsidian\\MyVault\\MyVault）でベクトル化を実行してください。"
            )
            db.job_queue.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
            print(f"ベクトル化ジョブ {job_id} が失敗しました")
            print(f"検索パス: {directory_path}")
            print(f"正規化後: {os.path.abspath(directory_path)}")
            print(f"データベース内のサンプルパス:")
            for path in sample_paths:
                print(f"  - {path}")
            return
        
        db.job_queue.update_job_progress(
            job_id, 0, len(documents),
            f"ドキュメントを確認しました ({len(documents)}件)。ベクトル化を開始します..."
        )
        
        # Embeddingプロバイダーを決定
        if provider_type_str:
            try:
                provider_type = EmbeddingProviderType(provider_type_str)
            except ValueError:
                provider_type = EmbeddingProviderType.OPENROUTER
                print(f"警告: 無効なプロバイダータイプ '{provider_type_str}'。デフォルトのOpenRouterを使用します。")
        else:
            provider_type_str = os.environ.get("EMBEDDING_PROVIDER", "openrouter")
            try:
                provider_type = EmbeddingProviderType(provider_type_str)
            except ValueError:
                provider_type = EmbeddingProviderType.OPENROUTER
        
        # プロバイダーを作成
        try:
            # プロバイダー固有のパラメータを準備
            provider_kwargs = {}
            if model:
                # AWS Bedrockの場合はmodel_id、その他はmodel
                if provider_type == EmbeddingProviderType.AWS_BEDROCK:
                    provider_kwargs["model_id"] = model
                else:
                    provider_kwargs["model"] = model
            if provider_type == EmbeddingProviderType.LITELLM and api_base:
                provider_kwargs["api_base"] = api_base
            if provider_type == EmbeddingProviderType.AWS_BEDROCK:
                if aws_region_name:
                    provider_kwargs["region_name"] = aws_region_name
                if aws_access_key_id:
                    provider_kwargs["aws_access_key_id"] = aws_access_key_id
                if aws_secret_access_key:
                    provider_kwargs["aws_secret_access_key"] = aws_secret_access_key
            
            embedding_provider = create_embedding_provider(provider_type, **provider_kwargs)
        except ValueError as e:
            # APIキー関連のエラーの場合
            error_msg = (
                f"Embeddingプロバイダーの作成に失敗しました。\n\n"
                f"プロバイダー: {provider_type_str}\n"
                f"エラー: {str(e)}\n\n"
                f"※ APIキーが設定されていないか、無効です。\n"
                f"   環境変数 OPENROUTER_API_KEY を設定してください。"
            )
            db.job_queue.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
            print(f"ベクトル化ジョブ {job_id} が失敗しました: {error_msg}")
            return
        except Exception as e:
            error_msg = f"Embeddingプロバイダーの作成に失敗しました: {str(e)}\nプロバイダー: {provider_type_str}"
            db.job_queue.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
            print(f"ベクトル化ジョブ {job_id} が失敗しました: {error_msg}")
            import traceback
            print(traceback.format_exc())
            return
        
        # ベクトルストアを作成
        try:
            vector_store = VectorStore()
        except Exception as e:
            error_msg = f"ベクトルストアの作成に失敗しました: {str(e)}"
            db.job_queue.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
            print(f"ベクトル化ジョブ {job_id} が失敗しました: {error_msg}")
            import traceback
            print(traceback.format_exc())
            return
        
        # チャンカーを作成
        chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer
        )
        
        # ベクトライザーを作成
        vectorizer = DocumentVectorizer(
            db=db,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            chunker=chunker
        )
        
        # 進捗コールバック
        def progress_callback(current: int, total: int, message: str):
            db.job_queue.update_job_progress(job_id, current, total, message)
        
        # ベクトル化を実行
        result = vectorizer.vectorize_directory(
            directory_path=directory_path,
            batch_size=10,
            progress_callback=progress_callback,
            force_revectorize=force_revectorize
        )
        
        # 結果を確認
        if result.get("processed_files", 0) == 0 and result.get("total_files", 0) == 0:
            error_msg = (
                f"ベクトル化できるファイルがありませんでした。\n"
                f"ディレクトリパス: {directory_path}\n"
                f"結果: {result.get('message', '不明')}\n"
                f"※ データベースにインデックス化されたドキュメントが存在しないか、すべて空のコンテンツです。"
            )
            db.job_queue.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
            print(f"ベクトル化ジョブ {job_id} が失敗しました: {error_msg}")
            return
        
        # 完了メッセージにスキップ件数を含める
        skipped_files = result.get("skipped_files", 0)
        skipped_no_content = result.get("skipped_no_content", 0)
        skipped_not_updated = result.get("skipped_not_updated", 0)
        skipped_no_chunks = result.get("skipped_no_chunks", 0)
        processed_files = result.get("processed_files", 0)
        total_chunks = result.get("total_chunks", 0)
        total_files = result.get("total_files", 0)
        
        completion_message = (
            f"ベクトル化が完了しました。\n"
            f"総ファイル数: {total_files}件\n"
            f"処理済み: {processed_files}件\n"
        )
        if skipped_files > 0:
            completion_message += f"スキップ: {skipped_files}件\n"
            if skipped_not_updated > 0:
                completion_message += f"  - 更新日時が変更されていない: {skipped_not_updated}件\n"
            if skipped_no_content > 0:
                completion_message += f"  - コンテンツが空: {skipped_no_content}件\n"
            if skipped_no_chunks > 0:
                completion_message += f"  - チャンクが生成されなかった: {skipped_no_chunks}件\n"
        completion_message += f"チャンク数: {total_chunks}件"
        
        # スキップされたファイルの詳細をログに出力（最初の10件）
        skipped_file_details = result.get("skipped_file_details", [])
        if skipped_file_details:
            print(f"\nスキップされたファイルの詳細（最初の10件）:")
            for detail in skipped_file_details[:10]:
                print(f"  - {detail['file_path']}: {detail['reason']}")
            if len(skipped_file_details) > 10:
                print(f"  ... 他 {len(skipped_file_details) - 10}件")
        
        # 完了
        db.job_queue.update_job_status(job_id, JobStatus.COMPLETED)
        db.job_queue.update_job_result(job_id, result)
        print(f"ベクトル化ジョブ {job_id} が完了しました: {completion_message}")
        
    except Exception as e:
        # エラーを記録
        import traceback
        error_details = traceback.format_exc()
        error_msg = f"{str(e)}\n\n詳細:\n{error_details}"
        db.job_queue.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
        print(f"ベクトル化ジョブ {job_id} が失敗しました: {error_msg}")


@router.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_directory(request: VectorizeRequest):
    """
    指定されたディレクトリ内のドキュメントをベクトル化してChromaDBに保存（バックグラウンド処理）
    
    Args:
        request: ベクトル化リクエスト
    
    Returns:
        ジョブIDを含むレスポンス
    """
    try:
        directory_path = os.path.abspath(request.directory_path)
        
        # ジョブを作成
        job_id = db.job_queue.create_job(
            job_type=JobType.VECTORIZE.value,
            parameters={
                "directory_path": directory_path,
                "provider": request.provider,
                "model": request.model,
                "api_base": request.api_base,
                "aws_region_name": request.aws_region_name,
                "aws_access_key_id": request.aws_access_key_id,
                "aws_secret_access_key": request.aws_secret_access_key,
                "chunk_size": request.chunk_size,
                "chunk_overlap": request.chunk_overlap,
                "force_revectorize": request.force_revectorize
            },
            initial_progress={
                "current": 0,
                "total": 0,
                "percentage": 0,
                "message": "ジョブを開始しています..."
            }
        )
        
        # バックグラウンドスレッドで処理を開始
        thread = threading.Thread(
            target=process_vectorize_job,
            args=(
                job_id,
                directory_path,
                request.provider,
                request.model,
                request.api_base,
                request.aws_region_name,
                request.aws_access_key_id,
                request.aws_secret_access_key,
                request.chunk_size,
                request.chunk_overlap,
                request.force_revectorize
            ),
            daemon=True
        )
        thread.start()
        
        return VectorizeResponse(
            message="ベクトル化ジョブを開始しました",
            job_id=job_id,
            directory_path=directory_path
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ベクトル化ジョブの作成中にエラーが発生しました: {str(e)}")


@router.get("/vectorize/stats")
async def get_vectorize_stats():
    """
    ベクトルストアの統計情報を取得
    
    Returns:
        統計情報（チャンク数など）
    """
    try:
        vector_store = VectorStore()
        stats = vector_store.get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"統計情報の取得中にエラーが発生しました: {str(e)}")


@router.get("/db/stats")
async def get_db_stats():
    """
    データベースのサイズと統計情報を取得
    
    Returns:
        データベースサイズ情報（SQLite、ChromaDB、テーブル統計など）
    """
    try:
        import os
        from pathlib import Path
        
        # SQLiteデータベースのサイズ
        sqlite_path = Path(db.db_path)
        sqlite_size = 0
        if sqlite_path.exists():
            sqlite_size = sqlite_path.stat().st_size
        
        # SQLiteテーブル統計
        cursor = db.conn.cursor()
        
        # documentsテーブルの統計
        cursor.execute("SELECT COUNT(*) as count FROM documents")
        doc_count = cursor.fetchone()['count']
        
        # code_filesテーブルの統計
        cursor.execute("SELECT COUNT(*) as count FROM code_files")
        code_file_count = cursor.fetchone()['count']
        
        # code_indicesテーブルの統計
        cursor.execute("SELECT COUNT(*) as count FROM code_indices")
        code_index_count = cursor.fetchone()['count']
        
        # job_queueテーブルの統計
        cursor.execute("SELECT COUNT(*) as count FROM job_queue")
        job_count = cursor.fetchone()['count']
        
        # watched_directoriesテーブルの統計
        cursor.execute("SELECT COUNT(*) as count FROM watched_directories")
        watched_dir_count = cursor.fetchone()['count']
        
        # SQLiteデータベースのページサイズとページ数を取得（より正確なサイズ計算）
        try:
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            sqlite_actual_size = page_count * page_size
        except:
            sqlite_actual_size = sqlite_size
        
        # ChromaDBディレクトリのサイズ
        chroma_path = Path("./chroma_db")
        chroma_size = 0
        chroma_file_count = 0
        if chroma_path.exists():
            for file_path in chroma_path.rglob("*"):
                if file_path.is_file():
                    chroma_size += file_path.stat().st_size
                    chroma_file_count += 1
        
        # ChromaDBコレクション統計
        vector_store = VectorStore()
        vector_stats = vector_store.get_collection_stats()
        
        # サイズを読みやすい形式に変換
        def format_size(size_bytes):
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.2f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.2f} PB"
        
        return {
            "sqlite": {
                "path": str(sqlite_path.absolute()),
                "size_bytes": sqlite_size,
                "size_formatted": format_size(sqlite_size),
                "actual_size_bytes": sqlite_actual_size,
                "actual_size_formatted": format_size(sqlite_actual_size),
                "tables": {
                    "documents": doc_count,
                    "code_files": code_file_count,
                    "code_indices": code_index_count,
                    "job_queue": job_count,
                    "watched_directories": watched_dir_count
                }
            },
            "chromadb": {
                "path": str(chroma_path.absolute()),
                "size_bytes": chroma_size,
                "size_formatted": format_size(chroma_size),
                "file_count": chroma_file_count,
                "collection_stats": vector_stats
            },
            "total": {
                "size_bytes": sqlite_size + chroma_size,
                "size_formatted": format_size(sqlite_size + chroma_size)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"データベース統計情報の取得中にエラーが発生しました: {str(e)}")


@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(request: HybridSearchRequest):
    """
    ハイブリッド検索を実行（全文検索 + ベクトル検索 + RRF）
    
    処理フロー:
    1. クエリの前処理（形態素解析、ストップワード除去、類義語展開）
    2. 全文検索とベクトル検索を並行実行
    3. RRF（Reciprocal Rank Fusion）で統合
    4. 上位結果を返却
    
    Args:
        request: ハイブリッド検索リクエスト
    
    Returns:
        検索結果（RRFスコア順にソート済み）
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="検索キーワードが指定されていません")
    
    try:
        # 1. クエリの前処理
        # 形態素解析とキーワード抽出
        keywords = tokenizer.extract_keywords(request.query)
        
        if not keywords:
            # キーワードが抽出できない場合は元のクエリを使用
            keywords = [request.query.strip()]
        
        # 類義語展開（オプション）
        if request.expand_synonyms:
            query_expander = QueryExpansion()
            keywords = query_expander.expand(keywords, use_llm=False)
        
        # 2. 全文検索を実行（各キーワードごとに検索）
        keyword_results = db.search_by_keywords(
            keywords=keywords,
            limit_per_keyword=request.keyword_limit,
            max_total=50  # マージ後の最大件数
        )
        
        # 3. ベクトル検索を実行
        vector_results = []
        
        try:
            # ベクトライザーとベクトルストアを取得
            vectorizer_instance = get_vectorizer()
            vector_store = VectorStore()
            
            # クエリをベクトル化
            query_embedding = vectorizer_instance.embedding_provider.get_embedding(request.query)
            
            # ベクトル検索を実行
            vector_search_results = vector_store.search(
                query_embedding=query_embedding,
                n_results=request.vector_limit
            )
            
            # ベクトル検索結果を統一フォーマットに変換
            for rank, result in enumerate(vector_search_results, start=1):
                metadata = result.get('metadata', {})
                doc_id = normalize_doc_id(
                    file_path=metadata.get('file_path', ''),
                    location_info=metadata.get('location_info')
                )
                
                # スニペットを生成（ドキュメント本文から）
                document_text = result.get('document', '')
                snippet = document_text[:200] + '...' if len(document_text) > 200 else document_text
                
                vector_results.append({
                    'doc_id': doc_id,
                    'file_path': metadata.get('file_path', ''),
                    'file_type': metadata.get('file_name', '').split('.')[-1] if metadata.get('file_name') else None,
                    'location_info': metadata.get('location_info'),
                    'snippet': snippet,
                    'rank': rank
                })
        except Exception as e:
            # ベクトル検索が失敗した場合は、全文検索のみで続行
            print(f"ベクトル検索が失敗しました: {str(e)}。全文検索のみで続行します。")
        
        # 4. RRFで統合
        fused_results = reciprocal_rank_fusion(
            keyword_results=keyword_results,
            vector_results=vector_results,
            k=60,  # RRFの定数
            alpha=request.hybrid_weight,
            max_results=request.limit
        )
        
        # 5. レスポンスモデルに変換
        search_results = [
            SearchResult(
                file_path=result.file_path,
                file_type=result.file_type,
                location_info=result.location_info,
                snippet=result.snippet
            )
            for result in fused_results
        ]
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total=len(search_results)
        )
    
    except Exception as e:
        import traceback
        error_detail = f"ハイブリッド検索中にエラーが発生しました: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@router.get("/hybrid", response_model=SearchResponse)
async def hybrid_search_get(
    query: str,
    limit: int = 20,
    hybrid_weight: float = 0.5,
    keyword_limit: int = 10,
    vector_limit: int = 20,
    expand_synonyms: bool = False
):
    """
    ハイブリッド検索を実行（GET版）
    
    Args:
        query: 検索キーワード
        limit: 返却する結果の最大数
        hybrid_weight: ベクトル検索の重み（0.0-1.0）
        keyword_limit: 各キーワードあたりの全文検索取得件数
        vector_limit: ベクトル検索の取得件数
        expand_synonyms: 類義語展開を使用するかどうか
    
    Returns:
        検索結果
    """
    request = HybridSearchRequest(
        query=query,
        limit=limit,
        hybrid_weight=hybrid_weight,
        keyword_limit=keyword_limit,
        vector_limit=vector_limit,
        expand_synonyms=expand_synonyms
    )
    return await hybrid_search(request)


@router.get("/llm/models")
async def get_llm_models(api_base: Optional[str] = None):
    """
    LiteLLMの利用可能なモデルリストを取得
    
    Obsidianから問い合わせを受けた際に、アクセスポイント（api_base）を指定してもらい、
    そこへ /models でアクセスしてモデルリストを取得します。
    
    Args:
        api_base: LiteLLMのカスタムエンドポイントURL（必須）
                例: "http://localhost:4000" または "https://api.example.com/v1"
    
    Returns:
        利用可能なモデルのリスト
    """
    try:
        import requests
        
        # エンドポイントURLを決定
        if api_base:
            endpoint = api_base.rstrip('/')
            # /models パスを追加
            if not endpoint.endswith('/models'):
                endpoint = endpoint + '/models'
        else:
            # 環境変数から取得を試みる
            api_base_env = os.environ.get("LITELLM_API_BASE")
            if api_base_env:
                endpoint = api_base_env.rstrip('/') + '/models'
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "api_baseパラメータが必要です。\n"
                        "例: GET /search/llm/models?api_base=http://localhost:4000"
                    )
                )
        
        # モデルリストを取得
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        
        models_data = response.json()
        
        # レスポンス形式を統一（OpenAI互換形式を想定）
        if isinstance(models_data, dict) and "data" in models_data:
            models = models_data["data"]
        elif isinstance(models_data, list):
            models = models_data
        else:
            models = []
        
        # モデル情報を整形
        formatted_models = []
        for model in models:
            if isinstance(model, dict):
                formatted_models.append({
                    "id": model.get("id", ""),
                    "name": model.get("name", model.get("id", "")),
                    "object": model.get("object", "model"),
                    "created": model.get("created"),
                    "owned_by": model.get("owned_by", ""),
                    "permission": model.get("permission", [])
                })
            elif isinstance(model, str):
                formatted_models.append({
                    "id": model,
                    "name": model,
                    "object": "model"
                })
        
        return {
            "api_base": endpoint.replace('/models', ''),
            "models": formatted_models,
            "total": len(formatted_models)
        }
        
    except requests.exceptions.RequestException as e:
        error_detail = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.text
            except:
                pass
        raise HTTPException(
            status_code=500,
            detail=f"モデルリストの取得に失敗しました: {error_detail}\nエンドポイント: {endpoint if 'endpoint' in locals() else '不明'}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"エラーが発生しました: {str(e)}"
        )


# LLM設定関連のAPIエンドポイント
@task_router.get("/llm-settings/", response_class=HTMLResponse)
async def llm_settings_page():
    """
    LLM設定ページ
    
    Returns:
        HTMLページ
    """
    try:
        # 現在の設定を取得
        llm_setting = db.get_llm_setting("rag")
        embedding_setting = db.get_embedding_setting()
        
        # ベクトルストアの統計情報を取得
        try:
            from .vector_store import VectorStore
            vector_store = VectorStore()
            vector_stats = vector_store.get_collection_stats()
        except:
            vector_stats = {"total_chunks": 0, "total_files": 0}
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>LLM設定 - Obsidian MCP Server</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 30px;
                }}
                h1 {{
                    color: #333;
                    margin-bottom: 10px;
                }}
                .back-link {{
                    display: inline-block;
                    margin-bottom: 20px;
                    color: #007bff;
                    text-decoration: none;
                    font-size: 14px;
                }}
                .back-link:hover {{
                    text-decoration: underline;
                }}
                .settings-section {{
                    margin-bottom: 40px;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                }}
                .settings-section h2 {{
                    color: #333;
                    margin-top: 0;
                    margin-bottom: 20px;
                    font-size: 20px;
                }}
                .form-group {{
                    margin-bottom: 15px;
                }}
                .form-group label {{
                    display: block;
                    margin-bottom: 5px;
                    font-weight: 600;
                    color: #333;
                }}
                .form-group input,
                .form-group select {{
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    font-size: 14px;
                    font-family: inherit;
                }}
                .form-group input[type="checkbox"] {{
                    width: auto;
                    margin-right: 5px;
                }}
                .form-row {{
                    display: flex;
                    gap: 15px;
                    flex-wrap: wrap;
                }}
                .form-row .form-group {{
                    flex: 1;
                    min-width: 200px;
                }}
                .btn {{
                    padding: 12px 24px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 16px;
                    font-weight: 600;
                    transition: background-color 0.2s;
                    margin-right: 10px;
                }}
                .btn-primary {{
                    background-color: #007bff;
                    color: white;
                }}
                .btn-primary:hover {{
                    background-color: #0056b3;
                }}
                .btn-secondary {{
                    background-color: #6c757d;
                    color: white;
                }}
                .btn-secondary:hover {{
                    background-color: #5a6268;
                }}
                .btn-danger {{
                    background-color: #dc3545;
                    color: white;
                }}
                .btn-danger:hover {{
                    background-color: #c82333;
                }}
                .btn:disabled {{
                    background-color: #6c757d;
                    cursor: not-allowed;
                }}
                .alert {{
                    padding: 15px;
                    border-radius: 4px;
                    margin-bottom: 20px;
                }}
                .alert-warning {{
                    background-color: #fff3cd;
                    border-left: 4px solid #ffc107;
                    color: #856404;
                }}
                .alert-info {{
                    background-color: #d1ecf1;
                    border-left: 4px solid #17a2b8;
                    color: #0c5460;
                }}
                .alert-danger {{
                    background-color: #f8d7da;
                    border-left: 4px solid #dc3545;
                    color: #721c24;
                }}
                .alert-success {{
                    background-color: #d4edda;
                    border-left: 4px solid #28a745;
                    color: #155724;
                }}
                .status-badge {{
                    display: inline-block;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: 600;
                    margin-left: 10px;
                }}
                .status-badge.locked {{
                    background-color: #dc3545;
                    color: white;
                }}
                .status-badge.unlocked {{
                    background-color: #28a745;
                    color: white;
                }}
                .loading {{
                    text-align: center;
                    padding: 20px;
                    color: #666;
                }}
                .model-list {{
                    max-height: 200px;
                    overflow-y: auto;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    padding: 10px;
                    background-color: white;
                }}
                .model-item {{
                    padding: 8px;
                    cursor: pointer;
                    border-radius: 4px;
                    margin-bottom: 5px;
                }}
                .model-item:hover {{
                    background-color: #f8f9fa;
                }}
                .model-item.selected {{
                    background-color: #007bff;
                    color: white;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <a href="/task/" class="back-link">← タスク管理に戻る</a>
                <h1>LLM設定</h1>
                
                <div id="alertContainer"></div>
                
                <!-- LLM設定セクション -->
                <div class="settings-section">
                    <h2>LLM設定（RAG用）</h2>
                    <form id="llmForm" onsubmit="saveLLMSettings(event)">
                        <div class="form-group">
                            <label for="llm_provider">プロバイダー:</label>
                            <select id="llm_provider" name="llm_provider" required onchange="onLLMProviderChange()">
                                <option value="openrouter" {'selected' if llm_setting and llm_setting.get('provider') == 'openrouter' else ''}>OpenRouter</option>
                                <option value="litellm" {'selected' if llm_setting and llm_setting.get('provider') == 'litellm' else ''}>LiteLLM</option>
                                <option value="aws_bedrock" {'selected' if llm_setting and llm_setting.get('provider') == 'aws_bedrock' else ''}>AWS Bedrock</option>
                            </select>
                        </div>
                        <div class="form-group" id="llm_litellm_baseurl_group" style="display: {'block' if llm_setting and llm_setting.get('provider') == 'litellm' else 'none'};">
                            <label for="llm_litellm_baseurl">BaseURL:</label>
                            <input type="text" id="llm_litellm_baseurl" name="llm_litellm_baseurl" 
                                   placeholder="http://localhost:4000" 
                                   value="{llm_setting.get('api_base', '') if llm_setting else ''}">
                            <button type="button" class="btn btn-secondary" onclick="testLLMConnection()" style="margin-top: 10px;">接続確認</button>
                            <button type="button" class="btn btn-secondary" onclick="loadLLMModels()" style="margin-top: 10px;">モデル一覧を取得</button>
                        </div>
                        <div class="form-group" id="llm_aws_bedrock_group" style="display: {'block' if llm_setting and llm_setting.get('provider') == 'aws_bedrock' else 'none'};">
                            <div class="form-row">
                                <div class="form-group">
                                    <label for="llm_aws_region">AWSリージョン:</label>
                                    <input type="text" id="llm_aws_region" name="llm_aws_region" 
                                           placeholder="us-east-1" 
                                           value="{llm_setting.get('aws_region_name', '') if llm_setting else ''}">
                                </div>
                                <div class="form-group">
                                    <label for="llm_aws_access_key_id">AWSアクセスキーID:</label>
                                    <input type="text" id="llm_aws_access_key_id" name="llm_aws_access_key_id" 
                                           placeholder="環境変数から取得可能" 
                                           value="{llm_setting.get('aws_access_key_id', '') if llm_setting else ''}">
                                </div>
                                <div class="form-group">
                                    <label for="llm_aws_secret_access_key">AWSシークレットキー:</label>
                                    <input type="password" id="llm_aws_secret_access_key" name="llm_aws_secret_access_key" 
                                           placeholder="環境変数から取得可能" 
                                           value="{llm_setting.get('aws_secret_access_key', '') if llm_setting else ''}">
                                </div>
                            </div>
                        </div>
                        <div class="form-group">
                            <label for="llm_model">モデル:</label>
                            <input type="text" id="llm_model" name="llm_model" 
                                   placeholder="モデル名を入力または選択" 
                                   value="{llm_setting.get('model', '') if llm_setting else ''}">
                            <div id="llm_model_list" class="model-list" style="display: none; margin-top: 10px;"></div>
                        </div>
                        <button type="submit" class="btn btn-primary">LLM設定を保存</button>
                    </form>
                </div>
                
                <!-- Embedding設定セクション -->
                <div class="settings-section">
                    <h2>Embedding設定（ベクトル化用）</h2>
                    {'<div class="alert alert-warning">⚠️ ベクトルモデルはロックされています。変更する場合は既存のベクトルデータを削除してください。</div>' if embedding_setting and embedding_setting.get('is_locked') else ''}
                    {'<div class="alert alert-info">現在のベクトルデータ: {vector_stats.get("total_chunks", 0)} チャンク、{vector_stats.get("total_files", 0)} ファイル</div>' if vector_stats.get("total_chunks", 0) > 0 else ''}
                    <form id="embeddingForm" onsubmit="saveEmbeddingSettings(event)">
                        <div class="form-group">
                            <label for="embedding_provider">プロバイダー:</label>
                            <select id="embedding_provider" name="embedding_provider" required 
                                    onchange="onEmbeddingProviderChange()" 
                                    {'disabled' if embedding_setting and embedding_setting.get('is_locked') else ''}>
                                <option value="openrouter" {'selected' if embedding_setting and embedding_setting.get('provider') == 'openrouter' else ''}>OpenRouter</option>
                                <option value="litellm" {'selected' if embedding_setting and embedding_setting.get('provider') == 'litellm' else ''}>LiteLLM</option>
                                <option value="aws_bedrock" {'selected' if embedding_setting and embedding_setting.get('provider') == 'aws_bedrock' else ''}>AWS Bedrock</option>
                            </select>
                        </div>
                        <div class="form-group" id="embedding_litellm_baseurl_group" style="display: {'block' if embedding_setting and embedding_setting.get('provider') == 'litellm' else 'none'};">
                            <label for="embedding_litellm_baseurl">BaseURL:</label>
                            <input type="text" id="embedding_litellm_baseurl" name="embedding_litellm_baseurl" 
                                   placeholder="http://localhost:4000" 
                                   value="{embedding_setting.get('api_base', '') if embedding_setting else ''}"
                                   {'disabled' if embedding_setting and embedding_setting.get('is_locked') else ''}>
                            <button type="button" class="btn btn-secondary" onclick="testEmbeddingConnection()" style="margin-top: 10px;" {'disabled' if embedding_setting and embedding_setting.get('is_locked') else ''}>接続確認</button>
                            <button type="button" class="btn btn-secondary" onclick="loadEmbeddingModels()" style="margin-top: 10px;" {'disabled' if embedding_setting and embedding_setting.get('is_locked') else ''}>モデル一覧を取得</button>
                        </div>
                        <div class="form-group" id="embedding_aws_bedrock_group" style="display: {'block' if embedding_setting and embedding_setting.get('provider') == 'aws_bedrock' else 'none'};">
                            <div class="form-row">
                                <div class="form-group">
                                    <label for="embedding_aws_region">AWSリージョン:</label>
                                    <input type="text" id="embedding_aws_region" name="embedding_aws_region" 
                                           placeholder="us-east-1" 
                                           value="{embedding_setting.get('aws_region_name', '') if embedding_setting else ''}"
                                           {'disabled' if embedding_setting and embedding_setting.get('is_locked') else ''}>
                                </div>
                                <div class="form-group">
                                    <label for="embedding_aws_access_key_id">AWSアクセスキーID:</label>
                                    <input type="text" id="embedding_aws_access_key_id" name="embedding_aws_access_key_id" 
                                           placeholder="環境変数から取得可能" 
                                           value="{embedding_setting.get('aws_access_key_id', '') if embedding_setting else ''}"
                                           {'disabled' if embedding_setting and embedding_setting.get('is_locked') else ''}>
                                </div>
                                <div class="form-group">
                                    <label for="embedding_aws_secret_access_key">AWSシークレットキー:</label>
                                    <input type="password" id="embedding_aws_secret_access_key" name="embedding_aws_secret_access_key" 
                                           placeholder="環境変数から取得可能" 
                                           value="{embedding_setting.get('aws_secret_access_key', '') if embedding_setting else ''}"
                                           {'disabled' if embedding_setting and embedding_setting.get('is_locked') else ''}>
                                </div>
                            </div>
                        </div>
                        <div class="form-group">
                            <label for="embedding_model">モデル:</label>
                            <input type="text" id="embedding_model" name="embedding_model" 
                                   placeholder="モデル名を入力または選択" 
                                   value="{embedding_setting.get('model', '') if embedding_setting else ''}"
                                   {'disabled' if embedding_setting and embedding_setting.get('is_locked') else ''}>
                            <div id="embedding_model_list" class="model-list" style="display: none; margin-top: 10px;"></div>
                            {'<div class="alert alert-warning" style="margin-top: 10px;">⚠️ モデルを変更すると、既存のベクトルデータが使用できなくなります。次元サイズが異なる場合は再ベクトル化が必要です。</div>' if embedding_setting and not embedding_setting.get('is_locked') else ''}
                        </div>
                        <button type="submit" class="btn btn-primary" {'disabled' if embedding_setting and embedding_setting.get('is_locked') else ''}>Embedding設定を保存</button>
                    </form>
                </div>
            </div>
            
            <script>
                // LLMプロバイダー変更時の処理
                function onLLMProviderChange() {{
                    const provider = document.getElementById('llm_provider').value;
                    const baseurlGroup = document.getElementById('llm_litellm_baseurl_group');
                    const awsGroup = document.getElementById('llm_aws_bedrock_group');
                    baseurlGroup.style.display = provider === 'litellm' ? 'block' : 'none';
                    awsGroup.style.display = provider === 'aws_bedrock' ? 'block' : 'none';
                }}
                
                // Embeddingプロバイダー変更時の処理
                function onEmbeddingProviderChange() {{
                    const provider = document.getElementById('embedding_provider').value;
                    const baseurlGroup = document.getElementById('embedding_litellm_baseurl_group');
                    const awsGroup = document.getElementById('embedding_aws_bedrock_group');
                    baseurlGroup.style.display = provider === 'litellm' ? 'block' : 'none';
                    awsGroup.style.display = provider === 'aws_bedrock' ? 'block' : 'none';
                }}
                
                // アラート表示
                function showAlert(message, type = 'info') {{
                    const container = document.getElementById('alertContainer');
                    container.innerHTML = `<div class="alert alert-${{type}}">${{message}}</div>`;
                    setTimeout(() => {{
                        container.innerHTML = '';
                    }}, 5000);
                }}
                
                // LLM接続確認
                async function testLLMConnection() {{
                    const baseurl = document.getElementById('llm_litellm_baseurl').value;
                    if (!baseurl) {{
                        showAlert('BaseURLを入力してください', 'warning');
                        return;
                    }}
                    
                    try {{
                        const response = await fetch(`/search/llm/models?api_base=${{encodeURIComponent(baseurl)}}`);
                        if (response.ok) {{
                            showAlert('接続に成功しました', 'success');
                        }} else {{
                            const error = await response.text();
                            showAlert(`接続に失敗しました: ${{error}}`, 'danger');
                        }}
                    }} catch (error) {{
                        showAlert(`接続に失敗しました: ${{error.message}}`, 'danger');
                    }}
                }}
                
                // Embedding接続確認
                async function testEmbeddingConnection() {{
                    const baseurl = document.getElementById('embedding_litellm_baseurl').value;
                    if (!baseurl) {{
                        showAlert('BaseURLを入力してください', 'warning');
                        return;
                    }}
                    
                    try {{
                        const response = await fetch(`/search/llm/models?api_base=${{encodeURIComponent(baseurl)}}`);
                        if (response.ok) {{
                            showAlert('接続に成功しました', 'success');
                        }} else {{
                            const error = await response.text();
                            showAlert(`接続に失敗しました: ${{error}}`, 'danger');
                        }}
                    }} catch (error) {{
                        showAlert(`接続に失敗しました: ${{error.message}}`, 'danger');
                    }}
                }}
                
                // LLMモデル一覧を取得
                async function loadLLMModels() {{
                    const baseurl = document.getElementById('llm_litellm_baseurl').value;
                    if (!baseurl) {{
                        showAlert('BaseURLを入力してください', 'warning');
                        return;
                    }}
                    
                    try {{
                        const response = await fetch(`/search/llm/models?api_base=${{encodeURIComponent(baseurl)}}`);
                        if (response.ok) {{
                            const data = await response.json();
                            displayModelList('llm_model_list', 'llm_model', data.models);
                            showAlert(`${{data.total}}件のモデルを取得しました`, 'success');
                        }} else {{
                            const error = await response.text();
                            showAlert(`モデル一覧の取得に失敗しました: ${{error}}`, 'danger');
                        }}
                    }} catch (error) {{
                        showAlert(`モデル一覧の取得に失敗しました: ${{error.message}}`, 'danger');
                    }}
                }}
                
                // Embeddingモデル一覧を取得
                async function loadEmbeddingModels() {{
                    const baseurl = document.getElementById('embedding_litellm_baseurl').value;
                    if (!baseurl) {{
                        showAlert('BaseURLを入力してください', 'warning');
                        return;
                    }}
                    
                    try {{
                        const response = await fetch(`/search/llm/models?api_base=${{encodeURIComponent(baseurl)}}`);
                        if (response.ok) {{
                            const data = await response.json();
                            displayModelList('embedding_model_list', 'embedding_model', data.models);
                            showAlert(`${{data.total}}件のモデルを取得しました`, 'success');
                        }} else {{
                            const error = await response.text();
                            showAlert(`モデル一覧の取得に失敗しました: ${{error}}`, 'danger');
                        }}
                    }} catch (error) {{
                        showAlert(`モデル一覧の取得に失敗しました: ${{error.message}}`, 'danger');
                    }}
                }}
                
                // モデルリストを表示
                function displayModelList(listId, inputId, models) {{
                    const listDiv = document.getElementById(listId);
                    const input = document.getElementById(inputId);
                    
                    listDiv.innerHTML = '';
                    models.forEach(model => {{
                        const item = document.createElement('div');
                        item.className = 'model-item';
                        item.textContent = model.id || model.name || model;
                        item.onclick = () => {{
                            input.value = model.id || model.name || model;
                            listDiv.style.display = 'none';
                        }};
                        listDiv.appendChild(item);
                    }});
                    listDiv.style.display = 'block';
                }}
                
                // LLM設定を保存
                async function saveLLMSettings(event) {{
                    event.preventDefault();
                    
                    const formData = new FormData(event.target);
                    const provider = formData.get('llm_provider');
                    const data = {{
                        provider: provider,
                        model: formData.get('llm_model'),
                        api_base: provider === 'litellm' ? (formData.get('llm_litellm_baseurl') || null) : null,
                        aws_region_name: provider === 'aws_bedrock' ? (formData.get('llm_aws_region') || null) : null,
                        aws_access_key_id: provider === 'aws_bedrock' ? (formData.get('llm_aws_access_key_id') || null) : null,
                        aws_secret_access_key: provider === 'aws_bedrock' ? (formData.get('llm_aws_secret_access_key') || null) : null
                    }};
                    
                    try {{
                        const response = await fetch('/task/llm-settings/llm', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify(data)
                        }});
                        
                        if (response.ok) {{
                            showAlert('LLM設定を保存しました', 'success');
                        }} else {{
                            const error = await response.text();
                            showAlert(`保存に失敗しました: ${{error}}`, 'danger');
                        }}
                    }} catch (error) {{
                        showAlert(`保存に失敗しました: ${{error.message}}`, 'danger');
                    }}
                }}
                
                // Embedding設定を保存
                async function saveEmbeddingSettings(event) {{
                    event.preventDefault();
                    
                    const formData = new FormData(event.target);
                    const provider = formData.get('embedding_provider');
                    const data = {{
                        provider: provider,
                        model: formData.get('embedding_model'),
                        api_base: provider === 'litellm' ? (formData.get('embedding_litellm_baseurl') || null) : null,
                        aws_region_name: provider === 'aws_bedrock' ? (formData.get('embedding_aws_region') || null) : null,
                        aws_access_key_id: provider === 'aws_bedrock' ? (formData.get('embedding_aws_access_key_id') || null) : null,
                        aws_secret_access_key: provider === 'aws_bedrock' ? (formData.get('embedding_aws_secret_access_key') || null) : null
                    }};
                    
                    // 既存の設定がある場合は確認
                    {'if (true) {' if embedding_setting and embedding_setting.get('is_locked') else 'if (false) {'}
                        if (!confirm('⚠️ 警告: ベクトルモデルを変更すると、既存のベクトルデータが使用できなくなります。\\n\\n続行しますか？')) {{
                            return;
                        }}
                    }}
                    
                    try {{
                        const response = await fetch('/task/llm-settings/embedding', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify(data)
                        }});
                        
                        if (response.ok) {{
                            showAlert('Embedding設定を保存しました', 'success');
                            setTimeout(() => {{
                                location.reload();
                            }}, 1000);
                        }} else {{
                            const error = await response.text();
                            showAlert(`保存に失敗しました: ${{error}}`, 'danger');
                        }}
                    }} catch (error) {{
                        showAlert(`保存に失敗しました: ${{error.message}}`, 'danger');
                    }}
                }}
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        import traceback
        error_html = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <title>エラー - LLM設定</title>
        </head>
        <body>
            <h1>エラーが発生しました</h1>
            <p>{str(e)}</p>
            <pre>{traceback.format_exc()}</pre>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=500)


# LLM設定保存API
@task_router.post("/llm-settings/llm")
async def save_llm_settings(request: dict):
    """LLM設定を保存"""
    try:
        db.save_llm_setting(
            setting_type="rag",
            provider=request.get("provider", "openrouter"),
            model=request.get("model"),
            api_base=request.get("api_base"),
            aws_region_name=request.get("aws_region_name"),
            aws_access_key_id=request.get("aws_access_key_id"),
            aws_secret_access_key=request.get("aws_secret_access_key")
        )
        return {"message": "LLM設定を保存しました"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@task_router.post("/llm-settings/embedding")
async def save_embedding_settings(request: dict):
    """Embedding設定を保存"""
    try:
        provider = request.get("provider", "openrouter")
        model = request.get("model")
        api_base = request.get("api_base")
        aws_region_name = request.get("aws_region_name")
        aws_access_key_id = request.get("aws_access_key_id")
        aws_secret_access_key = request.get("aws_secret_access_key")
        
        if not model:
            raise HTTPException(status_code=400, detail="モデル名が指定されていません")
        
        # モデルの次元数を取得
        dimensions = 1536  # デフォルト
        if provider == "openrouter":
            # OpenRouterのモデル次元数マッピング
            dimensions_map = {
                "qwen/qwen3-embedding-8b": 4096,
            }
            dimensions = dimensions_map.get(model, 4096)
        elif provider == "litellm":
            # LiteLLMのモデル次元数マッピング
            dimensions_map = {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "gemini/text-embedding-004": 768,
                "voyage-large-2": 1536,
            }
            dimensions = dimensions_map.get(model, 1536)
        elif provider == "aws_bedrock":
            # AWS Bedrockのモデル次元数マッピング
            dimensions_map = {
                "amazon.titan-embed-text-v1": 1536,
                "amazon.titan-embed-text-v2": 1024,
            }
            dimensions = dimensions_map.get(model, 1536)
        
        # 既存の設定を確認
        existing = db.get_embedding_setting()
        is_locked = False
        
        if existing:
            # 既存の設定がある場合、ロック状態を確認
            if existing.get("is_locked", False):
                # ロックされている場合は、モデルが同じかチェック
                if existing.get("model") != model or existing.get("provider") != provider:
                    raise HTTPException(
                        status_code=400,
                        detail="Embedding設定はロックされています。変更する場合は既存のベクトルデータを削除してください。"
                    )
                is_locked = True
            else:
                # ロックされていない場合、モデルが変更されたら警告
                if existing.get("model") != model or existing.get("provider") != provider:
                    # ベクトルデータがある場合はロックする
                    try:
                        from .vector_store import VectorStore
                        vector_store = VectorStore()
                        stats = vector_store.get_collection_stats()
                        if stats.get("total_chunks", 0) > 0:
                            is_locked = True
                    except:
                        pass
        
        db.save_embedding_setting(
            provider=provider,
            model=model,
            api_base=api_base,
            dimensions=dimensions,
            is_locked=is_locked,
            aws_region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        # 初回保存時は自動的にロック
        if not existing:
            db.lock_embedding_setting()
        
        return {
            "message": "Embedding設定を保存しました",
            "dimensions": dimensions,
            "is_locked": is_locked
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    """
    RAG（Retrieval-Augmented Generation）による回答生成
    
    処理フロー:
    1. ハイブリッド検索を実行してコンテキストを取得
    2. 取得したコンテキストと質問を組み合わせてプロンプトを作成
    3. LLMで回答を生成
    
    Args:
        request: RAGリクエスト
    
    Returns:
        回答とソース情報
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="質問が指定されていません")
    
    try:
        # 1. ベクトルストアの利用可能性を確認
        vector_store_available = False
        try:
            vector_store = VectorStore()
            stats = vector_store.get_collection_stats()
            if stats.get("total_chunks", 0) > 0:
                vector_store_available = True
        except:
            pass
        
        # 2. ハイブリッド検索を実行してコンテキストを取得
        # ベクトルストアが利用できない場合は、全文検索のみで実行
        if not vector_store_available:
            # 全文検索のみで検索を実行
            keywords = tokenizer.extract_keywords(request.query)
            if not keywords:
                keywords = [request.query.strip()]
            
            # 類義語展開（オプション）
            if request.expand_synonyms:
                query_expander = QueryExpansion()
                keywords = query_expander.expand(keywords, use_llm=False)
            
            # 全文検索を実行
            keyword_results = db.search_by_keywords(
                keywords=keywords,
                limit_per_keyword=request.keyword_limit,
                max_total=request.limit
            )
            
            # SearchResult形式に変換
            search_results = [
                SearchResult(
                    file_path=result.get('file_path', ''),
                    file_type=result.get('file_type'),
                    location_info=result.get('location_info'),
                    snippet=result.get('snippet', '')
                )
                for result in keyword_results[:request.limit]
            ]
            
            search_response = SearchResponse(
                query=request.query,
                results=search_results,
                total=len(search_results)
            )
        else:
            # ベクトルストアが利用可能な場合はハイブリッド検索を実行
            hybrid_request = HybridSearchRequest(
                query=request.query,
                limit=request.limit,
                hybrid_weight=request.hybrid_weight,
                keyword_limit=request.keyword_limit,
                vector_limit=request.vector_limit,
                expand_synonyms=request.expand_synonyms
            )
            
            try:
                search_response = await hybrid_search(hybrid_request)
            except Exception as e:
                # ハイブリッド検索が失敗した場合、全文検索のみで再試行
                print(f"ハイブリッド検索が失敗しました: {str(e)}。全文検索のみで続行します。")
                
                keywords = tokenizer.extract_keywords(request.query)
                if not keywords:
                    keywords = [request.query.strip()]
                
                if request.expand_synonyms:
                    query_expander = QueryExpansion()
                    keywords = query_expander.expand(keywords, use_llm=False)
                
                keyword_results = db.search_by_keywords(
                    keywords=keywords,
                    limit_per_keyword=request.keyword_limit,
                    max_total=request.limit
                )
                
                search_results = [
                    SearchResult(
                        file_path=result.get('file_path', ''),
                        file_type=result.get('file_type'),
                        location_info=result.get('location_info'),
                        snippet=result.get('snippet', '')
                    )
                    for result in keyword_results[:request.limit]
                ]
                
                search_response = SearchResponse(
                    query=request.query,
                    results=search_results,
                    total=len(search_results)
                )
        
        if not search_response.results or len(search_response.results) == 0:
            return RAGResponse(
                query=request.query,
                answer="関連する情報が見つかりませんでした。",
                sources=[],
                model_used="",
                provider_used=""
            )
        
        # 2. コンテキストを構築
        context_parts = []
        for i, result in enumerate(search_response.results[:request.limit], 1):
            context_part = f"[資料{i}] {result.file_path}"
            if result.location_info:
                context_part += f" ({result.location_info})"
            context_part += f"\n{result.snippet}\n"
            context_parts.append(context_part)
        
        context = "\n".join(context_parts)
        
        # 3. プロンプトを作成
        system_prompt = """あなたは知識ベースの質問応答システムです。
提供された検索コンテキストに基づいて、ユーザーの質問に対して構造化された回答を生成してください。
回答は日本語で、明確かつ論理的に記述してください。"""
        
        user_prompt = f"""以下の「検索コンテキスト」に基づいて、ユーザーの質問に回答してください。

回答の指針:

1. 要約: まず、結論を簡潔に述べてください。

2. 詳細分析: その結論に至る主な要因を3点挙げ、それぞれについてコンテキスト内の具体的な根拠を用いて詳しく説明してください。

3. 背景と洞察: 単なる事実だけでなく、「なぜそのような状況になっているのか」という背景や、データが示唆する潜在的な課題についてあなたの考察を述べてください。

4. 制約: コンテキストにない情報は含めず、情報が不足している場合は「XXに関する記述がないため不明」と明記してください。

ユーザーの質問: {request.query}

検索コンテキスト:
{context}

【回答】"""
        
        # 4. LLMプロバイダーを決定
        if request.llm_provider:
            try:
                provider_type = LLMProviderType(request.llm_provider)
            except ValueError:
                # デフォルトは環境変数から取得
                provider_type_str = os.environ.get("LLM_PROVIDER", "openrouter")
                try:
                    provider_type = LLMProviderType(provider_type_str)
                except ValueError:
                    provider_type = LLMProviderType.OPENROUTER
        else:
            # 環境変数から取得（デフォルトはOpenRouter）
            provider_type_str = os.environ.get("LLM_PROVIDER", "openrouter")
            try:
                provider_type = LLMProviderType(provider_type_str)
            except ValueError:
                provider_type = LLMProviderType.OPENROUTER
        
        # 5. LLMプロバイダーを作成
        try:
            llm_provider_kwargs = {
                "provider_type": provider_type,
                "model": request.model,
            }
            
            # LiteLLMプロバイダーの場合
            if provider_type == LLMProviderType.LITELLM and request.api_base:
                llm_provider_kwargs["api_base"] = request.api_base
            
            # AWS Bedrockプロバイダーの場合
            if provider_type == LLMProviderType.AWS_BEDROCK:
                if request.aws_region_name:
                    llm_provider_kwargs["region_name"] = request.aws_region_name
                if request.aws_access_key_id:
                    llm_provider_kwargs["aws_access_key_id"] = request.aws_access_key_id
                if request.aws_secret_access_key:
                    llm_provider_kwargs["aws_secret_access_key"] = request.aws_secret_access_key
            
            llm_provider = create_llm_provider(**llm_provider_kwargs)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"LLMプロバイダーの作成に失敗しました: {str(e)}"
            )
        
        # 6. 回答を生成
        try:
            answer = llm_provider.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"回答の生成に失敗しました: {str(e)}"
            )
        
        # 7. 使用したモデル名を取得
        if hasattr(llm_provider, 'model') and llm_provider.model:
            model_used = llm_provider.model
        else:
            model_used = llm_provider.get_default_model()
        
        # 8. レスポンスを返却
        return RAGResponse(
            query=request.query,
            answer=answer,
            sources=search_response.results[:request.limit],
            model_used=model_used,
            provider_used=provider_type.value
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"RAG処理中にエラーが発生しました: {str(e)}\n\n詳細:\n{error_details}"
        )


@router.get("/rag", response_model=RAGResponse)
async def rag_query_get(
    query: str,
    limit: int = 20,
    hybrid_weight: float = 0.5,
    keyword_limit: int = 10,
    vector_limit: int = 20,
    expand_synonyms: bool = False,
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    aws_region_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
):
    """
    RAGによる回答生成（GET版）
    
    Args:
        query: ユーザーの質問
        limit: 検索結果の最大数
        hybrid_weight: ベクトル検索の重み
        keyword_limit: 各キーワードあたりの全文検索取得件数
        vector_limit: ベクトル検索の取得件数
        expand_synonyms: 類義語展開を使用するかどうか
        llm_provider: LLMプロバイダー（openrouter, litellm, aws_bedrock）
        model: 使用するLLMモデル名またはARN（ARNは推論エンドポイントの場合に使用）
        api_base: LiteLLMのカスタムエンドポイントURL
        aws_region_name: AWSリージョン名（aws_bedrockプロバイダーの場合のみ）
        aws_access_key_id: AWSアクセスキーID（aws_bedrockプロバイダーの場合のみ）
        aws_secret_access_key: AWSシークレットキー（aws_bedrockプロバイダーの場合のみ）
        temperature: 温度パラメータ
        max_tokens: 最大トークン数
    
    Returns:
        回答とソース情報
    """
    request = RAGRequest(
        query=query,
        limit=limit,
        hybrid_weight=hybrid_weight,
        keyword_limit=keyword_limit,
        vector_limit=vector_limit,
        expand_synonyms=expand_synonyms,
        llm_provider=llm_provider,
        model=model,
        api_base=api_base,
        aws_region_name=aws_region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return await rag_query(request)

