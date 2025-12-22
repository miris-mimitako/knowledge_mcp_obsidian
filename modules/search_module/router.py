"""
Search Module Router
全文検索エンジンのAPIエンドポイントを提供するルーター
"""
import os
import threading
from pathlib import Path
from typing import List, Optional, Callable
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
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
    model: Optional[str] = Field(default=None, description="使用するモデル名（例: text-embedding-ada-002, gemini/text-embedding-004）")
    api_base: Optional[str] = Field(default=None, description="LiteLLMのカスタムエンドポイントURL（litellmプロバイダーの場合のみ）")
    chunk_size: int = Field(default=512, description="チャンクサイズ（トークン数）")
    chunk_overlap: int = Field(default=50, description="オーバーラップサイズ（トークン数）")


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
    llm_provider: Optional[str] = Field(default=None, description="LLMプロバイダー（openrouter, litellm）")
    model: Optional[str] = Field(default=None, description="使用するLLMモデル名")
    api_base: Optional[str] = Field(default=None, description="LiteLLMのカスタムエンドポイントURL（litellmプロバイダーの場合のみ）")
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
            chunk_size=512,
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
    
    for root, dirs, filenames in os.walk(directory_path):
        # 隠しディレクトリや不要なディレクトリをスキップ
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for filename in filenames:
            if Path(filename).suffix.lower() in supported_extensions:
                file_path = os.path.join(root, filename)
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
    
    Args:
        file_path: インデックスに追加するファイルのパス
        db: データベースインスタンス
        tokenizer: トークナイザーインスタンス
        job_id: ジョブID（進捗更新用）
    
    Returns:
        成功した場合、またはスキップされた場合はTrue、失敗した場合はFalse
    """
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


@task_router.get("/index_lists/", response_class=HTMLResponse)
async def index_lists_page():
    """
    インデックスされたファイルのリストを表示するWebページ
    
    Returns:
        HTMLページ
    """
    try:
        # データベースから全ドキュメントを取得
        documents = db.get_all_documents()
        total_count = db.get_document_count()
        
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
            </style>
        </head>
        <body>
            <div class="container">
                <a href="/task/" class="back-link">← タスク管理に戻る</a>
                <h1>インデックスリスト</h1>
                <div class="stats">
                    総ファイル数: {len(files_list)} | 総ドキュメント数: {total_count}
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


def process_vectorize_job(
    job_id: int,
    directory_path: str,
    provider_type_str: Optional[str] = None,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50
):
    """
    ベクトル化ジョブをバックグラウンドで実行
    
    Args:
        job_id: ジョブID
        directory_path: ベクトル化するディレクトリのパス
        provider_type_str: Embeddingプロバイダーのタイプ
        model: 使用するモデル名（オプション）
        api_base: LiteLLMのカスタムエンドポイントURL（litellmプロバイダーの場合のみ）
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
                provider_kwargs["model"] = model
            if provider_type == EmbeddingProviderType.LITELLM and api_base:
                provider_kwargs["api_base"] = api_base
            
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
            progress_callback=progress_callback
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
        processed_files = result.get("processed_files", 0)
        total_chunks = result.get("total_chunks", 0)
        
        completion_message = (
            f"ベクトル化が完了しました。\n"
            f"処理済み: {processed_files}件"
        )
        if skipped_files > 0:
            completion_message += f", スキップ: {skipped_files}件（更新日時が変更されていないファイル）"
        completion_message += f"\nチャンク数: {total_chunks}件"
        
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
                "chunk_size": request.chunk_size,
                "chunk_overlap": request.chunk_overlap
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
                request.chunk_size,
                request.chunk_overlap
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
        # 1. ハイブリッド検索を実行してコンテキストを取得
        hybrid_request = HybridSearchRequest(
            query=request.query,
            limit=request.limit,
            hybrid_weight=request.hybrid_weight,
            keyword_limit=request.keyword_limit,
            vector_limit=request.vector_limit,
            expand_synonyms=request.expand_synonyms
        )
        
        search_response = await hybrid_search(hybrid_request)
        
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
提供された資料を基に、ユーザーの質問に対して正確で分かりやすい回答を生成してください。
資料に記載されていない情報については推測せず、「資料に記載がありません」と答えてください。
回答は日本語で、簡潔かつ明確に記述してください。"""
        
        user_prompt = f"""以下の資料を参考に、質問に答えてください。

【資料】
{context}

【質問】
{request.query}

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
            llm_provider = create_llm_provider(
                provider_type=provider_type,
                model=request.model,
                api_base=request.api_base
            )
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
        llm_provider: LLMプロバイダー（openrouter, litellm）
        model: 使用するLLMモデル名
        api_base: LiteLLMのカスタムエンドポイントURL
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
        temperature=temperature,
        max_tokens=max_tokens
    )
    return await rag_query(request)

