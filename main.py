from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import os

# .envファイルのサポート（python-dotenvがインストールされている場合）
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ .envファイルを読み込みました（python-dotenvが利用可能な場合）")
except ImportError:
    print("⚠ python-dotenvがインストールされていません。.envファイルは読み込まれません。")

# モジュールのrouterをインポート
from modules.hello_module import router as hello_router
from modules.search_module import router as search_router
from modules.search_module import task_router

app = FastAPI(title="Obsidian MCP Server")

# CORS設定: ANYで受け付ける
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # すべてのオリジンを許可
    allow_credentials=True,
    allow_methods=["*"],  # すべてのHTTPメソッドを許可
    allow_headers=["*"],  # すべてのヘッダーを許可
)

# ルートパス用のエンドポイント - /task/ にリダイレクト
@app.get("/")
async def root():
    """ルートエンドポイント - /task/ にリダイレクト"""
    return RedirectResponse(url="/task/")

@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント（ルートパス）"""
    return {"status": "healthy"}


@app.get("/env-check")
async def env_check():
    """
    環境変数の確認エンドポイント
    APIキーが正しく読み込まれているかを確認できます
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "openrouter")
    
    # APIキーの有無を確認（実際の値は表示しない）
    api_key_set = api_key is not None and api_key.strip() != ""
    api_key_preview = f"{api_key[:10]}..." if api_key_set and len(api_key) > 10 else ("設定されています" if api_key_set else "未設定")
    
    return {
        "OPENROUTER_API_KEY": {
            "設定済み": api_key_set,
            "プレビュー": api_key_preview,
            "長さ": len(api_key) if api_key_set else 0
        },
        "EMBEDDING_PROVIDER": embedding_provider,
        "すべての環境変数": {
            key: ("設定済み" if value else "未設定")
            for key, value in {
                "OPENROUTER_API_KEY": api_key_set,
                "EMBEDDING_PROVIDER": bool(embedding_provider)
            }.items()
        }
    }

# モジュールのrouterを登録
app.include_router(hello_router)
app.include_router(search_router)
app.include_router(task_router)

# 監視スケジューラーを開始
from modules.search_module.router import start_watcher
start_watcher()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

