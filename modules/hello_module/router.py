"""
Hello Module Router
基本的なHello Worldエンドポイントを提供するルーター
"""
from fastapi import APIRouter

router = APIRouter(
    prefix="/hello",
    tags=["hello"],
    responses={404: {"description": "Not found"}},
)


@router.get("/")
async def hello_world():
    """Hello Worldエンドポイント"""
    return {"message": "Hello World from Obsidian MCP Server"}


@router.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {"status": "healthy"}

