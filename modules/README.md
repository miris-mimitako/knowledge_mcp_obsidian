# Modules ディレクトリ

このディレクトリには、機能単位で分離されたモジュールが配置されます。

## ディレクトリ構造

各機能は `機能名_module` というディレクトリ名で作成します。

```
modules/
  ├── __init__.py
  ├── hello_module/
  │   ├── __init__.py
  │   └── router.py
  └── 機能名_module/
      ├── __init__.py
      └── router.py
```

## 新しいモジュールの作成方法

1. `modules/機能名_module/` ディレクトリを作成
2. `__init__.py` を作成し、routerをエクスポート
3. `router.py` を作成し、`APIRouter` を定義
4. `main.py` でrouterをインポートして登録

### 例: `example_module` の作成

#### 1. `modules/example_module/__init__.py`
```python
"""
Example Module
説明をここに書く
"""
from .router import router

__all__ = ["router"]
```

#### 2. `modules/example_module/router.py`
```python
"""
Example Module Router
説明をここに書く
"""
from fastapi import APIRouter

router = APIRouter(
    prefix="/example",  # エンドポイントのプレフィックス
    tags=["example"],   # OpenAPIドキュメントのタグ
    responses={404: {"description": "Not found"}},
)


@router.get("/")
async def example_endpoint():
    """エンドポイントの説明"""
    return {"message": "This is an example endpoint"}
```

#### 3. `main.py` での登録
```python
from modules.example_module import router as example_router

# routerを登録
app.include_router(example_router)
```

## モジュール設計のガイドライン

1. **1モジュール = 1機能**: 各モジュールは単一の責任を持つように設計
2. **独立したrouter**: 各モジュールは独自の`APIRouter`を持つ
3. **明確なプレフィックス**: routerには機能を表すプレフィックスを設定
4. **ドキュメント**: 各モジュールとエンドポイントには適切なdocstringを記述

