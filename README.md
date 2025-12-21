# Obsidian MCP Server

Obsidianとの連携を行うFastMCPサーバーです。

## セットアップ

### 簡単な起動方法（推奨）

Windowsの場合、`start.bat` をダブルクリックするか、コマンドプロンプトで実行してください：

```bash
start.bat
```

このスクリプトは以下を自動的に実行します：
1. venv環境の作成（存在しない場合）
2. venv環境の有効化
3. 依存関係のインストール
4. サーバーの起動

### 手動セットアップ

#### 1. venv環境の作成

```bash
python -m venv venv
```

#### 2. venv環境の有効化

Windows:
```bash
venv\Scripts\activate
```

#### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

#### 4. サーバーの起動

```bash
python main.py
```

または

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## プロジェクト構造

このプロジェクトは機能単位でモジュール化されています：

```
knowledge_mcp_obsidian/
  ├── main.py              # FastAPIアプリケーションのエントリーポイント
  ├── modules/             # 機能別モジュールディレクトリ
  │   ├── hello_module/    # Hello World機能モジュール
  │   │   ├── __init__.py
  │   │   └── router.py
  │   └── README.md        # モジュール作成ガイド
  └── ...
```

各機能は `機能名_module` というディレクトリに分離され、独自の`APIRouter`を持ちます。
新しい機能を追加する場合は、`modules/README.md` を参照してください。

## API エンドポイント

### ルートエンドポイント
- `GET /` - サーバー情報
- `GET /health` - ヘルスチェックエンドポイント

### Hello Module
- `GET /hello/` - Hello Worldエンドポイント
- `GET /hello/health` - Hello Module のヘルスチェック

### Search Module (全文検索エンジン)
- `POST /search/index` - ディレクトリをスキャンしてインデックスを作成（バックグラウンド処理、ジョブIDを返す）
- `GET /search/jobs/{job_id}` - ジョブの進捗状況を取得
- `GET /search/jobs?job_type={タイプ}&status={ステータス}&limit={件数}` - ジョブ一覧を取得
- `POST /search/jobs/{job_id}/cancel` - ジョブをキャンセル
- `GET /search/query?query={キーワード}&limit={件数}` - 全文検索を実行（GET版）
- `POST /search/query` - 全文検索を実行（POST版）
- `GET /search/stats` - インデックス統計情報を取得
- `DELETE /search/index` - すべてのインデックスをクリア

### Task Module (タスク管理UI)
- `GET /task/create_index` - インデックス作成状況を確認するWebページ（リアルタイム進捗表示）

#### 対応ファイル形式
- **PDF**: ページ単位で解析（PyMuPDF使用）
- **Word (.docx)**: ドキュメント全体を解析（python-docx使用）
- **PowerPoint (.pptx)**: スライド単位で解析（python-pptx使用）
- **Excel (.xlsx)**: シート単位で解析（openpyxl使用、数値のみのシートは除外）
- **テキストファイル**: .txt, .md, .py, .js, .ts, .json, .xml, .html, .css, .yaml, .csvなど

#### 技術仕様
- **データベース**: SQLite (FTS5拡張機能)
- **日本語解析**: Janome（分かち書き）
- **検索**: FTS5のMATCH構文を使用した高速全文検索
- **キュー管理**: 汎用的なジョブキュー管理システム（進捗追跡対応）

#### キュー管理システム
インデックス作成などの長時間処理は、バックグラウンドで実行され、進捗を追跡できます。

- **ジョブステータス**: `pending` → `processing` → `completed` / `failed` / `cancelled`
- **進捗情報**: 現在の処理数、総数、パーセンテージ、メッセージ
- **結果**: ジョブ完了時の結果データ（JSON形式）
- **エラー処理**: 失敗時はエラーメッセージを記録

## CORS設定

すべてのオリジンからのリクエストを受け付けます（開発環境用）。

## アクセス

サーバー起動後、以下のURLにアクセスできます：

- API: http://127.0.0.1:8000
- ドキュメント: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Obsidianプラグイン開発者向け

Obsidianプラグインからこのサーバーに接続する方法については、[OBSIDIAN_PLUGIN_INTEGRATION.md](./OBSIDIAN_PLUGIN_INTEGRATION.md) を参照してください。

このドキュメントには以下の情報が含まれています：
- TypeScriptでの実装例
- エンドポイントの詳細
- エラーハンドリングの方法
- 完全なプラグインコード例

