# データベーススキーマ設計

## 概要

このモジュールは、SQLiteデータベースを使用して全文検索機能とジョブキュー管理を提供します。

## テーブル構成

### 1. `documents` テーブル（メタデータ管理）

全文検索対象のドキュメントのメタデータを管理します。

| カラム名 | 型 | 説明 |
|---------|-----|------|
| id | INTEGER PRIMARY KEY | ドキュメントID（自動採番） |
| file_path | TEXT NOT NULL | ファイルのフルパス |
| file_type | TEXT | ファイルタイプ（pdf, docx, xlsx, txtなど） |
| location_info | TEXT | 位置情報（ページ番号、シート名など） |
| file_modified_time | REAL | ファイルの最終更新日時（Unix timestamp、秒単位） |
| updated_at | DATETIME | 更新日時（デフォルト: CURRENT_TIMESTAMP） |

**インデックス**: なし（FTS5との紐付けはrowidで行う）

### 2. `documents_fts` テーブル（FTS5仮想テーブル）

全文検索用の仮想テーブルです。分かち書き済みのテキストを格納します。

| カラム名 | 型 | 説明 |
|---------|-----|------|
| rowid | INTEGER | documentsテーブルのidと紐付け |
| content | TEXT | 分かち書き済みの検索対象テキスト |

**特徴**:
- FTS5仮想テーブルを使用した高速全文検索
- `rowid`で`documents`テーブルと紐付け
- `MATCH`構文で検索可能

### 3. `job_queue` テーブル（汎用的なキュー管理）

各種ジョブ（インデックス作成、エクスポート、バックアップなど）の進捗を管理します。

| カラム名 | 型 | 説明 |
|---------|-----|------|
| id | INTEGER PRIMARY KEY | ジョブID（自動採番） |
| job_type | TEXT NOT NULL | ジョブタイプ（'index', 'export', 'backup'など） |
| status | TEXT NOT NULL | ステータス（'pending', 'processing', 'completed', 'failed', 'cancelled'） |
| parameters | TEXT | ジョブのパラメータ（JSON形式） |
| progress | TEXT | 進捗情報（JSON形式: `{"current": 10, "total": 100, "percentage": 10.0, "message": "処理中..."}`） |
| result | TEXT | 結果データ（JSON形式） |
| error_message | TEXT | エラーメッセージ（失敗時） |
| created_at | DATETIME | 作成日時（デフォルト: CURRENT_TIMESTAMP） |
| started_at | DATETIME | 開始日時 |
| completed_at | DATETIME | 完了日時 |
| updated_at | DATETIME | 更新日時（デフォルト: CURRENT_TIMESTAMP） |

**インデックス**:
- `idx_job_queue_status`: statusカラム
- `idx_job_queue_type`: job_typeカラム
- `idx_job_queue_created`: created_atカラム（降順）

## ジョブステータスの遷移

```
pending → processing → completed
                    → failed
                    → cancelled (pending/processing時のみ)
```

## 使用例

### インデックス作成ジョブの進捗確認

1. ジョブ作成: `POST /search/index`
   ```json
   {
     "directory_path": "C:/documents",
     "clear_existing": false
   }
   ```
   レスポンス: `{"job_id": 1, ...}`

2. 進捗確認: `GET /search/jobs/1`
   ```json
   {
     "id": 1,
     "job_type": "index",
     "status": "processing",
     "progress": {
       "current": 45,
       "total": 100,
       "percentage": 45.0,
       "message": "処理中: document.pdf"
     },
     ...
   }
   ```

3. 完了確認: `GET /search/jobs/1`
   ```json
   {
     "id": 1,
     "status": "completed",
     "result": {
       "indexed_files": 100,
       "indexed_documents": 150,
       ...
     },
     ...
   }
   ```

## 拡張性

`job_queue`テーブルは汎用的に設計されており、以下のようなジョブタイプを追加できます：

- `export`: データエクスポートジョブ
- `backup`: バックアップジョブ
- `sync`: 同期ジョブ
- その他の長時間処理タスク

各ジョブタイプは、`parameters`と`result`フィールドにJSON形式で自由にデータを格納できます。

