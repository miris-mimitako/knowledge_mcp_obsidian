# Obsidianプラグイン開発者向け MCPサーバー接続ガイド

このドキュメントは、ObsidianプラグインからこのMCPサーバーに接続する方法を説明します。

## サーバー情報

- **ベースURL**: `http://127.0.0.1:8000`
- **CORS**: すべてのオリジンから許可されています
- **プロトコル**: HTTP REST API

## 前提条件

1. MCPサーバーが起動していること（`start.bat`を実行してサーバーを起動してください）
2. ObsidianプラグインがHTTPリクエストを送信できること

## 利用可能なエンドポイント

### 1. Hello World

**エンドポイント**: `GET /`

**説明**: サーバーが正常に動作しているか確認するためのエンドポイント

**レスポンス例**:
```json
{
  "message": "Obsidian MCP Server",
  "version": "1.0.0",
  "status": "running"
}
```

### 2. ヘルスチェック

**エンドポイント**: `GET /health`

**説明**: サーバーの健全性を確認するエンドポイント

**レスポンス例**:
```json
{
  "status": "healthy"
}
```

### 3. 全文検索エンジン（Search Module）

#### 3.1 インデックス作成

**エンドポイント**: `POST /search/index`

**説明**: 指定されたディレクトリ内のファイルをスキャンしてインデックスを作成（バックグラウンド処理）

**リクエストボディ**:
```json
{
  "directory_path": "C:/path/to/documents",
  "clear_existing": false
}
```

**レスポンス例**:
```json
{
  "message": "インデックス作成ジョブを開始しました",
  "job_id": 1,
  "directory_path": "C:/path/to/documents"
}
```

**対応ファイル形式**:
- PDF（ページ単位）
- Word (.docx)
- PowerPoint (.pptx)
- Excel (.xlsx、数値のみのシートは除外）
- テキストファイル（.txt, .md, .py, .js, .ts, .json, .xml, .html, .css, .yaml, .csvなど）

#### 3.2 全文検索

**エンドポイント**: `GET /search/query?query={キーワード}&limit={件数}` または `POST /search/query`

**説明**: キーワードで全文検索を実行

**リクエスト例（GET）**:
```
GET /search/query?query=プロジェクト&limit=50
```

**リクエストボディ（POST）**:
```json
{
  "query": "プロジェクト",
  "limit": 50
}
```

**レスポンス例**:
```json
{
  "query": "プロジェクト",
  "results": [
    {
      "file_path": "C:/documents/project.md",
      "file_type": "md",
      "location_info": "Full Document",
      "snippet": "【プロジェクトの概要】このプロジェクトは..."
    }
  ],
  "total": 1
}
```

#### 3.3 ジョブ管理

**エンドポイント**: `GET /search/jobs/{job_id}`

**説明**: ジョブの進捗状況を取得

**レスポンス例**:
```json
{
  "id": 1,
  "job_type": "index",
  "status": "processing",
  "parameters": {
    "directory_path": "C:/documents",
    "clear_existing": false
  },
  "progress": {
    "current": 45,
    "total": 100,
    "percentage": 45.0,
    "message": "処理中: document.pdf"
  },
  "created_at": "2024-01-01T10:00:00",
  "started_at": "2024-01-01T10:00:05",
  "updated_at": "2024-01-01T10:05:30"
}
```

**エンドポイント**: `GET /search/jobs?status={ステータス}&limit={件数}`

**説明**: ジョブ一覧を取得（フィルタ可能）

**クエリパラメータ**:
- `status`: `pending`, `processing`, `completed`, `failed`, `cancelled`（オプション）
- `limit`: 取得件数の上限（デフォルト: 100）

**エンドポイント**: `POST /search/jobs/{job_id}/cancel`

**説明**: ジョブをキャンセル

**レスポンス例**:
```json
{
  "message": "ジョブ 1 をキャンセルしました",
  "job_id": 1
}
```

#### 3.4 統計情報

**エンドポイント**: `GET /search/stats`

**説明**: インデックス統計情報を取得

**レスポンス例**:
```json
{
  "total_documents": 150,
  "database_path": "search_index.db"
}
```

#### 3.5 ベクトル化（Vectorization）

**エンドポイント**: `POST /search/vectorize`

**説明**: 指定されたディレクトリ内のドキュメントをベクトル化してChromaDBに保存（バックグラウンド処理）

**ファイル更新日時チェック**: ベクトル化時、ファイルの更新日時をチェックします。更新日時が変更されていないファイルは自動的にスキップされ、処理時間とAPI呼び出しコストを削減します。

**リクエストボディ**:
```json
{
  "directory_path": "C:/path/to/documents",
  "provider": "openrouter",
  "model": "text-embedding-ada-002",
  "api_base": "http://localhost:4000",
  "chunk_size": 512,
  "chunk_overlap": 50
}
```

**パラメータ説明**:
- `directory_path`: ベクトル化するディレクトリのパス（必須）
- `provider`: Embeddingプロバイダー（`openrouter`, `aws_bedrock`, `litellm`、オプション）
- `model`: 使用するモデル名（例: `text-embedding-ada-002`, `gemini/text-embedding-004`、オプション）
- `api_base`: LiteLLMのカスタムエンドポイントURL（`litellm`プロバイダーの場合のみ、オプション）
- `chunk_size`: チャンクサイズ（トークン数、デフォルト: 512）
- `chunk_overlap`: オーバーラップサイズ（トークン数、デフォルト: 50）

**レスポンス例**:
```json
{
  "message": "ベクトル化ジョブを開始しました",
  "job_id": 2,
  "directory_path": "C:/path/to/documents"
}
```

**ジョブ結果に含まれる情報**:
```json
{
  "processed_files": 45,
  "skipped_files": 10,
  "failed_files": 0,
  "total_files": 55,
  "total_chunks": 1250
}
```

- `skipped_files`: 更新日時が変更されていないためスキップされたファイル数

**エンドポイント**: `GET /search/vectorize/stats`

**説明**: ベクトルストアの統計情報を取得

**レスポンス例**:
```json
{
  "collection_name": "obsidian_knowledge_base",
  "total_chunks": 1250,
  "persist_directory": "./chroma_db"
}
```

#### 3.6 ハイブリッド検索（Hybrid Search）

**エンドポイント**: `POST /search/hybrid` または `GET /search/hybrid`

**説明**: 全文検索（BM25）とベクトル検索を組み合わせたハイブリッド検索を実行。結果はRRF（Reciprocal Rank Fusion）で統合されます。

**処理フロー**:
1. クエリの前処理（形態素解析、ストップワード除去、オプションで類義語展開）
2. 全文検索とベクトル検索を並行実行
3. RRF（Reciprocal Rank Fusion）で統合
4. 上位結果を返却

**リクエストボディ（POST）**:
```json
{
  "query": "Pythonの文字列処理について",
  "limit": 20,
  "hybrid_weight": 0.5,
  "keyword_limit": 10,
  "vector_limit": 20,
  "expand_synonyms": false
}
```

**クエリパラメータ（GET）**:
- `query`: 検索キーワード（必須）
- `limit`: 返却する結果の最大数（デフォルト: 20、1-100）
- `hybrid_weight`: ベクトル検索の重み（デフォルト: 0.5、0.0-1.0）
  - `0.0`: 全文検索のみ
  - `0.5`: 等価（デフォルト）
  - `1.0`: ベクトル検索のみ
- `keyword_limit`: 各キーワードあたりの全文検索取得件数（デフォルト: 10、1-50）
- `vector_limit`: ベクトル検索の取得件数（デフォルト: 20、1-100）
- `expand_synonyms`: 類義語展開を使用するか（デフォルト: false）

**GETリクエスト例**:
```
GET /search/hybrid?query=Pythonの文字列処理&limit=20&hybrid_weight=0.5
```

**レスポンス例**:
```json
{
  "query": "Pythonの文字列処理について",
  "results": [
    {
      "file_path": "C:/documents/python_strings.md",
      "file_type": "md",
      "location_info": "Full Document",
      "snippet": "【Pythonの文字列処理】Pythonでは文字列を..."
    }
  ],
  "total": 1
}
```

**注意事項**:
- ハイブリッド検索を使用するには、事前にベクトル化（`POST /search/vectorize`）を実行する必要があります
- ベクトル検索が失敗した場合でも、全文検索のみで動作します

#### 3.7 RAG回答生成（Retrieval-Augmented Generation）

**エンドポイント**: `POST /search/rag` または `GET /search/rag`

**説明**: ハイブリッド検索でコンテキストを取得し、LLMで質問に対する回答を生成します。

**処理フロー**:
1. ハイブリッド検索を実行してコンテキストを取得
2. 取得したコンテキストと質問を組み合わせてプロンプトを作成
3. LLMで回答を生成

**リクエストボディ（POST）**:
```json
{
  "query": "Pythonの文字列処理について教えてください",
  "limit": 20,
  "hybrid_weight": 0.5,
  "keyword_limit": 10,
  "vector_limit": 20,
  "expand_synonyms": false,
  "llm_provider": "openrouter",
  "model": "google/gemini-3-flash-preview",
  "api_base": "http://localhost:4000",
  "temperature": 0.7,
  "max_tokens": null
}
```

**クエリパラメータ（GET）**:
- `query`: ユーザーの質問（必須）
- `limit`: 検索結果の最大数（デフォルト: 20、1-100）
- `hybrid_weight`: ベクトル検索の重み（デフォルト: 0.5、0.0-1.0）
- `keyword_limit`: 各キーワードあたりの全文検索取得件数（デフォルト: 10、1-50）
- `vector_limit`: ベクトル検索の取得件数（デフォルト: 20、1-100）
- `expand_synonyms`: 類義語展開を使用するか（デフォルト: false）
- `llm_provider`: LLMプロバイダー（`openrouter`, `litellm`、オプション）
- `model`: 使用するLLMモデル名（オプション）
- `api_base`: LiteLLMのカスタムエンドポイントURL（`litellm`プロバイダーの場合のみ、オプション）
- `temperature`: 温度パラメータ（デフォルト: 0.7、0.0-2.0）
- `max_tokens`: 最大トークン数（オプション）

**GETリクエスト例**:
```
GET /search/rag?query=Pythonの文字列処理について&llm_provider=openrouter&limit=10
```

**レスポンス例**:
```json
{
  "query": "Pythonの文字列処理について教えてください",
  "answer": "Pythonの文字列処理には以下のような方法があります...\n\n1. 文字列の結合: + 演算子や join() メソッドを使用...",
  "sources": [
    {
      "file_path": "C:/documents/python_strings.md",
      "file_type": "md",
      "location_info": "Full Document",
      "snippet": "【Pythonの文字列処理】Pythonでは文字列を..."
    }
  ],
  "model_used": "google/gemini-3-flash-preview",
  "provider_used": "openrouter"
}
```

**デフォルト動作**:
- `llm_provider`が未指定の場合、環境変数`LLM_PROVIDER`から取得（デフォルト: `openrouter`）
- `openrouter`プロバイダーの場合、デフォルトモデルは`google/gemini-3-flash-preview`
- `litellm`プロバイダーの場合、LiteLLMのデフォルトモデルを使用（環境変数`LITELLM_MODEL`で指定可能、デフォルト: `gpt-3.5-turbo`）

**注意事項**:
- RAGを使用するには、事前にベクトル化（`POST /search/vectorize`）を実行する必要があります
- LLMプロバイダーのAPIキーが設定されている必要があります（`OPENROUTER_API_KEY`など）

#### 3.8 LiteLLMモデルリスト取得

**エンドポイント**: `GET /search/llm/models`

**説明**: LiteLLMの利用可能なモデルリストを取得します。Obsidianから問い合わせを受けた際に、アクセスポイント（`api_base`）を指定してもらい、そこへ `/models` でアクセスしてモデルリストを取得します。

**クエリパラメータ**:
- `api_base`: LiteLLMのカスタムエンドポイントURL（必須）
  - 例: `http://localhost:4000` または `https://api.example.com/v1`
  - 指定しない場合は環境変数`LITELLM_API_BASE`から取得

**リクエスト例**:
```
GET /search/llm/models?api_base=http://localhost:4000
```

**レスポンス例**:
```json
{
  "api_base": "http://localhost:4000",
  "models": [
    {
      "id": "gpt-3.5-turbo",
      "name": "gpt-3.5-turbo",
      "object": "model"
    },
    {
      "id": "gpt-4",
      "name": "gpt-4",
      "object": "model"
    }
  ],
  "total": 2
}
```

**エラーレスポンス**:
```json
{
  "detail": "api_baseパラメータが必要です。\n例: GET /search/llm/models?api_base=http://localhost:4000"
}
```

### 4. タスク管理UI（Task Module）

#### 4.1 インデックス作成状況ページ

**エンドポイント**: `GET /task/create_index`

**説明**: インデックス作成状況を確認するWebページ（HTML形式）

**説明**: ブラウザでアクセスすると、リアルタイムで進捗を確認できるWebページが表示されます。自動更新機能（2秒ごと）やフィルタ機能、ジョブキャンセル機能が利用できます。

**アクセス方法**: ブラウザで `http://127.0.0.1:8000/task/create_index` にアクセス

#### 4.2 ベクトル化状況ページ

**エンドポイント**: `GET /task/create_vector`

**説明**: ベクトル化ジョブの進捗状況を確認するWebページ（HTML形式）

**説明**: ブラウザでアクセスすると、リアルタイムでベクトル化の進捗を確認できるWebページが表示されます。自動更新機能（2秒ごと）やフィルタ機能、ジョブキャンセル機能が利用できます。

**アクセス方法**: ブラウザで `http://127.0.0.1:8000/task/create_vector` にアクセス

## Obsidianプラグインからの接続方法

### TypeScriptでの実装例

```typescript
// サーバーのベースURL
const MCP_SERVER_URL = 'http://127.0.0.1:8000';

/**
 * MCPサーバーにGETリクエストを送信
 */
async function callMCPServer(endpoint: string): Promise<any> {
    try {
        const response = await fetch(`${MCP_SERVER_URL}${endpoint}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('MCPサーバーへの接続エラー:', error);
        throw error;
    }
}

/**
 * MCPサーバーにPOSTリクエストを送信
 */
async function postToMCPServer(endpoint: string, data: any): Promise<any> {
    try {
        const response = await fetch(`${MCP_SERVER_URL}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('MCPサーバーへの接続エラー:', error);
        throw error;
    }
}

// 使用例
async function checkServerHealth() {
    try {
        const result = await callMCPServer('/health');
        console.log('サーバーステータス:', result.status);
    } catch (error) {
        console.error('サーバーに接続できません:', error);
    }
}

async function getHelloWorld() {
    try {
        const result = await callMCPServer('/');
        console.log('メッセージ:', result.message);
    } catch (error) {
        console.error('エラー:', error);
    }
}

// 全文検索の使用例
async function createIndex(directoryPath: string, clearExisting: boolean = false) {
    try {
        const result = await postToMCPServer('/search/index', {
            directory_path: directoryPath,
            clear_existing: clearExisting
        });
        console.log('インデックス作成ジョブ開始:', result.job_id);
        return result.job_id;
    } catch (error) {
        console.error('インデックス作成エラー:', error);
        throw error;
    }
}

async function searchDocuments(query: string, limit: number = 50) {
    try {
        const encodedQuery = encodeURIComponent(query);
        const result = await callMCPServer(`/search/query?query=${encodedQuery}&limit=${limit}`);
        console.log(`検索結果: ${result.total}件`);
        return result.results;
    } catch (error) {
        console.error('検索エラー:', error);
        throw error;
    }
}

async function vectorizeDirectory(directoryPath: string, provider?: string) {
    try {
        const result = await postToMCPServer('/search/vectorize', {
            directory_path: directoryPath,
            provider: provider,
            chunk_size: 512,
            chunk_overlap: 50
        });
        console.log('ベクトル化ジョブ開始:', result.job_id);
        return result.job_id;
    } catch (error) {
        console.error('ベクトル化エラー:', error);
        throw error;
    }
}

async function hybridSearch(
    query: string,
    limit: number = 20,
    hybridWeight: number = 0.5,
    expandSynonyms: boolean = false
) {
    try {
        const encodedQuery = encodeURIComponent(query);
        const url = `/search/hybrid?query=${encodedQuery}&limit=${limit}&hybrid_weight=${hybridWeight}&expand_synonyms=${expandSynonyms}`;
        const result = await callMCPServer(url);
        console.log(`ハイブリッド検索結果: ${result.total}件`);
        return result.results;
    } catch (error) {
        console.error('ハイブリッド検索エラー:', error);
        throw error;
    }
}

async function ragQuery(
    query: string,
    llmProvider?: string,
    model?: string,
    apiBase?: string,
    limit: number = 20,
    temperature: number = 0.7
) {
    try {
        const encodedQuery = encodeURIComponent(query);
        let url = `/search/rag?query=${encodedQuery}&limit=${limit}&temperature=${temperature}`;
        if (llmProvider) url += `&llm_provider=${llmProvider}`;
        if (model) url += `&model=${encodeURIComponent(model)}`;
        if (apiBase) url += `&api_base=${encodeURIComponent(apiBase)}`;
        
        const result = await callMCPServer(url);
        console.log(`RAG回答生成完了（モデル: ${result.model_used}, プロバイダー: ${result.provider_used}）`);
        return result;
    } catch (error) {
        console.error('RAG回答生成エラー:', error);
        throw error;
    }
}

async function getLLMModels(apiBase: string) {
    try {
        const encodedApiBase = encodeURIComponent(apiBase);
        const result = await callMCPServer(`/search/llm/models?api_base=${encodedApiBase}`);
        console.log(`利用可能なモデル数: ${result.total}`);
        return result.models;
    } catch (error) {
        console.error('モデルリスト取得エラー:', error);
        throw error;
    }
}

async function getJobStatus(jobId: number) {
    try {
        const result = await callMCPServer(`/search/jobs/${jobId}`);
        console.log(`ジョブステータス: ${result.status}`);
        console.log(`進捗: ${result.progress.current}/${result.progress.total} (${result.progress.percentage}%)`);
        return result;
    } catch (error) {
        console.error('ジョブ取得エラー:', error);
        throw error;
    }
}

async function cancelJob(jobId: number) {
    try {
        const result = await postToMCPServer(`/search/jobs/${jobId}/cancel`, {});
        console.log('ジョブをキャンセルしました:', result.message);
        return result;
    } catch (error) {
        console.error('ジョブキャンセルエラー:', error);
        throw error;
    }
}
```

### プラグインの設定ファイル例（manifest.json）

```json
{
  "id": "your-plugin-id",
  "name": "Your Plugin Name",
  "version": "0.1.0",
  "minAppVersion": "0.15.0",
  "description": "MCPサーバーと連携するプラグイン",
  "author": "Your Name",
  "authorUrl": "",
  "fundingUrl": "",
  "isDesktopOnly": false
}
```

### プラグインのメインファイル例（main.ts）

```typescript
import { Plugin, Notice } from 'obsidian';

export default class MyPlugin extends Plugin {
    private readonly MCP_SERVER_URL = 'http://127.0.0.1:8000';

    async onload() {
        // サーバー接続確認
        await this.checkServerConnection();

        // コマンドの追加例
        this.addCommand({
            id: 'call-mcp-server',
            name: 'MCPサーバーを呼び出す',
            callback: async () => {
                await this.callMCPServer();
            },
        });

        // インデックス作成コマンド
        this.addCommand({
            id: 'create-index',
            name: 'インデックスを作成',
            callback: async () => {
                const vaultPath = this.app.vault.adapter.basePath;
                await this.createIndex(vaultPath);
            },
        });

        // 検索コマンド
        this.addCommand({
            id: 'search-documents',
            name: 'ドキュメントを検索',
            callback: async () => {
                // 検索ダイアログを表示してクエリを入力
                const query = await this.showSearchDialog();
                if (query) {
                    await this.searchDocuments(query);
                }
            },
        });

        // ベクトル化コマンド
        this.addCommand({
            id: 'vectorize-directory',
            name: 'ディレクトリをベクトル化',
            callback: async () => {
                const vaultPath = this.app.vault.adapter.basePath;
                await this.vectorizeDirectory(vaultPath);
            },
        });

        // ハイブリッド検索コマンド
        this.addCommand({
            id: 'hybrid-search',
            name: 'ハイブリッド検索',
            callback: async () => {
                const query = await this.showSearchDialog();
                if (query) {
                    await this.hybridSearch(query);
                }
            },
        });

        // RAG回答生成コマンド
        this.addCommand({
            id: 'rag-query',
            name: 'RAGで質問に回答',
            callback: async () => {
                const query = await this.showSearchDialog();
                if (query) {
                    await this.ragQuery(query);
                }
            },
        });
    }

    async checkServerConnection() {
        try {
            const response = await fetch(`${this.MCP_SERVER_URL}/health`);
            if (response.ok) {
                const data = await response.json();
                new Notice(`MCPサーバーに接続しました: ${data.status}`);
            } else {
                new Notice('MCPサーバーに接続できませんでした');
            }
        } catch (error) {
            new Notice('MCPサーバーが起動していない可能性があります');
            console.error('MCPサーバー接続エラー:', error);
        }
    }

    async callMCPServer() {
        try {
            const response = await fetch(`${this.MCP_SERVER_URL}/`);
            const data = await response.json();
            new Notice(data.message);
        } catch (error) {
            new Notice('MCPサーバーへのリクエストが失敗しました');
            console.error('エラー:', error);
        }
    }

    async createIndex(directoryPath: string) {
        try {
            const response = await fetch(`${this.MCP_SERVER_URL}/search/index`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    directory_path: directoryPath,
                    clear_existing: false
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            new Notice(`インデックス作成を開始しました（ジョブID: ${data.job_id}）`);
            
            // 進捗を監視
            this.monitorJobProgress(data.job_id);
        } catch (error) {
            new Notice('インデックス作成に失敗しました');
            console.error('エラー:', error);
        }
    }

    async monitorJobProgress(jobId: number) {
        const checkInterval = setInterval(async () => {
            try {
                const response = await fetch(`${this.MCP_SERVER_URL}/search/jobs/${jobId}`);
                if (!response.ok) return;

                const job = await response.json();
                const progress = job.progress;

                if (job.status === 'completed') {
                    clearInterval(checkInterval);
                    new Notice(`インデックス作成が完了しました（${progress.total}ファイル）`);
                } else if (job.status === 'failed') {
                    clearInterval(checkInterval);
                    new Notice(`インデックス作成が失敗しました: ${job.error_message}`);
                } else if (job.status === 'processing') {
                    console.log(`進捗: ${progress.current}/${progress.total} (${progress.percentage}%)`);
                }
            } catch (error) {
                console.error('進捗確認エラー:', error);
            }
        }, 2000); // 2秒ごとに確認
    }

    async searchDocuments(query: string) {
        try {
            const encodedQuery = encodeURIComponent(query);
            const response = await fetch(`${this.MCP_SERVER_URL}/search/query?query=${encodedQuery}&limit=20`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log(`検索結果: ${data.total}件`);
            
            // 検索結果を表示（ここではコンソールに出力）
            data.results.forEach((result: any, index: number) => {
                console.log(`${index + 1}. ${result.file_path}`);
                console.log(`   ${result.snippet}`);
            });

            new Notice(`${data.total}件の検索結果が見つかりました`);
        } catch (error) {
            new Notice('検索に失敗しました');
            console.error('エラー:', error);
        }
    }

    async showSearchDialog(): Promise<string | null> {
        // 簡易的な検索ダイアログ（実際の実装では、より高度なUIを使用することを推奨）
        return prompt('検索キーワードを入力してください:');
    }

    async vectorizeDirectory(directoryPath: string) {
        try {
            const response = await fetch(`${this.MCP_SERVER_URL}/search/vectorize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    directory_path: directoryPath,
                    chunk_size: 512,
                    chunk_overlap: 50
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            new Notice(`ベクトル化を開始しました（ジョブID: ${data.job_id}）`);
            
            // 進捗を監視
            this.monitorJobProgress(data.job_id);
        } catch (error) {
            new Notice('ベクトル化に失敗しました');
            console.error('エラー:', error);
        }
    }

    async hybridSearch(query: string, hybridWeight: number = 0.5) {
        try {
            const encodedQuery = encodeURIComponent(query);
            const response = await fetch(
                `${this.MCP_SERVER_URL}/search/hybrid?query=${encodedQuery}&limit=20&hybrid_weight=${hybridWeight}`
            );
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log(`ハイブリッド検索結果: ${data.total}件`);
            
            // 検索結果を表示
            data.results.forEach((result: any, index: number) => {
                console.log(`${index + 1}. ${result.file_path}`);
                console.log(`   ${result.snippet}`);
            });

            new Notice(`${data.total}件のハイブリッド検索結果が見つかりました`);
        } catch (error) {
            new Notice('ハイブリッド検索に失敗しました');
            console.error('エラー:', error);
        }
    }

    async ragQuery(query: string, llmProvider?: string, model?: string) {
        try {
            const encodedQuery = encodeURIComponent(query);
            let url = `${this.MCP_SERVER_URL}/search/rag?query=${encodedQuery}&limit=20`;
            if (llmProvider) url += `&llm_provider=${llmProvider}`;
            if (model) url += `&model=${encodeURIComponent(model)}`;
            
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log(`RAG回答（モデル: ${data.model_used}）:`);
            console.log(data.answer);
            console.log(`ソース: ${data.sources.length}件`);
            
            // 回答を表示（実際の実装では、より良いUIを使用することを推奨）
            new Notice(`RAG回答を生成しました（${data.sources.length}件のソース）`);
            return data;
        } catch (error) {
            new Notice('RAG回答生成に失敗しました');
            console.error('エラー:', error);
        }
    }

    async getLLMModels(apiBase: string) {
        try {
            const encodedApiBase = encodeURIComponent(apiBase);
            const response = await fetch(`${this.MCP_SERVER_URL}/search/llm/models?api_base=${encodedApiBase}`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log(`利用可能なモデル数: ${data.total}`);
            data.models.forEach((model: any) => {
                console.log(`- ${model.id}`);
            });
            
            new Notice(`${data.total}個のモデルが見つかりました`);
            return data.models;
        } catch (error) {
            new Notice('モデルリストの取得に失敗しました');
            console.error('エラー:', error);
        }
    }
}
```

## エラーハンドリング

サーバーが起動していない場合や、ネットワークエラーが発生した場合のエラーハンドリングを行ってください：

```typescript
async function safeCallMCPServer(endpoint: string) {
    try {
        const response = await fetch(`${MCP_SERVER_URL}${endpoint}`);
        
        if (!response.ok) {
            throw new Error(`サーバーエラー: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        if (error instanceof TypeError && error.message.includes('fetch')) {
            // ネットワークエラー（サーバーが起動していない可能性）
            throw new Error('MCPサーバーに接続できません。サーバーが起動しているか確認してください。');
        }
        throw error;
    }
}
```

## 開発時の注意事項

1. **CORS設定**: サーバー側でCORSが有効になっているため、ブラウザからのリクエストも可能です
2. **localhost専用**: 現在の設定は `127.0.0.1` のみでリッスンしているため、同一マシンからのみアクセス可能です
3. **ポート番号**: デフォルトではポート `8000` を使用しています。変更する場合は `main.py` を編集してください

## 技術仕様

### 全文検索エンジン

- **データベース**: SQLite (FTS5拡張機能)
- **日本語解析**: Janome（分かち書き、形態素解析）
- **検索方式**: FTS5のMATCH構文を使用した高速全文検索
- **キュー管理**: 汎用的なジョブキュー管理システム（進捗追跡対応）

### ベクトル検索エンジン

- **ベクトルストア**: ChromaDB（永続化対応）
- **Embeddingプロバイダー**: OpenRouter、AWS Bedrock、LiteLLM対応
- **チャンキング**: オーバーラップ付きテキストチャンキング
- **検索方式**: コサイン類似度によるベクトル検索
- **最適化**: ファイル更新日時チェックによる自動スキップ機能（変更されていないファイルは再ベクトル化しない）

### ハイブリッド検索

- **検索方式**: 全文検索（BM25）とベクトル検索の組み合わせ
- **統合アルゴリズム**: RRF（Reciprocal Rank Fusion）
- **クエリ前処理**: 
  - 形態素解析（Janome）
  - ストップワード除去
  - キーワード抽出（名詞、動詞、形容詞）
  - 類義語展開（オプション）
- **重み付け**: 全文検索とベクトル検索の重要度を調整可能（`hybrid_weight`パラメータ）

### RAG（Retrieval-Augmented Generation）

- **LLMプロバイダー**: OpenRouter、LiteLLM対応
- **デフォルトモデル**:
  - OpenRouter: `google/gemini-3-flash-preview`
  - LiteLLM: 環境変数`LITELLM_MODEL`で指定可能（デフォルト: `gpt-3.5-turbo`）
- **処理フロー**: ハイブリッド検索 → コンテキスト構築 → LLM回答生成
- **モデル選択**: API経由で利用可能なモデルリストを取得可能（LiteLLM）

### 最適化機能

- **インデックス作成**: ファイル更新日時をチェックし、変更されていないファイルはスキップ
- **ベクトル化**: ファイル更新日時をチェックし、変更されていないファイルはスキップ（処理時間とAPIコストを削減）

### ジョブステータス

ジョブは以下のステータスを持ちます：

- `pending`: 待機中
- `processing`: 処理中
- `completed`: 完了
- `failed`: 失敗
- `cancelled`: キャンセル済み

### 進捗情報

進捗情報には以下の情報が含まれます：

```json
{
  "current": 45,
  "total": 100,
  "percentage": 45.0,
  "message": "処理中: document.pdf"
}
```

## 実装のヒント

### インデックス作成の進捗監視

長時間かかるインデックス作成処理の場合、定期的にジョブステータスを確認することを推奨します：

```typescript
async function waitForJobCompletion(jobId: number): Promise<any> {
    while (true) {
        const job = await getJobStatus(jobId);
        
        if (job.status === 'completed') {
            return job.result;
        } else if (job.status === 'failed') {
            throw new Error(job.error_message);
        } else if (job.status === 'cancelled') {
            throw new Error('ジョブがキャンセルされました');
        }
        
        // 2秒待機してから再確認
        await new Promise(resolve => setTimeout(resolve, 2000));
    }
}
```

### エラーハンドリング

すべてのAPIリクエストで適切なエラーハンドリングを実装してください：

```typescript
async function safeApiCall<T>(
    apiCall: () => Promise<T>,
    errorMessage: string
): Promise<T | null> {
    try {
        return await apiCall();
    } catch (error) {
        console.error(`${errorMessage}:`, error);
        new Notice(`${errorMessage}: ${error.message}`);
        return null;
    }
}
```

## 次のステップ

MCPサーバーに新しいエンドポイントを追加する場合は、`main.py` を編集してください。新しいエンドポイントが追加されたら、このドキュメントも更新してください。

## サポート

問題が発生した場合：
1. サーバーが起動しているか確認（`start.bat`を実行）
2. ポート8000が使用可能か確認
3. ブラウザで `http://127.0.0.1:8000/docs` にアクセスしてサーバーが正常に動作しているか確認
4. インデックス作成の進捗は `http://127.0.0.1:8000/task/create_index` で確認できます
5. ベクトル化の進捗は `http://127.0.0.1:8000/task/create_vector` で確認できます


