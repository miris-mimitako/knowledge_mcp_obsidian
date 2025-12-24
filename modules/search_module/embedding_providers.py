"""
Embeddingプロバイダーモジュール
Strategyパターンで複数のEmbeddingプロバイダーをサポート
"""
import os
import json
import requests
from abc import ABC, abstractmethod
from typing import List, Optional
from enum import Enum


class EmbeddingProviderType(str, Enum):
    """Embeddingプロバイダーのタイプ"""
    OPENROUTER = "openrouter"
    AWS_BEDROCK = "aws_bedrock"
    LITELLM = "litellm"


class EmbeddingProvider(ABC):
    """Embeddingプロバイダーの基底クラス"""
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        テキストをベクトル化
        
        Args:
            text: ベクトル化するテキスト
        
        Returns:
            ベクトル（浮動小数点数のリスト）
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        複数のテキストをベクトル化（バッチ処理）
        
        Args:
            texts: ベクトル化するテキストのリスト
        
        Returns:
            ベクトルのリスト
        """
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """
        ベクトルの次元数を取得
        
        Returns:
            ベクトルの次元数
        """
        pass


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """OpenRouter Embeddingプロバイダー"""
    
    def __init__(
        self,
        model: str = "qwen/qwen3-embedding-8b",
        api_key: Optional[str] = None,
        site_url: str = "https://knowledge-mcp-obsidian.local",
        site_name: str = "Knowledge MCP Obsidian"
    ):
        """
        OpenRouter Embeddingプロバイダーを初期化
        
        Args:
            model: 使用するモデル名
            api_key: APIキー（環境変数OPENROUTER_API_KEYから取得可能）
            site_url: HTTP-Refererヘッダー用のURL
            site_name: X-Titleヘッダー用のサイト名
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.site_url = site_url
        self.site_name = site_name
        self.endpoint = "https://openrouter.ai/api/v1/embeddings"
        self.dimensions = 4096  # qwen3-embedding-8bの次元数
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY環境変数が設定されていません")
    
    def get_embedding(self, text: str) -> List[float]:
        """単一テキストをベクトル化"""
        return self.get_embeddings([text])[0]
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """複数のテキストをベクトル化"""
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY環境変数が設定されていません。\n"
                "環境変数を設定するか、APIキーを提供してください。"
            )
        
        try:
            response = requests.post(
                url=self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "input": texts if len(texts) > 1 else texts[0]
                },
                timeout=60
            )
            
            # HTTPステータスコードをチェック
            if response.status_code == 401:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get("error", {}).get("message", error_detail)
                except:
                    pass
                
                raise ValueError(
                    f"OpenRouter APIの認証に失敗しました (401 Unauthorized)。\n"
                    f"エラー詳細: {error_detail}\n\n"
                    f"対処方法:\n"
                    f"1. OPENROUTER_API_KEY環境変数が正しく設定されているか確認してください\n"
                    f"2. APIキーが有効か確認してください（https://openrouter.ai/keys で確認）\n"
                    f"3. APIキーに十分なクレジットがあるか確認してください"
                )
            elif response.status_code == 403:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get("error", {}).get("message", error_detail)
                except:
                    pass
                
                raise ValueError(
                    f"OpenRouter APIへのアクセスが拒否されました (403 Forbidden)。\n"
                    f"エラー詳細: {error_detail}\n\n"
                    f"モデル '{self.model}' へのアクセス権限がない可能性があります。"
                )
            
            response.raise_for_status()
            result = response.json()
            
            # レスポンス形式の確認
            if "data" in result:
                # 複数のテキストの場合
                embeddings = [item["embedding"] for item in result["data"]]
            elif "embedding" in result:
                # 単一テキストの場合
                embeddings = [result["embedding"]]
            else:
                raise ValueError(f"予期しないレスポンス形式: {result}")
            
            return embeddings
            
        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.HTTPError):
                raise
            raise ValueError(
                f"OpenRouter APIへのリクエストが失敗しました: {str(e)}\n"
                f"ネットワーク接続またはAPIキーの設定を確認してください。"
            ) from e
    
    def get_dimensions(self) -> int:
        """ベクトルの次元数を取得"""
        return self.dimensions


class AWSBedrockEmbeddingProvider(EmbeddingProvider):
    """AWS Bedrock Embeddingプロバイダー"""
    
    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v1",
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None
    ):
        """
        AWS Bedrock Embeddingプロバイダーを初期化
        
        Args:
            model_id: 使用するBedrockモデルIDまたはARN
                    - モデルID例: "amazon.titan-embed-text-v1"
                    - ARN例: "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1"
                    - 推論エンドポイントARN例: "arn:aws:bedrock:us-east-1:123456789012:inference-profile/my-endpoint"
            region_name: AWSリージョン名
            aws_access_key_id: AWSアクセスキーID（環境変数から取得可能）
            aws_secret_access_key: AWSシークレットキー（環境変数から取得可能）
        """
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3がインストールされていません。pip install boto3 を実行してください。")
        
        self.model_id = model_id
        self.region_name = region_name
        
        # boto3クライアントを作成
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=region_name
        )
        self.bedrock_runtime = session.client("bedrock-runtime")
        
        # モデルごとの次元数（必要に応じて拡張）
        self.dimensions_map = {
            "amazon.titan-embed-text-v1": 1536,
            "amazon.titan-embed-text-v2": 1024,
        }
        self.dimensions = self.dimensions_map.get(model_id, 1536)
    
    def get_embedding(self, text: str) -> List[float]:
        """単一テキストをベクトル化"""
        return self.get_embeddings([text])[0]
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """複数のテキストをベクトル化"""
        embeddings = []
        
        for text in texts:
            # Bedrockのリクエスト形式
            body = json.dumps({"inputText": text})
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            
            result = json.loads(response["body"].read())
            embedding = result.get("embedding", [])
            embeddings.append(embedding)
        
        return embeddings
    
    def get_dimensions(self) -> int:
        """ベクトルの次元数を取得"""
        return self.dimensions


class LiteLLMEmbeddingProvider(EmbeddingProvider):
    """LiteLLM Embeddingプロバイダー"""
    
    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None
    ):
        """
        LiteLLM Embeddingプロバイダーを初期化
        
        Args:
            model: 使用するモデル名（例: "text-embedding-ada-002", "gemini/text-embedding-004"）
            api_key: APIキー（環境変数から取得可能、モデルに応じて適切な環境変数を設定）
            api_base: カスタムエンドポイントURL（環境変数LITELLM_API_BASEから取得可能）
                    例: "http://localhost:4000" または "https://api.example.com/v1"
        """
        try:
            import litellm
        except ImportError:
            raise ImportError("litellmがインストールされていません。pip install litellm を実行してください。")
        
        self.model = model
        self.litellm = litellm
        
        # APIキーを設定（環境変数から取得）
        if api_key:
            self.api_key = api_key
        else:
            # モデルに応じた環境変数からAPIキーを取得
            # OpenAI系の場合
            if model.startswith("text-embedding") or model.startswith("gpt"):
                self.api_key = os.environ.get("OPENAI_API_KEY")
            # その他の場合は、LITELLM_API_KEYまたはモデル名に応じた環境変数を試す
            else:
                self.api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        
        # カスタムエンドポイントを設定
        self.api_base = api_base or os.environ.get("LITELLM_API_BASE")
        
        # モデルごとの次元数（必要に応じて拡張）
        self.dimensions_map = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "gemini/text-embedding-004": 768,
            "voyage-large-2": 1536,
        }
        self.dimensions = self.dimensions_map.get(model, 1536)
    
    def get_embedding(self, text: str) -> List[float]:
        """単一テキストをベクトル化"""
        return self.get_embeddings([text])[0]
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """複数のテキストをベクトル化"""
        try:
            # リクエストパラメータを構築
            embedding_params = {
                "model": self.model,
                "input": texts
            }
            
            # カスタムエンドポイントが指定されている場合は追加
            if self.api_base:
                embedding_params["api_base"] = self.api_base
            
            # APIキーが設定されている場合は追加
            if self.api_key:
                embedding_params["api_key"] = self.api_key
            
            response = self.litellm.embedding(**embedding_params)
            
            # LiteLLMのレスポンス形式
            embeddings = [item["embedding"] for item in response.data]
            return embeddings
            
        except Exception as e:
            error_msg = f"LiteLLM embeddingエラー: {str(e)}"
            if self.api_base:
                error_msg += f"\nエンドポイント: {self.api_base}"
            if not self.api_key:
                error_msg += "\n※ APIキーが設定されていない可能性があります。"
            raise ValueError(error_msg) from e
    
    def get_dimensions(self) -> int:
        """ベクトルの次元数を取得"""
        return self.dimensions


def create_embedding_provider(
    provider_type: EmbeddingProviderType,
    **kwargs
) -> EmbeddingProvider:
    """
    Embeddingプロバイダーを作成
    
    Args:
        provider_type: プロバイダーのタイプ
        **kwargs: プロバイダー固有のパラメータ
    
    Returns:
        EmbeddingProviderインスタンス
    """
    if provider_type == EmbeddingProviderType.OPENROUTER:
        return OpenRouterEmbeddingProvider(**kwargs)
    elif provider_type == EmbeddingProviderType.AWS_BEDROCK:
        return AWSBedrockEmbeddingProvider(**kwargs)
    elif provider_type == EmbeddingProviderType.LITELLM:
        return LiteLLMEmbeddingProvider(**kwargs)
    else:
        raise ValueError(f"サポートされていないプロバイダータイプ: {provider_type}")

