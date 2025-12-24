"""
LLMプロバイダーモジュール
RAGの回答生成に使用するLLMプロバイダー
"""
import os
import json
import requests
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from enum import Enum


class LLMProviderType(str, Enum):
    """LLMプロバイダーのタイプ"""
    OPENROUTER = "openrouter"
    LITELLM = "litellm"
    AWS_BEDROCK = "aws_bedrock"


class LLMProvider(ABC):
    """LLMプロバイダーの基底クラス"""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        テキストを生成
        
        Args:
            prompt: プロンプト
            system_prompt: システムプロンプト（オプション）
            temperature: 温度パラメータ（0.0-2.0）
            max_tokens: 最大トークン数
        
        Returns:
            生成されたテキスト
        """
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """
        デフォルトモデル名を取得
        
        Returns:
            デフォルトモデル名
        """
        pass


class OpenRouterLLMProvider(LLMProvider):
    """OpenRouter LLMプロバイダー"""
    
    def __init__(
        self,
        model: str = "google/gemini-3-flash-preview",
        api_key: Optional[str] = None,
        site_url: str = "https://knowledge-mcp-obsidian.local",
        site_name: str = "Knowledge MCP Obsidian"
    ):
        """
        OpenRouter LLMプロバイダーを初期化
        
        Args:
            model: 使用するモデル名（デフォルト: google/gemini-3-flash-preview）
            api_key: APIキー（環境変数OPENROUTER_API_KEYから取得可能）
            site_url: HTTP-Refererヘッダー用のURL
            site_name: X-Titleヘッダー用のサイト名
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.site_url = site_url
        self.site_name = site_name
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY環境変数が設定されていません")
    
    def get_default_model(self) -> str:
        """デフォルトモデル名を取得"""
        return "google/gemini-3-flash-preview"
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """テキストを生成"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            response = requests.post(
                url=self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=120
            )
            
            if response.status_code == 401:
                raise ValueError(
                    "OpenRouter APIの認証に失敗しました (401 Unauthorized)。\n"
                    "OPENROUTER_API_KEY環境変数が正しく設定されているか確認してください。"
                )
            
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise ValueError(f"予期しないレスポンス形式: {result}")
                
        except requests.exceptions.RequestException as e:
            raise ValueError(f"OpenRouter APIへのリクエストが失敗しました: {str(e)}") from e


class LiteLLMLLMProvider(LLMProvider):
    """LiteLLM LLMプロバイダー"""
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None
    ):
        """
        LiteLLM LLMプロバイダーを初期化
        
        Args:
            model: 使用するモデル名（Noneの場合はLiteLLMのデフォルトモデルを使用）
            api_key: APIキー（環境変数から取得可能）
            api_base: カスタムエンドポイントURL（環境変数LITELLM_API_BASEから取得可能）
        """
        try:
            import litellm
        except ImportError:
            raise ImportError("litellmがインストールされていません。pip install litellm を実行してください。")
        
        self.litellm = litellm
        self.model = model  # Noneの場合はLiteLLMのデフォルトを使用
        self.api_key = api_key or os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("LITELLM_API_BASE")
    
    def get_default_model(self) -> str:
        """デフォルトモデル名を取得（LiteLLMのデフォルト）"""
        # モデルが指定されている場合はそれを使用、そうでなければLiteLLMのデフォルト
        if self.model:
            return self.model
        # LiteLLMのデフォルトモデル（通常はgpt-3.5-turbo）
        # 環境変数から取得を試みる
        default_model = os.environ.get("LITELLM_MODEL", "gpt-3.5-turbo")
        return default_model
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """テキストを生成"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            # リクエストパラメータを構築
            # モデルが指定されていない場合はLiteLLMのデフォルトを使用
            model_to_use = self.model if self.model else os.environ.get("LITELLM_MODEL", "gpt-3.5-turbo")
            
            completion_params = {
                "model": model_to_use,
                "messages": messages,
                "temperature": temperature
            }
            
            if max_tokens:
                completion_params["max_tokens"] = max_tokens
            
            # カスタムエンドポイントが指定されている場合は追加
            if self.api_base:
                completion_params["api_base"] = self.api_base
            
            # APIキーが設定されている場合は追加
            if self.api_key:
                completion_params["api_key"] = self.api_key
            
            response = self.litellm.completion(**completion_params)
            
            # LiteLLMのレスポンス形式
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                raise ValueError(f"予期しないレスポンス形式: {response}")
                
        except Exception as e:
            error_msg = f"LiteLLM completionエラー: {str(e)}"
            if self.api_base:
                error_msg += f"\nエンドポイント: {self.api_base}"
            raise ValueError(error_msg) from e


class AWSBedrockLLMProvider(LLMProvider):
    """AWS Bedrock LLMプロバイダー"""
    
    def __init__(
        self,
        model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None
    ):
        """
        AWS Bedrock LLMプロバイダーを初期化
        
        Args:
            model: 使用するモデルIDまたはARN
                  - モデルID例: "anthropic.claude-3-sonnet-20240229-v1:0"
                  - ARN例: "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
                  - 推論エンドポイントARN例: "arn:aws:bedrock:us-east-1:123456789012:inference-profile/my-endpoint"
            region_name: AWSリージョン名（デフォルト: us-east-1）
            aws_access_key_id: AWSアクセスキーID（環境変数AWS_ACCESS_KEY_IDから取得可能）
            aws_secret_access_key: AWSシークレットキー（環境変数AWS_SECRET_ACCESS_KEYから取得可能）
        """
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3がインストールされていません。pip install boto3 を実行してください。")
        
        self.model = model
        self.region_name = region_name
        
        # boto3クライアントを作成
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=region_name
        )
        self.bedrock_runtime = session.client("bedrock-runtime")
    
    def get_default_model(self) -> str:
        """デフォルトモデル名を取得"""
        return "anthropic.claude-3-sonnet-20240229-v1:0"
    
    def _is_arn(self, model_id: str) -> bool:
        """モデルIDがARNかどうかを判定"""
        return model_id.startswith("arn:aws:bedrock:")
    
    def _prepare_bedrock_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """
        Bedrockリクエストボディを準備
        
        モデルに応じて適切なリクエスト形式を返す
        """
        # Claudeモデルの場合
        if "claude" in self.model.lower() or "anthropic" in self.model.lower():
            # メッセージをClaude形式に変換
            system_message = None
            conversation_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    conversation_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens or 4096,
                "temperature": temperature,
                "messages": conversation_messages
            }
            
            if system_message:
                body["system"] = system_message
            
            return body
        
        # Llamaモデルの場合
        elif "llama" in self.model.lower() or "meta" in self.model.lower():
            # メッセージをテキストに結合
            prompt_parts = []
            for msg in messages:
                if msg["role"] == "system":
                    prompt_parts.append(f"System: {msg['content']}")
                elif msg["role"] == "user":
                    prompt_parts.append(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"Assistant: {msg['content']}")
            
            prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
            
            body = {
                "prompt": prompt,
                "max_gen_len": max_tokens or 2048,
                "temperature": temperature
            }
            
            return body
        
        # Titanモデルの場合
        elif "titan" in self.model.lower() or "amazon" in self.model.lower():
            # メッセージをテキストに結合
            prompt_parts = []
            for msg in messages:
                if msg["role"] == "system":
                    prompt_parts.append(f"System: {msg['content']}")
                elif msg["role"] == "user":
                    prompt_parts.append(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"Assistant: {msg['content']}")
            
            prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
            
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens or 4096,
                    "temperature": temperature
                }
            }
            
            return body
        
        # デフォルト: Claude形式を試す
        else:
            system_message = None
            conversation_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    conversation_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens or 4096,
                "temperature": temperature,
                "messages": conversation_messages
            }
            
            if system_message:
                body["system"] = system_message
            
            return body
    
    def _extract_response(self, response_body: Dict[str, Any]) -> str:
        """Bedrockレスポンスからテキストを抽出"""
        # Claudeモデルの場合
        if "content" in response_body:
            if isinstance(response_body["content"], list) and len(response_body["content"]) > 0:
                return response_body["content"][0].get("text", "")
            elif isinstance(response_body["content"], str):
                return response_body["content"]
        
        # Llamaモデルの場合
        if "generation" in response_body:
            return response_body["generation"]
        
        # Titanモデルの場合
        if "results" in response_body:
            if isinstance(response_body["results"], list) and len(response_body["results"]) > 0:
                return response_body["results"][0].get("outputText", "")
        
        # その他の形式
        if "text" in response_body:
            return response_body["text"]
        
        raise ValueError(f"予期しないレスポンス形式: {response_body}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """テキストを生成"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # リクエストボディを準備
        body = self._prepare_bedrock_request(messages, temperature, max_tokens)
        
        try:
            # モデルIDまたはARNを使用
            model_id = self.model
            
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )
            
            # レスポンスを解析
            response_body = json.loads(response["body"].read())
            return self._extract_response(response_body)
            
        except Exception as e:
            error_msg = f"AWS Bedrock APIへのリクエストが失敗しました: {str(e)}"
            if not (os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_SECRET_ACCESS_KEY")):
                error_msg += "\n※ AWS認証情報が設定されていない可能性があります。"
            raise ValueError(error_msg) from e


def create_llm_provider(
    provider_type: LLMProviderType,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    region_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None
) -> LLMProvider:
    """
    LLMプロバイダーを作成
    
    Args:
        provider_type: プロバイダーのタイプ
        model: 使用するモデル名またはARN（オプション）
        api_base: LiteLLMのカスタムエンドポイントURL（litellmプロバイダーの場合のみ）
        region_name: AWSリージョン名（aws_bedrockプロバイダーの場合のみ）
        aws_access_key_id: AWSアクセスキーID（aws_bedrockプロバイダーの場合のみ）
        aws_secret_access_key: AWSシークレットキー（aws_bedrockプロバイダーの場合のみ）
    
    Returns:
        LLMProviderインスタンス
    """
    if provider_type == LLMProviderType.OPENROUTER:
        return OpenRouterLLMProvider(model=model or "google/gemini-3-flash-preview")
    elif provider_type == LLMProviderType.LITELLM:
        return LiteLLMLLMProvider(model=model, api_base=api_base)
    elif provider_type == LLMProviderType.AWS_BEDROCK:
        kwargs = {}
        if model:
            kwargs["model"] = model
        if region_name:
            kwargs["region_name"] = region_name
        if aws_access_key_id:
            kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            kwargs["aws_secret_access_key"] = aws_secret_access_key
        return AWSBedrockLLMProvider(**kwargs)
    else:
        raise ValueError(f"サポートされていないプロバイダータイプ: {provider_type}")

