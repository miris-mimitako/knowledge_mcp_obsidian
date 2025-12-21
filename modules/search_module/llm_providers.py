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


def create_llm_provider(
    provider_type: LLMProviderType,
    model: Optional[str] = None,
    api_base: Optional[str] = None
) -> LLMProvider:
    """
    LLMプロバイダーを作成
    
    Args:
        provider_type: プロバイダーのタイプ
        model: 使用するモデル名（オプション）
        api_base: LiteLLMのカスタムエンドポイントURL（litellmプロバイダーの場合のみ）
    
    Returns:
        LLMProviderインスタンス
    """
    if provider_type == LLMProviderType.OPENROUTER:
        return OpenRouterLLMProvider(model=model or "google/gemini-3-flash-preview")
    elif provider_type == LLMProviderType.LITELLM:
        return LiteLLMLLMProvider(model=model, api_base=api_base)
    else:
        raise ValueError(f"サポートされていないプロバイダータイプ: {provider_type}")

