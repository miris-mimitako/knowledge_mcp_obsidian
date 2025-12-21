"""
クエリ拡張モジュール
類義語展開による検索クエリの拡張機能
"""
import os
from typing import List, Dict, Optional
from .embedding_providers import EmbeddingProvider, EmbeddingProviderType, create_embedding_provider


class QueryExpansion:
    """クエリ拡張クラス（類義語展開）"""
    
    def __init__(self, embedding_provider: Optional[EmbeddingProvider] = None):
        """
        クエリ拡張を初期化
        
        Args:
            embedding_provider: Embeddingプロバイダー（Noneの場合は自動生成）
        """
        if embedding_provider is None:
            provider_type_str = os.environ.get("EMBEDDING_PROVIDER", "openrouter")
            try:
                provider_type = EmbeddingProviderType(provider_type_str)
            except ValueError:
                provider_type = EmbeddingProviderType.OPENROUTER
            self.embedding_provider = create_embedding_provider(provider_type)
        else:
            self.embedding_provider = embedding_provider
        
        # 簡易的な類義語辞書（必要に応じて拡張可能）
        self.synonym_dict: Dict[str, List[str]] = {}
    
    def expand_with_llm(self, keywords: List[str], max_synonyms: int = 3) -> List[str]:
        """
        LLMを使用して類義語を生成（将来の拡張用）
        
        注意: 現在は簡易実装。本格的な実装では、LLM APIを使用して
        各キーワードの類義語を生成する必要があります。
        
        Args:
            keywords: 拡張するキーワードのリスト
            max_synonyms: 各キーワードに対して生成する最大類義語数
        
        Returns:
            元のキーワード + 類義語のリスト
        """
        # TODO: LLM APIを使用した類義語生成を実装
        # 現在は元のキーワードをそのまま返す
        return keywords
    
    def expand_with_dict(self, keywords: List[str]) -> List[str]:
        """
        辞書ベースの類義語展開
        
        Args:
            keywords: 拡張するキーワードのリスト
        
        Returns:
            元のキーワード + 類義語のリスト
        """
        expanded = set(keywords)  # 元のキーワードを含める
        
        for keyword in keywords:
            # 辞書から類義語を検索
            if keyword in self.synonym_dict:
                expanded.update(self.synonym_dict[keyword])
        
        return list(expanded)
    
    def expand(self, keywords: List[str], use_llm: bool = False) -> List[str]:
        """
        キーワードを拡張（類義語展開）
        
        Args:
            keywords: 拡張するキーワードのリスト
            use_llm: LLMを使用するかどうか（将来の拡張用）
        
        Returns:
            拡張されたキーワードのリスト
        """
        if not keywords:
            return []
        
        # 辞書ベースの拡張を実行
        expanded = self.expand_with_dict(keywords)
        
        # LLMを使用する場合は追加で拡張
        if use_llm:
            llm_expanded = self.expand_with_llm(keywords)
            expanded = list(set(expanded + llm_expanded))
        
        return expanded
    
    def add_synonym(self, word: str, synonyms: List[str]):
        """
        類義語辞書にエントリを追加
        
        Args:
            word: 元の単語
            synonyms: 類義語のリスト
        """
        if word not in self.synonym_dict:
            self.synonym_dict[word] = []
        self.synonym_dict[word].extend(synonyms)
        # 重複を除去
        self.synonym_dict[word] = list(set(self.synonym_dict[word]))

