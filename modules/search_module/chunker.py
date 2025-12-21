"""
チャンキングモジュール
テキストを適切なサイズのチャンクに分割
"""
import re
from typing import List, Tuple, Dict, Optional, Any
from .tokenizer import JapaneseTokenizer


class TextChunker:
    """テキストをチャンクに分割するクラス"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        tokenizer: JapaneseTokenizer = None
    ):
        """
        チャンカーを初期化
        
        Args:
            chunk_size: チャンクサイズ（トークン数）
            chunk_overlap: オーバーラップサイズ（トークン数）
            tokenizer: トークナイザー（Noneの場合は簡易カウントを使用）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer or JapaneseTokenizer()
    
    def count_tokens(self, text: str) -> int:
        """
        テキストのトークン数をカウント
        
        Args:
            text: カウントするテキスト
        
        Returns:
            トークン数
        """
        # Janomeを使用してトークン数を概算
        # 実際のトークン数はモデル依存だが、ここでは簡易的に文字数ベースで計算
        # 日本語は1文字≈1トークン、英語は1単語≈1トークン程度として概算
        # より正確にはtiktoken等を使用するが、ここでは簡易実装
        
        # 日本語文字数をカウント
        japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text))
        # 英語単語数をカウント（簡易版）
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        # その他の文字（数字、記号など）
        other_chars = len(text) - japanese_chars - len(re.findall(r'[a-zA-Z\s]', text))
        
        # 概算: 日本語1文字≈1トークン、英語1単語≈1トークン、その他0.5トークン
        estimated_tokens = japanese_chars + english_words + int(other_chars * 0.5)
        
        return max(1, estimated_tokens)
    
    def find_sentence_boundary(self, text: str, start_pos: int, max_pos: int) -> int:
        """
        文の境界を見つける（日本語の句点「。」を優先）
        
        Args:
            text: 対象テキスト
            start_pos: 開始位置
            max_pos: 最大位置（この位置を超えない）
        
        Returns:
            文の境界位置（見つからない場合はmax_pos）
        """
        # max_posまでの範囲で句点を探す
        search_text = text[start_pos:max_pos]
        
        # 日本語の句点「。」を探す
        period_pos = search_text.rfind('。')
        if period_pos != -1:
            return start_pos + period_pos + 1  # 「。」の後ろ
        
        # 英語のピリオドを探す（ただし、小数点やURLの可能性があるので慎重に）
        # スペースの後のピリオドを優先
        period_match = re.search(r'\.\s+', search_text)
        if period_match:
            return start_pos + period_match.end()
        
        # 改行を探す
        newline_pos = search_text.rfind('\n')
        if newline_pos != -1:
            return start_pos + newline_pos + 1
        
        # 見つからない場合はmax_posを返す
        return max_pos
    
    def chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        テキストをチャンクに分割
        
        Args:
            text: 分割するテキスト
        
        Returns:
            (チャンクテキスト, 開始位置, 終了位置)のタプルのリスト
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        current_pos = 0
        text_length = len(text)
        
        while current_pos < text_length:
            # 現在位置からchunk_size分のテキストを取得
            end_pos = min(current_pos + self.chunk_size * 2, text_length)  # 余裕を持たせる
            
            # この範囲内でトークン数を確認しながら適切な境界を見つける
            best_end = current_pos
            best_token_count = 0
            
            # バイナリサーチ的に最適な位置を探す
            search_start = current_pos
            search_end = min(current_pos + self.chunk_size * 3, text_length)
            
            # 段階的に範囲を狭めて最適な位置を探す
            for test_end in range(search_start + self.chunk_size // 2, search_end, 50):
                test_text = text[current_pos:test_end]
                token_count = self.count_tokens(test_text)
                
                if token_count <= self.chunk_size:
                    best_end = test_end
                    best_token_count = token_count
                else:
                    break
            
            # 最適な位置が見つかったら、その位置付近で文の境界を探す
            if best_end > current_pos:
                # 文の境界を探す（best_endを超えない範囲で）
                boundary_pos = self.find_sentence_boundary(text, current_pos, best_end)
                if boundary_pos > current_pos:
                    chunk_end = boundary_pos
                else:
                    chunk_end = best_end
            else:
                # 最適な位置が見つからない場合は強制的にchunk_size分取る
                # ただし、文の境界を優先
                test_end = min(current_pos + self.chunk_size * 2, text_length)
                boundary_pos = self.find_sentence_boundary(text, current_pos, test_end)
                chunk_end = boundary_pos if boundary_pos > current_pos else test_end
            
            # チャンクを取得
            chunk_text = text[current_pos:chunk_end].strip()
            if chunk_text:
                chunks.append((chunk_text, current_pos, chunk_end))
            
            # 次のチャンクの開始位置（オーバーラップを考慮）
            if chunk_end >= text_length:
                break
            
            # オーバーラップ分戻る
            overlap_start = max(current_pos, chunk_end - self.chunk_overlap * 2)
            # オーバーラップ開始位置から文の境界を探す
            next_start = self.find_sentence_boundary(text, overlap_start, chunk_end)
            if next_start <= current_pos:
                # 境界が見つからない場合は、オーバーラップ分だけ戻る
                next_start = max(current_pos + 1, chunk_end - self.chunk_overlap * 2)
            
            current_pos = next_start
        
        return chunks
    
    def chunk_document(
        self,
        content: str,
        file_path: str,
        location_info: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        ドキュメントをチャンクに分割し、メタデータを付与
        
        Args:
            content: ドキュメントの内容
            file_path: ファイルパス
            location_info: 位置情報（ページ番号など）
        
        Returns:
            チャンク情報のリスト（各チャンクにtext, file_path, location_info, chunk_indexを含む）
        """
        chunks_data = []
        text_chunks = self.chunk_text(content)
        
        for idx, (chunk_text, start_pos, end_pos) in enumerate(text_chunks):
            chunks_data.append({
                "text": chunk_text,
                "file_path": file_path,
                "location_info": location_info,
                "chunk_index": idx,
                "total_chunks": len(text_chunks),
                "start_pos": start_pos,
                "end_pos": end_pos
            })
        
        return chunks_data

