"""
日本語トークナイザーモジュール
Janomeを使用した日本語テキストの分かち書き処理
"""
from janome.tokenizer import Tokenizer
from typing import List, Set


class JapaneseTokenizer:
    """日本語テキストを分かち書きするクラス"""
    
    # ストップワード（助詞、助動詞、記号など）
    STOP_WORDS: Set[str] = {
        'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ',
        'ある', 'いる', 'も', 'する', 'から', 'な', 'こと', 'として', 'い',
        'や', 'れる', 'など', 'なっ', 'ない', 'この', 'ため', 'その', 'あの',
        'よう', 'また', 'もの', 'という', 'あり', 'まで', 'られ', 'なる', 'へ',
        'か', 'だ', 'これ', 'によって', 'により', 'おり', 'より', 'による',
        'ず', 'なり', 'られる', 'において', 'における', 'でも', 'など', 'は',
        'ば', 'なら', 'が', 'も', 'と', 'に', 'で', 'へ', 'から', 'まで',
        'より', 'の', 'を', 'について', 'に対して', 'における', 'に関する'
    }
    
    def __init__(self):
        """トークナイザーを初期化"""
        self.tokenizer = Tokenizer()
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        テキストから検索キーワード（名詞、動詞、形容詞）を抽出
        
        Args:
            text: 解析するテキスト
        
        Returns:
            キーワードのリスト（ストップワード除去済み）
        """
        if not text:
            return []
        
        tokens = self.tokenizer.tokenize(text)
        keywords = []
        
        for token in tokens:
            pos = token.part_of_speech.split(',')
            surface = token.surface.strip()
            
            # 空文字や1文字の記号はスキップ
            if not surface or len(surface) < 2:
                continue
            
            # ストップワードを除外
            if surface in self.STOP_WORDS:
                continue
            
            # 品詞フィルタリング：名詞、動詞、形容詞のみ
            if pos[0] == '名詞':
                # 代名詞、非自立、接尾は除外（必要な場合は調整）
                if pos[1] not in ['代名詞', '非自立', '接尾']:
                    keywords.append(surface)
            elif pos[0] == '動詞':
                # 動詞の基本形を取得（必要に応じて活用形を正規化）
                base_form = token.base_form if hasattr(token, 'base_form') else surface
                if base_form not in self.STOP_WORDS:
                    keywords.append(base_form)
            elif pos[0] == '形容詞':
                base_form = token.base_form if hasattr(token, 'base_form') else surface
                if base_form not in self.STOP_WORDS:
                    keywords.append(base_form)
        
        # 重複を除去
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def tokenize(self, text: str) -> str:
        """
        テキストを分かち書きしてスペース区切りの文字列に変換
        
        Args:
            text: 解析するテキスト
        
        Returns:
            スペース区切りで分かち書きされたテキスト
        """
        keywords = self.extract_keywords(text)
        return ' '.join(keywords) if keywords else text

