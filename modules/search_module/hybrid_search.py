"""
ハイブリッド検索モジュール
全文検索とベクトル検索を組み合わせたハイブリッド検索とRRF実装
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HybridSearchResult:
    """ハイブリッド検索結果の統一フォーマット"""
    doc_id: str  # ドキュメントID（file_path + location_infoの組み合わせ）
    file_path: str
    file_type: Optional[str]
    location_info: Optional[str]
    snippet: str
    score: float = 0.0
    rank: int = 0


def reciprocal_rank_fusion(
    keyword_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    k: int = 60,
    alpha: float = 0.5,
    max_results: int = 20
) -> List[HybridSearchResult]:
    """
    Reciprocal Rank Fusion (RRF) による検索結果の統合
    
    Args:
        keyword_results: 全文検索の結果リスト（各要素にdoc_idを含む必要がある）
        vector_results: ベクトル検索の結果リスト（各要素にdoc_idを含む必要がある）
        k: RRFの定数（通常60が推奨される）
        alpha: ベクトル検索の重み (0.0 - 1.0)。0.5なら等価、1.0ならベクトルのみ重視
        max_results: 返却する最大結果数
    
    Returns:
        RRFスコアでソートされた検索結果のリスト
    """
    fused_scores: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    
    # 全文検索のスコア計算（重み 1 - alpha）
    keyword_weight = 1.0 - alpha
    for rank, result in enumerate(keyword_results, start=1):
        doc_id = result.get('doc_id', result.get('file_path', '') + '|' + str(result.get('location_info', '')))
        
        if doc_id not in fused_scores:
            fused_scores[doc_id] = (0.0, result)
        
        # RRFスコアを加算
        rrf_score = keyword_weight * (1.0 / (k + rank))
        fused_scores[doc_id] = (fused_scores[doc_id][0] + rrf_score, fused_scores[doc_id][1])
    
    # ベクトル検索のスコア計算（重み alpha）
    vector_weight = alpha
    for rank, result in enumerate(vector_results, start=1):
        doc_id = result.get('doc_id', result.get('file_path', '') + '|' + str(result.get('location_info', '')))
        
        if doc_id not in fused_scores:
            fused_scores[doc_id] = (0.0, result)
        
        # RRFスコアを加算
        rrf_score = vector_weight * (1.0 / (k + rank))
        fused_scores[doc_id] = (fused_scores[doc_id][0] + rrf_score, fused_scores[doc_id][1])
    
    # スコア順にソート
    sorted_results = sorted(
        fused_scores.items(),
        key=lambda x: x[1][0],
        reverse=True
    )
    
    # HybridSearchResultに変換
    search_results = []
    for rank, (doc_id, (score, result_data)) in enumerate(sorted_results[:max_results], start=1):
        search_result = HybridSearchResult(
            doc_id=doc_id,
            file_path=result_data.get('file_path', ''),
            file_type=result_data.get('file_type'),
            location_info=result_data.get('location_info'),
            snippet=result_data.get('snippet', ''),
            score=score,
            rank=rank
        )
        search_results.append(search_result)
    
    return search_results


def normalize_doc_id(file_path: str, location_info: Optional[str] = None) -> str:
    """
    ドキュメントIDを正規化
    
    Args:
        file_path: ファイルパス
        location_info: 位置情報
    
    Returns:
        正規化されたドキュメントID
    """
    location_str = str(location_info) if location_info else ''
    return f"{file_path}|{location_str}"

