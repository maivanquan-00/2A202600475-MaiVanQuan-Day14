from typing import List, Dict

class ExpertEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_id: str, retrieved_texts: List[str]) -> float:
        """
        Hit Rate đơn giản nếu đoạn text retrieve có chứa phần nội dung ground truth context.
        Thực tế thường dùng document IDs, nhưng ở đây agent trả về text nên check overlap % hoặc contains.
        """
        if not expected_id or not retrieved_texts:
            return 0.0
        # Check xem 'expected_id' (ở đây là đoạn context kỳ vọng) có xuất hiện hay không
        for text in retrieved_texts:
            # Overlap check
            if expected_id[:50] in text or text[:50] in expected_id: 
                return 1.0
        return 0.0

    def calculate_mrr(self, expected_id: str, retrieved_texts: List[str]) -> float:
        """Mean Reciprocal Rank"""
        if not expected_id or not retrieved_texts:
            return 0.0
        for i, text in enumerate(retrieved_texts):
            if expected_id[:50] in text or text[:50] in expected_id:
                return 1.0 / (i + 1)
        return 0.0

    async def score(self, case: Dict, response: Dict) -> Dict:
        """
        Giả lập RAGAS context_precision/faithfulness và tính trực tiếp retrieval
        """
        hit_rate = self.calculate_hit_rate(case.get("context", ""), response.get("contexts", []))
        mrr = self.calculate_mrr(case.get("context", ""), response.get("contexts", []))
        
        # Thêm random noise (hoặc RAGAS LLM if needed), bài lab chú trọng logic metrics
        faithfulness = 0.9 if hit_rate > 0 else 0.5
        relevancy = 0.8
        
        return {
            "faithfulness": faithfulness, 
            "relevancy": relevancy,
            "retrieval": {
                "hit_rate": hit_rate, 
                "mrr": mrr
            }
        }
