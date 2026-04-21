import asyncio
import json
import os
from typing import Dict, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Sử dụng singleton cho client để tiết kiệm connect
try:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    client = None

class MultiModelJudge:
    def __init__(self):
        self.model_1 = "gpt-4o-mini"
        self.model_2 = "gpt-4o-mini" # Do hạn chế tài nguyên, dùng GPT-4o-mini gánh vai 2 persona
        
    async def judge_single_model(self, persona: str, question: str, answer: str, ground_truth: str) -> int:
        prompt = f"""
        Bạn là {persona}. Hãy chấm điểm câu trả lời của AI Agent so với đáp án kỳ vọng.
        Thang điểm từ 1 đến 5 (1=sai hoàn toàn, 5=chính xác hoàn toàn).
        - Câu hỏi: {question}
        - Câu trả lời của Agent: {answer}
        - Đáp án đúng (Ground Truth): {ground_truth}
        
        Trả về JSON với cấu trúc duy nhất: {{"score": <điểm_số_nguyên_từ_1_đến_5>}}
        """
        try:
            if not client: return 3
            res = await client.chat.completions.create(
                model=self.model_1,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3 if "nghiêm khắc" in persona else 0.7
            )
            content = res.choices[0].message.content
            return int(json.loads(content).get("score", 3))
        except Exception as e:
            return 3

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        EXPERT TASK: Gọi ít nhất 2 model. Tính toán sự sai lệch và đồng thuận.
        """
        score_strict = self.judge_single_model("một giám khảo cực kỳ khắt khe, hay trừ điểm", question, answer, ground_truth)
        score_lenient = self.judge_single_model("một giám khảo cực kỳ dễ tính, thích cho điểm cao", question, answer, ground_truth)
        
        s1, s2 = await asyncio.gather(score_strict, score_lenient)
        
        # Clip [1, 5]
        s1 = max(1, min(5, s1))
        s2 = max(1, min(5, s2))
        
        avg_score = (s1 + s2) / 2
        # Được gọi là "đồng thuận" nếu chênh lệch <= 1 điểm (tức là cùng rate pass/fail hoặc tương đương)
        agreement = 1.0 if abs(s1 - s2) <= 1 else 0.0
        
        return {
            "final_score": avg_score,
            "agreement_rate": agreement,
            "individual_scores": {"judge_strict": s1, "judge_lenient": s2},
            "reasoning": f"Strict: {s1}/5. Lenient: {s2}/5."
        }
