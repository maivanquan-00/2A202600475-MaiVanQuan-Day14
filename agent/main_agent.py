import asyncio
import os
import numpy as np
from typing import List, Dict
from pypdf import PdfReader
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class MainAgent:
    def __init__(self):
        self.name = "GPT-Romance-RAG"
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chunks = []
        self.embeddings = None
        self.ready = False
        self.lock = asyncio.Lock()

    async def prepare_db(self):
        async with self.lock:
            if self.ready: return
            print("\n[Agent RAG] Đang đọc tiểu thuyết và khởi tạo Vector DB...")
            pdf_path = "Tỉnh Giấc Tan Mộng Người Bên Gối Đã Không Còn - Phán Duyệt Tây.pdf"
            if not os.path.exists(pdf_path):
                print("[Agent RAG] Lỗi: Không tìm thấy file PDF!")
                return
                
            reader = PdfReader(pdf_path)
            current_chunk = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    for i in range(0, len(text), 1000):
                        current_chunk += text[i:i+1000]
                        if len(current_chunk) >= 1000:
                            self.chunks.append(current_chunk)
                            current_chunk = ""
                            
            if current_chunk:
                self.chunks.append(current_chunk)
            
            try:
                # Chia batch embed để tránh limit nếu có
                batch_size = 500
                all_embeddings = []
                for i in range(0, len(self.chunks), batch_size):
                    batch = self.chunks[i:i+batch_size]
                    res = await self.client.embeddings.create(input=batch, model="text-embedding-3-small")
                    all_embeddings.extend([data.embedding for data in res.data])
                
                self.embeddings = np.array(all_embeddings)
                self.ready = True
                print(f"[Agent RAG] ✅ Embed {len(self.chunks)} chunks thành công.\n")
            except Exception as e:
                print(f"[Agent RAG] Lỗi Embed: {e}")

    async def query(self, question: str) -> Dict:
        if not self.ready:
            await self.prepare_db()

        contexts = []
        if not self.ready or self.embeddings is None:
            return {
                "answer": "Hệ thống RAG bị lỗi khởi tạo hoặc thiếu PDF.",
                "contexts": [],
                "metadata": {"model": "gpt-4o-mini", "vector_engine": "numpy_in_memory"}
            }

        try:
            # 1. Embed câu hỏi
            q_res = await self.client.embeddings.create(input=[question], model="text-embedding-3-small")
            q_emb = np.array(q_res.data[0].embedding)
            
            # 2. Tính Cosine Similarity
            similarities = np.dot(self.embeddings, q_emb) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb))
            
            # 3. Lấy Top 3 chunks
            k = min(3, len(similarities))
            top_k_idx = np.argsort(similarities)[-k:][::-1]
            contexts = [self.chunks[i] for i in top_k_idx]
            context_text = "\n---\n".join(contexts)
            
            # 4. Generate câu trả lời
            prompt = f"Bạn là AI hỗ trợ đọc truyện tiểu thuyết. Dựa vào đoạn trích gốc sau đây, trả lời câu hỏi một cách chính xác và bám sát nguyên tác. Nếu không có trích đoạn, bạn có thể dựa vào hiểu biết chung, nhưng khuyến khích ưu tiên trích đoạn trước.\n\n[Trích đoạn nguyên tác]:\n{context_text}\n\n[Câu hỏi từ User]: {question}"
            
            res = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            answer = res.choices[0].message.content
        except Exception as e:
            answer = f"Lỗi hệ thống: {e}"

        return {
            "answer": answer,
            "contexts": contexts,
            "metadata": {"model": "gpt-4o-mini", "vector_engine": "numpy_in_memory"}
        }

if __name__ == "__main__":
    agent = MainAgent()
    async def test():
        ans = await agent.query("Truyện kết thúc thế nào?")
        print(ans["answer"])
    asyncio.run(test())
