import asyncio
import os
import glob
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
            print("\n[Agent RAG] Dang doc file PDF va khoi tao Vector DB...")
            pdfs = glob.glob("*.pdf")
            if not pdfs:
                print("[Agent RAG] Loi: Khong tim thay file PDF!")
                return
            pdf_path = pdfs[0]
                
            reader = PdfReader(pdf_path)
            current_chunk = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    for i in range(0, len(text), 800):
                        current_chunk += text[i:i+800]
                        if len(current_chunk) >= 800:
                            self.chunks.append(current_chunk)
                            current_chunk = ""
                            
            if current_chunk:
                self.chunks.append(current_chunk)
            
            try:
                batch_size = 500
                all_embeddings = []
                for i in range(0, len(self.chunks), batch_size):
                    batch = self.chunks[i:i+batch_size]
                    res = await self.client.embeddings.create(input=batch, model="text-embedding-3-small")
                    all_embeddings.extend([data.embedding for data in res.data])                
                self.embeddings = np.array(all_embeddings)
                self.ready = True
                print(f"[Agent RAG] Embed {len(self.chunks)} chunks thanh cong.\n")
            except Exception as e:
                print(f"[Agent RAG] Loi Embed: {e}")

    async def query(self, question: str) -> Dict:
        if not self.ready:
            await self.prepare_db()

        contexts = []
        if not self.ready or self.embeddings is None:
            return {
                "answer": "He thong RAG bi loi hoac thieu QDF.",
                "contexts": [],
                "metadata": {"model": "gpt-4o-mini", "vector_engine": "numpy_in_memory"}
            }

        try:
            q_res = await self.client.embeddings.create(input=[question], model="text-embedding-3-small")
            q_emb = np.array(q_res.data[0].embedding)
            
            similarities = np.dot(self.embeddings, q_emb) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb))
            
            k = min(5, len(similarities))
            top_k_idx = np.argsort(similarities)[-k:][::-1]
            contexts = [self.chunks[i] for i in top_k_idx]
            context_text = "\n---\n".join(contexts)
            
            prompt = f"Duoi day la cac trich doan tu tieu thuyet. Dua vao doan trich, hay tra loi cau hoi bam sat nguyen tac.\n\n[Trich doan nguyen tac]:\n{context_text}\n\n[Cau hoi tu User]: {question}"
            
            res = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            answer = res.choices[0].message.content
        except Exception as e:
            answer = f"Loi he thong: {e}"

        return {
            "answer": answer,
            "contexts": contexts,
            "metadata": {"model": "gpt-4o-mini", "vector_engine": "numpy_in_memory"}
        }
if __name__ == "__main__":
    agent = MainAgent()
    async def test():
        ans = await agent.query("Truyen ket thuc the nao?")
        print(ans["answer"])
    asyncio.run(test())
