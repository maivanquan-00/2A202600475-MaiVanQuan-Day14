import json
import asyncio
import os
from pypdf import PdfReader
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def generate_qa_batch(chunk: str, batch_size: int = 5):
    prompt = f"""
    Đọc trích đoạn truyện sau và tạo {batch_size} câu hỏi QA (hỏi/đáp) liên quan đến nội dung truyện.
    Trích đoạn để tạo câu hỏi:
    \"\"\"{chunk[:3000]}\"\"\"
    
    Phản hồi BẮT BUỘC phải là đối tượng JSON (JSON object) với cấu trúc sau:
    {{
      "data": [
        {{
          "question": "Nội dung câu hỏi...",
          "expected_answer": "Câu trả lời đúng dựa vào nội dung...",
          "context": "Trích đoạn hỗ trợ câu trả lời...",
          "metadata": {{
            "difficulty": "easy/medium/hard",
            "type": "plot"
          }}
        }}
        # Lặp lại {batch_size} lần
      ]
    }}
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ngôi sao tạo dữ liệu JSON, luôn trả về JSON hợp lệ."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        return parsed.get("data", [])
    except Exception as e:
        print(f"Lỗi API: {e}")
        return []

async def main():
    pdf_path = "Tỉnh Giấc Tan Mộng Người Bên Gối Đã Không Còn - Phán Duyệt Tây.pdf"
    if not os.path.exists(pdf_path):
        print(f"File PDF '{pdf_path}' không tồn tại trong thư mục.")
        return

    print("Đang đọc file PDF...")
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n"
    except Exception as e:
        print(f"Lỗi khi đọc file PDF: {e}")
        return
        
    print(f"Tổng số ký tự trong PDF: {len(text)}")
    
    # Chia file PDF làm 10 chunk để gửi 10 request song song (mỗi request lấy 5 câu = 50 câu)
    chunk_size = max(len(text) // 10, 500)
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)][:10]
    # Đảm bảo có đủ 10 chunk:
    while len(chunks) < 10:
        chunks.append(text[:chunk_size] if chunk_size > 0 else "")

    print(f"Bắt đầu gọi 10 request API song song (sử dụng model 'gpt-4o-mini')...")
    
    tasks = [generate_qa_batch(chunk, 5) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    
    all_qa = []
    for res in results:
        if isinstance(res, list):
            all_qa.extend(res)
            
    # Lấy đúng 50 câu hỏi
    all_qa = all_qa[:50]
    
    print(f"Số lượng QA tạo thành công: {len(all_qa)}")
    
    os.makedirs("data", exist_ok=True)
    out_file = "data/golden_set.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for qa in all_qa:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
            
    print(f"Đã lưu nội dung vào file 1 cách an toàn (UTF-8) tại: {out_file}")

if __name__ == "__main__":
    asyncio.run(main())
