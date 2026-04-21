# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 50
- **Tỉ lệ Pass/Fail:** 24 Pass / 26 Fail
- **Điểm RAGAS trung bình:**
    - Faithfulness: 0.75
    - Relevancy: 0.8
- **Điểm LLM-Judge trung bình:** 2.96 / 5.0

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| Mất ngữ cảnh (Context Miss) | 12 | Top 3 chunk lấy bằng Cosine Similarity không có chứa câu trả lời do trùng lặp từ vựng nhưng khác ngữ nghĩa. |
| Hallucination / Tự bịa | 8 | Retriever lấy sai chunk nhưng LLM vẫn ráng suy diễn (hoặc dùng trí nhớ cá nhân) để trả lời. |
| Mâu thuẫn (Contradiction) | 6 | Trích đoạn có tình tiết nhưng câu văn gây hiểu nhầm, prompt chưa đủ nhắc nhở model bẻ gãy từ đồng âm hoặc lối nhân hoá của truyện ngôn tình. |

## 3. Phân tích 5 Whys (Các case nghiêm trọng)

### Case #1: Quách Phương muốn làm gì khi nhận được đơn ly hôn?
1. **Symptom:** AI Agent đưa câu trả lời về một "thuyết âm mưu 7 ngày", nhưng Giám khảo 1 và 2 đều thống nhất chấm 1/5 điểm do sai quá nghiêm trọng so với Ground Truth.
2. **Why 1:** Agent đọc phải một cụm từ "sau 7 ngày" ở 3 chunks được trả về.
3. **Why 2:** Vector DB (Numpy Dot Product) chấm điểm cao cho một chunk có chữ "Quách Phương" và "trở thành vợ chồng 7 ngày" dù nó không gắn liền với "đơn ly hôn".
4. **Why 3:** Mô hình text-embedding-3-small nhầm lẫn độ ưu tiên ngữ nghĩa đối với từ ngữ ngôn tình "thanh xuân".
5. **Why 4:** Không có Reranker (như BGE-Reranker hoặc Cohere) ở cuối để kiểm duyệt lại ngữ cảnh với câu hỏi.
6. **Root Cause:** Truy xuất Vector chênh lệch ngữ nghĩa của đoạn văn bản dài mà chỉ cắt 1000 kí tự cố định làm đứt mạch tiểu thuyết.

## 4. Kế hoạch cải tiến (Action Plan)
- [x] Đã hoàn thiện Auto-Gate Regression Gate.
- [x] Áp dụng "LLM Multi-Judge" tránh thiên vị, đã thể hiện rõ ràng Agent đang sai bét với điểm 2.96/5.0. 
- [ ] Tính thêm **Semantic Chunking** thay vì chặt thẳng "1000 char" vì truyện ngôn tình hay nối liền nhau.
- [ ] Gắn thêm bước dùng mô hình băm Reranker vào sau bước truy xuất Vectơ để nâng độ chính xác từ 50% lên 90%.
