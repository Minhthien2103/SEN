# 📑 QUY ĐỊNH GIAO TIẾP DỮ LIỆU (API SPEC)

Để hệ thống không bị lỗi khi ráp nối, yêu cầu 3 ông tuân thủ đúng định dạng JSON sau:

## 1. Luồng dữ liệu: Unity -> Python (Sự kiện: `unity_to_bot`)
Khi Unity gửi dữ liệu sang, phải bọc trong JSON:
{
    "type": "voice" | "image",
    "payload": "chuỗi Base64 dữ liệu âm thanh hoặc hình ảnh"
}

## 2. Luồng dữ liệu: Python -> Unity (Sự kiện: `bot_to_unity`)
Khi Python xử lý xong và trả về, bắt buộc phải có đủ các trường:
{
    "text": "Câu trả lời của SEN",
    "emotion": "happy" | "sad" | "thinking" | "surprised",
    "action": "wave" | "point_at_object" | "none",
    "audio_url": "link_to_tts_file_if_any"
}

*Lưu ý: Nếu thiếu bất kỳ trường nào, Unity sẽ không thể hiển thị nhân vật đúng cách.*