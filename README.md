# 🌸 SEN: The Next-Gen Multimodal Virtual Companion

Dự án phát triển trợ lý ảo đa phương thức tích hợp AI (Vision, Speech, LLM) và nhân vật tương tác 2D.

## 🏗 Cấu trúc dự án (Project Structure)

```text
SEN_Project/
├── backend/                
│   ├── modules/            # Chứa các file logic xử lý tách biệt
│   │   ├── speech.py       #  (Whisper STT / Edge-TTS)
│   │   ├── vision.py       #  (YOLOv10 / Rembg)
│   │   └── brain.py        # Code Logic & LLM (Kết nối ngôn ngữ)
│   ├── models/             # Nơi lưu trữ file Model (.pt, .onnx, .bin)
│   ├── main.py             # FILE ĐIỀU PHỐI (Socket Server)
│   ├── requirements.txt    # Danh sách thư viện Python tổng hợp
│   └── Dockerfile          # File đóng gói hệ thống
├── unity_client/           #  Giao diện & Nhân vật
│   ├── Assets/
│   │   ├── Scripts/        # Code C# xử lý Socket Client & Logic Unity
│   │   ├── Plugins/        # Thư viện Socket.io (ittai/socket.io-unity)
│   │   └── Resources/      # File nhân vật Live2D/Spine & Animation
├── docs/                   # [Tài liệu kỹ thuật]
│   ├── api_spec.md         # Quy định định dạng JSON truyền tải
│   └── setup_gpu.md        # Hướng dẫn cấu hình CUDA cho RTX 4060
└── docker-compose.yml      # File khởi chạy đồng bộ toàn bộ hệ thống
