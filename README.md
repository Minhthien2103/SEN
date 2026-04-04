# 🌸 SEN (Synthetic Emotional Neural companion)
> **The Next-Gen Multimodal Virtual Companion**

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Unity](https://img.shields.io/badge/Unity-2022.3%2B-black.svg)
![AI Models](https://img.shields.io/badge/Models-Whisper%20%7C%20PhoBERT%20%7C%20HuBERT-orange.svg)

**SEN** là một hệ thống trợ lý ảo đa phương thức (Multimodal AI VTuber) hoạt động theo thời gian thực. Dự án tích hợp các pipeline Trí tuệ Nhân tạo tiên tiến (Thị giác máy tính, Xử lý ngôn ngữ tự nhiên, và Tổng hợp giọng nói) kết hợp với môi trường đồ họa Unity để tạo ra một nhân vật ảo có khả năng tương tác, giao tiếp và phản hồi cảm xúc tự nhiên như con người.

---

## ✨ Tính năng nổi bật (Key Features)
* **🗣️ Real-time Speech-to-Speech:** Giao tiếp bằng giọng nói độ trễ thấp thông qua Whisper STT (tăng tốc bằng Groq LPU) và Edge-TTS.
* **🧠 Emotional Intelligence:** Tích hợp mô hình LLM kết hợp phân tích cảm xúc đa phương thức qua giọng nói (HuBERT) và văn bản (PhoBERT).
* **👁️ Computer Vision & Ambient Sensing:** Nhận diện biểu cảm khuôn mặt qua camera (MediaPipe + HSEmotion) và đánh giá môi trường âm thanh (CLAP).
* **🎭 Live2D Lip-sync & UI:** Đồng bộ khẩu hình miệng thời gian thực trên Unity và hệ thống hiển thị Chat Thread-Safe UI.
* **⚡ Event-Driven Architecture:** Hệ thống Client-Server phân tán, giao tiếp bất đồng bộ qua WebSocket (FastAPI/Socket.IO).

---

## 🏗️ Cấu trúc Dự án (Project Structure)

Hệ thống được thiết kế theo mô hình Client-Server, tách biệt rõ ràng giữa logic AI (Backend) và môi trường đồ họa (Frontend):

```text
SEN_Project/
├── backend/                        # 🧠 Python Server (AI Inference & Logic)
│   ├── core/                       # Não bộ trung tâm
│   │   └── brain.py                # Logic gọi LLM & Prompt Injection
│   ├── modules/                    # Các giác quan phân hệ
│   │   ├── audio/                  # Nhóm xử lý Âm thanh
│   │   │   ├── Env_classifier.py   # Nhận diện tiếng ồn môi trường
│   │   │   ├── SER.py              # Phân tích cảm xúc giọng nói
│   │   │   └── speech.py           # STT / TTS Engine
│   │   ├── generator/              # Nhóm Ảo hóa & 3D (R&D)
│   │   │   └── ObjectGenerationBackend.py
│   │   └── vision/                 # Nhóm Thị giác máy tính
│   │       ├── MoodManager.py      # Quản lý trạng thái cảm xúc
│   │       └── SenseVisionBackend.py # Xử lý Camera & Face Tracking
│   ├── tests/                      # Kịch bản kiểm thử độc lập
│   │   └── test_speech.py          
│   ├── training_scripts/           # Source code huấn luyện mô hình (Statistical Learning)
│   │   ├── dataset.py              
│   │   └── fine_tune.py            
│   ├── main_socket.py              # FILE ĐIỀU PHỐI (FastAPI / Socket.IO)
│   └── requirements.txt            # Danh sách thư viện Python tổng hợp
│
├── unity_client/                   # 🎮 Frontend (UI/UX & Character Rendering)
│   ├── Assets/
│   │   ├── Scripts/                # C# Scripts (Socket Client, AudioQueue, UI Threading)
│   │   ├── Plugins/                # Thư viện tích hợp
│   │   └── Resources/              # Assets nhân vật ảo
│
└── docs/                           # 📚 Tài liệu Kỹ thuật
    ├── api_spec.md                 # Đặc tả WebSocket & JSON Payload
    └── setup_gpu.md                # Hướng dẫn cấu hình môi trường