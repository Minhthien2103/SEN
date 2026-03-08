# SEN
SEN_Project/
├── backend/                
│   ├── modules/            
│   │   ├── speech.py       
│   │   ├── vision.py       
│   │   └── brain.py        
│   ├── models/             
│   ├── main.py             
│   ├── requirements.txt    
│   └── Dockerfile         
├── unity_client/          
│   ├── Assets/
│   │   ├── Scripts/        
│   │   ├── Plugins/        # Thư viện Socket.io cho Unity
│   │   └── Resources/      # Nhân vật Live2D/Spine
├── docs/                   # [Tài liệu chung]
│   ├── api_spec.md         # Quy định dữ liệu JSON 
│   └── setup_gpu.md        # Hướng dẫn cài CUDA cho máy 4060
└── docker-compose.yml      # File chạy cả hệ thống 
