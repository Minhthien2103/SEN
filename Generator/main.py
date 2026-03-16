from core import ObjectGenerationBackend
import os
from dotenv import load_dotenv
import time

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
HF_KEY = os.getenv("HUGGINGFACE_API_KEY")
if __name__ == "__main__":   
    gen_module = ObjectGenerationBackend(gemini_api_key=GEMINI_KEY, hf_api_key=HF_KEY)
    
    try:
        with open("Qua_Tao.jpg", "rb") as f:
            dummy_unity_bytes = f.read()
            
        response_json = gen_module.process_scan_request(dummy_unity_bytes)
        
        print("\n--- JSON GỬI CHO UNITY ---")
        print(response_json)
        
        print("\n[HỆ THỐNG] Đang chờ luồng chạy ngầm hoàn tất (có thể mất 15-30s tùy server HF)...")
        time.sleep(30) 
        print("[HỆ THỐNG] Đã tắt toàn bộ!")
        
    except FileNotFoundError:
        print("LỖI: Hãy chuẩn bị 1 tấm ảnh tên là 'Qua_Tao.jpg' để test.")