import os
import re
import asyncio
import base64
import time
import socketio
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf

# Import từ các file "não bộ" và "thanh quản" của hệ thống
from backend.modules.audio.speech import SpeechToText, _synthesize, generate_mouth_cues
from backend.core.brain import ResponseGenerator, EmotionPredictor
from backend.modules.vision.SenseVisionBackend import SenseVisionBackend

load_dotenv()

# ---------------------------------------------------------
# 1. KHỞI TẠO CẤU HÌNH & ENGINE
# ---------------------------------------------------------
app = FastAPI()
sio = socketio.AsyncServer(
    async_mode='asgi', 
    cors_allowed_origins='*', 
    max_http_buffer_size=50000000
)
combined_app = socketio.ASGIApp(sio, app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⏳ Đang khởi tạo bộ não và cảm xúc cho SEN...")
stt_module = SpeechToText() 
generator = ResponseGenerator()
emotion_engine = EmotionPredictor() 
stt_module.load_models()
vision_engine = SenseVisionBackend()
print("✅ Khởi tạo hoàn tất!")

# Biến lưu trữ trí nhớ tạm thời cho từng user (Cảm xúc, hình ảnh, trạng thái bận)
user_memory = {}

# ---------------------------------------------------------
# 2. CÁC SỰ KIỆN KẾT NỐI
# ---------------------------------------------------------
@sio.event
async def connect(sid, environ):
    print(f"🟢 [Kết nối] Client {sid} đã vào hệ thống!")
    user_memory[sid] = {"last_vision": None, "is_busy": False} 

@sio.event
async def disconnect(sid):
    print(f"🔴 [Ngắt kết nối] Client {sid} đã rời đi!")
    if sid in user_memory:
        del user_memory[sid]

# ---------------------------------------------------------
# HỨNG EVENT 1: KHUÔN MẶT (XỬ LÝ IM LẶNG)
# ---------------------------------------------------------
@sio.on('client_face_input')
async def handle_face_input(sid, data):
    required_keys = ["user_id", "session_id", "image_base64"]
    if not all(key in data for key in required_keys):
        print("❌ [Lỗi] Unity gửi thiếu dữ liệu khuôn mặt!")
        return 
    
    # Khởi tạo bộ nhớ cho User nếu chưa có (Rất quan trọng để tránh lỗi Key Error)
    if sid not in user_memory:
        user_memory[sid] = {"is_busy": False, "last_audio_time": time.time()}

    image_b64 = data["image_base64"]
    current_time = time.time()
    
    # Đưa ảnh cho AI xử lý ngầm. 
    vision_payload = await asyncio.to_thread(vision_engine.process_base64_frame, image_b64)
    
    # Lấy thông tin từ bộ nhớ của đúng Client này
    last_audio_time = user_memory[sid].get("last_audio_time", current_time)
    is_busy = user_memory[sid].get("is_busy", False)

    # KÍCH HOẠT SEN CHỦ ĐỘNG HỎI THĂM KHI USER IM LẶNG
    if (current_time - last_audio_time >= 60) and not is_busy:
        
        # --- 🛡️ BƯỚC 1: ĐÓNG CỬA KHÓA LUỒNG & RESET ĐỒNG HỒ NGAY LẬP TỨC ---
        user_memory[sid]["is_busy"] = True
        user_memory[sid]["last_audio_time"] = current_time # Bắt đầu đếm lại 1 phút mới
        # -------------------------------------------------------------------

        print(f"\n🧠 [NHẬN THỨC] Phát hiện User im lặng 1 phút:\n{vision_payload}")
        await sio.emit('server_text_reply', {"message": "SEN đang quan sát thấy bạn im lặng..."}, to=sid)
        
        message_id = f"msg_{int(time.time())}"
        chunk_idx = 0
        
        try:
            dummy_user_text = "[Người dùng đang im lặng]"
            hidden_context = f"Hệ thống báo cáo: User đã im lặng 1 phút. Dữ liệu khuôn mặt: {vision_payload}. Hãy chủ động hỏi thăm ngắn gọn, thân thiện dựa trên cảm xúc này. KHÔNG nhắc đến việc bạn đang đọc báo cáo."
            
            text_buffer = ""
            for token in generator.reply_stream(dummy_user_text, hidden_context=hidden_context):
                text_buffer += token
                await asyncio.sleep(0)
                
                if re.search(r'([.,!?:;\n]+)', text_buffer):
                    chunk_text = text_buffer.strip()
                    text_buffer = "" 
                    
                    if chunk_text:
                        print(f"        🗣️ Đang xử lý chunk (Proactive): {chunk_text}")
                        emo_res = await asyncio.to_thread(emotion_engine.predict, chunk_text)
                        dominant_id = emo_res.get("dominant_id", 6)
                        intensity = float(emo_res.get("probs_by_id", {}).get(dominant_id, 0.0))
                        
                        chunk_audio_bytes = await _synthesize(
                            text=chunk_text, voice="vi-VN-HoaiMyNeural", 
                            rate="+30%", volume="+0%", pitch="+0Hz"
                        )
                        audio_b64 = base64.b64encode(chunk_audio_bytes).decode('utf-8')
                        
                        # Rhubarb
                        mouth_cues_data = await asyncio.to_thread(
                            generate_mouth_cues, 
                            chunk_audio_bytes
                        )
                        
                        # JSON -> Unity
                        response_data = {
                            "message_id": message_id,
                            "chunk_index": chunk_idx,
                            "is_final": False,
                            "text_content": chunk_text,
                            "audio_base64": audio_b64,
                            "emotion_data": {
                                "emotionId": dominant_id,
                                "emotionIntensity": intensity
                            },
                            "mouthCues": mouth_cues_data
                        }
                        
                        await sio.emit('server_audio_chunk', response_data, to=sid)
                        chunk_idx += 1

            # XỬ LÝ CHUNK CUỐI CÙNG LỠ BỊ RỚT LẠI
            if text_buffer.strip():
                chunk_text = text_buffer.strip()
                print(f"        🗣️ Đang xử lý chunk cuối (Proactive): {chunk_text}")
                emo_res = await asyncio.to_thread(emotion_engine.predict, chunk_text)
                dominant_id = emo_res.get("dominant_id", 6)
                intensity = float(emo_res.get("probs_by_id", {}).get(dominant_id, 0.0))
                
                chunk_audio_bytes = await _synthesize(
                    text=chunk_text, voice="vi-VN-HoaiMyNeural", 
                    rate="+30%", volume="+0%", pitch="+0Hz"
                )
                audio_b64 = base64.b64encode(chunk_audio_bytes).decode('utf-8')
                
                # Sửa lỗi logic cũ: Đã bổ sung Rhubarb cho chunk cuối!
                mouth_cues_data = await asyncio.to_thread(generate_mouth_cues, chunk_audio_bytes)
                
                await sio.emit('server_audio_chunk', {
                    "message_id": message_id, "chunk_index": chunk_idx, "is_final": False,
                    "text_content": chunk_text, "audio_base64": audio_b64,
                    "emotion_data": {"emotionId": dominant_id, "emotionIntensity": intensity},
                    "mouthCues": mouth_cues_data
                }, to=sid)

            print("      [Proactive] Đã stream xong câu hỏi thăm!")
            await sio.emit('server_audio_chunk', {"message_id": message_id, "is_final": True}, to=sid)

        except Exception as e:
            print(f"❌ [LỖI HỎI THĂM]: {e}")
            await sio.emit('server_audio_chunk', {"message_id": message_id, "is_final": True}, to=sid)
        finally:
            # --- 🛡️ BƯỚC 2: MỞ KHÓA KHI AI NÓI XONG ---
            if sid in user_memory:
                user_memory[sid]["is_busy"] = False


# ---------------------------------------------------------
# HỨNG EVENT 2: ÂM THANH (GIAO TIẾP CHÍNH)
# ---------------------------------------------------------
@sio.on('client_audio_input')
async def handle_audio_input(sid, data):
    required_keys = ["user_id", "session_id", "audio_base64", "is_end_of_speech"]
    if not all(key in data for key in required_keys):   
        print("❌ [Lỗi] Unity gửi thiếu dữ liệu âm thanh!")
        return
    
    if sid not in user_memory:
        user_memory[sid] = {"is_busy": False, "last_audio_time": time.time()}
    else:
        user_memory[sid]["last_audio_time"] = time.time()

    is_end = data["is_end_of_speech"]
    is_speaking_now = not is_end 
    
    vision_payload = vision_engine.receive_audio_trigger(is_speaking_now)
    # vision_payload = None
    
    if sid not in user_memory:
        user_memory[sid] = {"last_vision": None, "is_busy": False}
    if vision_payload:
        user_memory[sid]["last_vision"] = vision_payload

    if is_end:
        if user_memory.get(sid, {}).get("is_busy"):
            print("      ⚠️ SEN đang bận nói, bỏ qua âm thanh này.")
            return
            
        user_memory[sid]["is_busy"] = True 

        print(f"\n🎤 [Nhận Audio] Chốt câu từ Client ID: {sid}")
        print("   -> 🤖 Bắt đầu chạy dây chuyền AI...")
        # await sio.emit('server_text_reply', {"message": "Server đang suy nghĩ..."}, to=sid)
        
        message_id = f"msg_{int(time.time())}"
        chunk_idx = 0

        try:
            audio_bytes = base64.b64decode(data["audio_base64"])
            filename = f"temp_{sid}_{int(time.time() * 1000)}.wav" 
            
            with open(filename, "wb") as f:
                f.write(audio_bytes)

            print("      [1/3] Đang dịch file Audio thành Text (Groq)...")
            def process_unity_audio(filepath):
                audio_data, sr = sf.read(filepath, dtype='float32')
                if len(audio_data.shape) > 1:
                    audio_data = audio_data[:, 0]

                cleaned_audio = stt_module.clean_audio(audio_data)
                tone = stt_module.tone_model.analyze(cleaned_audio, stt_module.config["sample_rate"])
                text = stt_module.groq_transcribe(cleaned_audio)
                
                if text and stt_module._is_hallucination(text):
                    print(f"        ⚠️ [Bộ lọc] Bắt được câu ảo giác: '{text}' -> Đã hủy!")
                    return "", tone
                
                if text:
                    stt_module._prev_transcripts.append(text)
                return text, tone

            try:
                user_text, user_tone = await asyncio.to_thread(process_unity_audio, filename)
            finally:
                if os.path.exists(filename):
                    os.remove(filename)

            # ✨ CẬP NHẬT TRÁNH TREO UNITY UI
            if not user_text or not user_text.strip():
                print("      ❌ Audio trống hoặc không nghe rõ, hủy phản hồi.")
                await sio.emit('server_text_reply', {"message": "SEN chưa nghe rõ, bạn nói lại nhé!"}, to=sid)
                await sio.emit('server_audio_chunk', {"message_id": message_id, "is_final": True}, to=sid)
                return

            vision_context = user_memory[sid].get("last_vision", "Không có dữ liệu khuôn mặt rõ ràng.")
            tone_context = user_tone.get('human_readable', 'Bình thường')
            
            user_memory[sid]["last_vision"] = None 

            context_str = f"Tone giọng của user: {tone_context} | Cảm xúc khuôn mặt: {vision_context} | Văn bản: {user_text}"
            print(f"      🧠 Đã đính kèm báo cáo ẩn vào Prompt cho SEN: [{context_str}]")

            print("      [2/3] Bắt đầu suy nghĩ, phân tích cảm xúc và stream giọng nói...")
            
            text_buffer = ""
            full_sen_text = ""

            await sio.emit('server_user_text', {"text": user_text}, to=sid)
            
            for token in generator.reply_stream(user_text, hidden_context=context_str):
                text_buffer += token
                full_sen_text += token
                await asyncio.sleep(0)
                
                if re.search(r'([.,!?:;\n]+)', text_buffer):
                    chunk_text = text_buffer.strip()
                    text_buffer = "" 
                    
                    if chunk_text:
                        print(f"        🗣️ Đang xử lý chunk: {chunk_text}")
                        
                        emo_res = await asyncio.to_thread(emotion_engine.predict, chunk_text)
                        dominant_id = emo_res.get("dominant_id", 6)
                        intensity = float(emo_res.get("probs_by_id", {}).get(dominant_id, 0.0))
                        
                        chunk_audio_bytes = await _synthesize(
                            text=chunk_text,
                            voice="vi-VN-HoaiMyNeural", 
                            rate="+30%", volume="+0%", pitch="+0Hz"
                        )
                        audio_b64 = base64.b64encode(chunk_audio_bytes).decode('utf-8')
                        
                        response_data = {
                            "message_id": message_id,
                            "chunk_index": chunk_idx,
                            "is_final": False,
                            "text_content": chunk_text,
                            "audio_base64": audio_b64,
                            "emotion_data": {
                                "emotionId": dominant_id,
                                "emotionIntensity": intensity
                            },
                            "mouthCues": [] 
                        }
                        
                        await sio.emit('server_audio_chunk', response_data, to=sid)
                        chunk_idx += 1

            if text_buffer.strip():
                chunk_text = text_buffer.strip()
                full_sen_text += chunk_text
                print(f"        🗣️ Đang xử lý chunk (cuối): {chunk_text}")
                
                emo_res = await asyncio.to_thread(emotion_engine.predict, chunk_text)
                dominant_id = emo_res.get("dominant_id", 6)
                intensity = float(emo_res.get("probs_by_id", {}).get(dominant_id, 0.0))
                
                chunk_audio_bytes = await _synthesize(
                    text=chunk_text,
                    voice="vi-VN-HoaiMyNeural", 
                    rate="+30%", volume="+0%", pitch="+0Hz"
                )
                audio_b64 = base64.b64encode(chunk_audio_bytes).decode('utf-8')
                
                response_data = {
                    "message_id": message_id,
                    "chunk_index": chunk_idx,
                    "is_final": False,
                    "text_content": chunk_text,
                    "audio_base64": audio_b64,
                    "emotion_data": {
                        "emotionId": dominant_id,
                        "emotionIntensity": intensity
                    },
                    "mouthCues": []
                }
                await sio.emit('server_audio_chunk', response_data, to=sid)
            
            if full_sen_text.strip():
                await sio.emit('server_text_reply', {"message": full_sen_text.strip()}, to=sid)

            print("      [3/3] Đã stream xong toàn bộ câu trả lời!")
            await sio.emit('server_audio_chunk', {"message_id": message_id, "is_final": True}, to=sid)

        except Exception as e:
            print(f"❌ [LỖI DÂY CHUYỀN AI]: {e}")
            await sio.emit('server_audio_chunk', {"message_id": message_id, "is_final": True}, to=sid)
        finally:
            if sid in user_memory:
                user_memory[sid]["is_busy"] = False

if __name__ == '__main__':
    print("🚀 Server đang mở cửa tại cổng 8000...")
    uvicorn.run(combined_app, host='0.0.0.0', port=8000)