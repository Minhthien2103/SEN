"""
FILE: main.py
MÔ TẢ:
    Trạm điều phối trung tâm (API Gateway & WebSocket Server) của hệ thống SEN.
    Sử dụng FastAPI để xử lý HTTP và Socket.IO (ASGI) để duy trì kết nối thời gian thực 
    hai chiều độ trễ thấp với Client Unity.
    Điều phối luồng dữ liệu giữa các module: Vision, Audio (STT/TTS), và Brain (LLM).
"""

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

# Import thư viện in màu console chuẩn hệ thống
try:
    from rich.console import Console
    console = Console()
except ImportError:
    raise SystemExit("Lỗi: Hãy chạy lệnh 'pip install rich'")

# Import từ các phân hệ của SEN
from backend.modules.audio.speech import SpeechToText, _synthesize, generate_mouth_cues
from backend.core.brain import ResponseGenerator, EmotionPredictor
from backend.modules.vision.SenseVisionBackend import SenseVisionBackend

load_dotenv()

# ---------------------------------------------------------
# 1. KHỞI TẠO CẤU HÌNH & ENGINE
# ---------------------------------------------------------
app = FastAPI()

# Nâng max_http_buffer_size lên 50MB để đảm bảo Server không chặn các luồng ảnh/âm thanh 
# chất lượng cao hoặc các gói Base64 dung lượng lớn từ Unity gửi lên.
sio = socketio.AsyncServer(
    async_mode='asgi', 
    cors_allowed_origins='*', 
    max_http_buffer_size=50000000
)
combined_app = socketio.ASGIApp(sio, app)

# Mở CORS hoàn toàn cho giai đoạn R&D để tránh các lỗi Blocked by CORS policy khi test Local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

console.print("[cyan]Đang khởi tạo bộ não và cảm xúc cho SEN...[/cyan]")
stt_module = SpeechToText() 
generator = ResponseGenerator()
emotion_engine = EmotionPredictor() 
stt_module.load_models()
vision_engine = SenseVisionBackend()
console.print("[bold green]Khởi tạo hoàn tất! Hệ thống đã sẵn sàng.[/bold green]")

# Bộ nhớ session lưu trữ trạng thái độc lập của từng user để Server có thể phục vụ nhiều người cùng lúc
user_memory = {}

# ---------------------------------------------------------
# 2. CÁC SỰ KIỆN KẾT NỐI
# ---------------------------------------------------------
@sio.event
async def connect(sid, environ):
    """Đăng ký session mới khi có Client kết nối thành công."""
    console.print(f"[green][Kết nối] Client {sid} đã vào hệ thống![/green]")
    user_memory[sid] = {"last_vision": None, "is_busy": False, "last_audio_time": time.time()} 

@sio.event
async def disconnect(sid):
    """Dọn dẹp rác bộ nhớ (Garbage Collection) khi Client ngắt kết nối để tránh tràn RAM."""
    console.print(f"[yellow][Ngắt kết nối] Client {sid} đã rời đi![/yellow]")
    if sid in user_memory:
        del user_memory[sid]

# ---------------------------------------------------------
# HỨNG EVENT 1: KHUÔN MẶT (XỬ LÝ IM LẶNG)
# ---------------------------------------------------------
@sio.on('client_face_input')
async def handle_face_input(sid, data):
    """
    Lắng nghe luồng hình ảnh camera. 
    Chức năng chính: Kích hoạt khả năng "Chủ động hỏi thăm" nếu phát hiện User im lặng quá lâu.
    """
    required_keys = ["user_id", "session_id", "image_base64"]
    if not all(key in data for key in required_keys):
        console.print("[red][Lỗi] Unity gửi thiếu dữ liệu khuôn mặt![/red]")
        return 
    
    # Dự phòng khởi tạo bộ nhớ nếu event connect bị lỡ nhịp
    if sid not in user_memory:
        user_memory[sid] = {"is_busy": False, "last_audio_time": time.time()}

    image_b64 = data["image_base64"]
    current_time = time.time()
    
    # Chạy phân tích thị giác trên luồng Worker riêng (to_thread) để không làm nghẽn Event Loop chính
    vision_payload = await asyncio.to_thread(vision_engine.process_base64_frame, image_b64)
    
    last_audio_time = user_memory[sid].get("last_audio_time", current_time)
    is_busy = user_memory[sid].get("is_busy", False)

    # KÍCH HOẠT HỎI THĂM CHỦ ĐỘNG
    # Điều kiện: Đã 60s kể từ câu nói cuối cùng VÀ hệ thống hiện không bận trả lời.
    if (current_time - last_audio_time >= 60) and not is_busy:
        
        # Khóa luồng (Locking) ngay lập tức để ngăn tình trạng "Race Condition"
        # (Nhiều frame ảnh cùng thỏa mãn điều kiện 60s khiến SEN gửi câu hỏi thăm liên tục).
        user_memory[sid]["is_busy"] = True
        user_memory[sid]["last_audio_time"] = current_time 

        console.print(f"\n[magenta][NHẬN THỨC] Phát hiện User im lặng 1 phút:[/magenta]\n{vision_payload}")
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
                
                # Cắt chuỗi gửi đi ngay khi gặp dấu câu để giảm độ trễ giọng nói xuống mức thấp nhất
                if re.search(r'([.,!?:;\n]+)', text_buffer):
                    chunk_text = text_buffer.strip()
                    text_buffer = "" 
                    
                    if chunk_text:
                        console.print(f"        [cyan]Đang xử lý chunk (Proactive): {chunk_text}[/cyan]")
                        emo_res = await asyncio.to_thread(emotion_engine.predict, chunk_text)
                        dominant_id = emo_res.get("dominant_id", 6)
                        intensity = float(emo_res.get("probs_by_id", {}).get(dominant_id, 0.0))
                        
                        chunk_audio_bytes = await _synthesize(
                            text=chunk_text, voice="vi-VN-HoaiMyNeural", 
                            rate="+30%", volume="+0%", pitch="+0Hz"
                        )
                        audio_b64 = base64.b64encode(chunk_audio_bytes).decode('utf-8')
                        
                        mouth_cues_data = await asyncio.to_thread(generate_mouth_cues, chunk_audio_bytes)
                        
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

            # Xả (flush) bộ đệm cho đoạn văn bản cuối cùng không có dấu câu
            if text_buffer.strip():
                chunk_text = text_buffer.strip()
                console.print(f"        [cyan]Đang xử lý chunk cuối (Proactive): {chunk_text}[/cyan]")
                emo_res = await asyncio.to_thread(emotion_engine.predict, chunk_text)
                dominant_id = emo_res.get("dominant_id", 6)
                intensity = float(emo_res.get("probs_by_id", {}).get(dominant_id, 0.0))
                
                chunk_audio_bytes = await _synthesize(
                    text=chunk_text, voice="vi-VN-HoaiMyNeural", 
                    rate="+30%", volume="+0%", pitch="+0Hz"
                )
                audio_b64 = base64.b64encode(chunk_audio_bytes).decode('utf-8')
                
                mouth_cues_data = await asyncio.to_thread(generate_mouth_cues, chunk_audio_bytes)
                
                await sio.emit('server_audio_chunk', {
                    "message_id": message_id, "chunk_index": chunk_idx, "is_final": False,
                    "text_content": chunk_text, "audio_base64": audio_b64,
                    "emotion_data": {"emotionId": dominant_id, "emotionIntensity": intensity},
                    "mouthCues": mouth_cues_data
                }, to=sid)

            console.print("      [green][Proactive] Đã stream xong câu hỏi thăm![/green]")
            await sio.emit('server_audio_chunk', {"message_id": message_id, "is_final": True}, to=sid)

        except Exception as e:
            console.print(f"[red][LỖI HỎI THĂM]: {e}[/red]")
            await sio.emit('server_audio_chunk', {"message_id": message_id, "is_final": True}, to=sid)
        finally:
            # Mở khóa hệ thống để tiếp tục nhận diện âm thanh từ người dùng
            if sid in user_memory:
                user_memory[sid]["is_busy"] = False


# ---------------------------------------------------------
# HỨNG EVENT 2: ÂM THANH (GIAO TIẾP CHÍNH)
# ---------------------------------------------------------
@sio.on('client_audio_input')
async def handle_audio_input(sid, data):
    """
    Lắng nghe luồng âm thanh do WebRTC VAD từ Client quyết định cắt gửi lên.
    Kích hoạt toàn bộ Pipeline: STT -> Phân tích Cảm xúc ẩn -> LLM -> TTS -> LipSync.
    """
    required_keys = ["user_id", "session_id", "audio_base64", "is_end_of_speech"]
    if not all(key in data for key in required_keys):   
        console.print("[red][Lỗi] Unity gửi thiếu dữ liệu âm thanh![/red]")
        return
    
    if sid not in user_memory:
        user_memory[sid] = {"is_busy": False, "last_audio_time": time.time()}
    else:
        user_memory[sid]["last_audio_time"] = time.time()

    is_end = data["is_end_of_speech"]
    is_speaking_now = not is_end 
    
    # Cập nhật trạng thái Nói/Im lặng cho module Vision để phân loại đúng bể chứa cảm xúc
    vision_payload = vision_engine.receive_audio_trigger(is_speaking_now)
    
    if sid not in user_memory:
        user_memory[sid] = {"last_vision": None, "is_busy": False}
    if vision_payload:
        user_memory[sid]["last_vision"] = vision_payload

    # Kích hoạt chuỗi xử lý chỉ khi người dùng ĐÃ NÓI XONG (is_end_of_speech == True)
    if is_end:
        # Chặn các gói tin âm thanh rác gửi lên trong lúc SEN đang bận trả lời
        if user_memory.get(sid, {}).get("is_busy"):
            console.print("      [yellow]SEN đang bận nói, bỏ qua âm thanh này.[/yellow]")
            return
            
        user_memory[sid]["is_busy"] = True 

        console.print(f"\n[bold green][Nhận Audio] Chốt câu từ Client ID: {sid}[/bold green]")
        console.print("   -> Bắt đầu chạy dây chuyền AI...")
        
        message_id = f"msg_{int(time.time())}"
        chunk_idx = 0

        try:
            audio_bytes = base64.b64decode(data["audio_base64"])
            filename = f"temp_{sid}_{int(time.time() * 1000)}.wav" 
            
            with open(filename, "wb") as f:
                f.write(audio_bytes)

            console.print("      [yellow][1/3] Đang dịch file Audio thành Text (Groq)...[/yellow]")
            
            def process_unity_audio(filepath):
                audio_data, sr = sf.read(filepath, dtype='float32')
                if len(audio_data.shape) > 1:
                    audio_data = audio_data[:, 0]

                cleaned_audio = stt_module.clean_audio(audio_data)
                tone = stt_module.tone_model.analyze(cleaned_audio, stt_module.config["sample_rate"])
                text = stt_module.groq_transcribe(cleaned_audio)
                
                if text and stt_module._is_hallucination(text):
                    console.print(f"        [yellow]Cảnh báo: Bắt được câu ảo giác: '{text}' -> Đã hủy![/yellow]")
                    return "", tone
                
                if text:
                    stt_module._prev_transcripts.append(text)
                return text, tone

            try:
                user_text, user_tone = await asyncio.to_thread(process_unity_audio, filename)
            finally:
                if os.path.exists(filename):
                    os.remove(filename)

            # Nếu STT trả về rỗng (nhiễu hoặc nói quá nhỏ), nhả khóa và thông báo cho Unity
            if not user_text or not user_text.strip():
                console.print("      [yellow]Audio trống hoặc không nghe rõ, hủy phản hồi.[/yellow]")
                await sio.emit('server_text_reply', {"message": "SEN chưa nghe rõ, bạn nói lại nhé!"}, to=sid)
                await sio.emit('server_audio_chunk', {"message_id": message_id, "is_final": True}, to=sid)
                user_memory[sid]["is_busy"] = False
                return

            # Gộp dữ liệu Đa phương thức (Multimodal) vào Prompt ẩn
            vision_context = user_memory[sid].get("last_vision", "Không có dữ liệu khuôn mặt rõ ràng.")
            tone_context = user_tone.get('human_readable', 'Bình thường')
            user_memory[sid]["last_vision"] = None 

            context_str = f"Tone giọng của user: {tone_context} | Cảm xúc khuôn mặt: {vision_context} | Văn bản: {user_text}"
            console.print(f"      [magenta]Đã đính kèm báo cáo ẩn vào Prompt cho SEN:[/magenta] [{context_str}]")

            console.print("      [yellow][2/3] Bắt đầu suy nghĩ, phân tích cảm xúc và stream giọng nói...[/yellow]")
            
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
                        console.print(f"        [cyan]Đang xử lý chunk: {chunk_text}[/cyan]")
                        
                        emo_res = await asyncio.to_thread(emotion_engine.predict, chunk_text)
                        dominant_id = emo_res.get("dominant_id", 6)
                        intensity = float(emo_res.get("probs_by_id", {}).get(dominant_id, 0.0))
                        
                        chunk_audio_bytes = await _synthesize(
                            text=chunk_text,
                            voice="vi-VN-HoaiMyNeural", 
                            rate="+30%", volume="+0%", pitch="+0Hz"
                        )
                        audio_b64 = base64.b64encode(chunk_audio_bytes).decode('utf-8')
                        
                        # --- VÁ LỖI LOGIC: Đã bổ sung Rhubarb Lip-sync cho nhánh Giao tiếp ---
                        mouth_cues_data = await asyncio.to_thread(generate_mouth_cues, chunk_audio_bytes)
                        
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

            if text_buffer.strip():
                chunk_text = text_buffer.strip()
                full_sen_text += chunk_text
                console.print(f"        [cyan]Đang xử lý chunk (cuối): {chunk_text}[/cyan]")
                
                emo_res = await asyncio.to_thread(emotion_engine.predict, chunk_text)
                dominant_id = emo_res.get("dominant_id", 6)
                intensity = float(emo_res.get("probs_by_id", {}).get(dominant_id, 0.0))
                
                chunk_audio_bytes = await _synthesize(
                    text=chunk_text,
                    voice="vi-VN-HoaiMyNeural", 
                    rate="+30%", volume="+0%", pitch="+0Hz"
                )
                audio_b64 = base64.b64encode(chunk_audio_bytes).decode('utf-8')
                
                # --- VÁ LỖI LOGIC: Đã bổ sung Rhubarb Lip-sync cho Chunk cuối ---
                mouth_cues_data = await asyncio.to_thread(generate_mouth_cues, chunk_audio_bytes)
                
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
            
            if full_sen_text.strip():
                await sio.emit('server_text_reply', {"message": full_sen_text.strip()}, to=sid)

            console.print("      [green][3/3] Đã stream xong toàn bộ câu trả lời![/green]")
            await sio.emit('server_audio_chunk', {"message_id": message_id, "is_final": True}, to=sid)

        except Exception as e:
            console.print(f"[red][LỖI DÂY CHUYỀN AI]: {e}[/red]")
            await sio.emit('server_audio_chunk', {"message_id": message_id, "is_final": True}, to=sid)
        finally:
            if sid in user_memory:
                user_memory[sid]["is_busy"] = False

if __name__ == '__main__':
    console.print("[bold yellow] Server đang mở cửa tại cổng 8000...[/bold yellow]")
    uvicorn.run(combined_app, host='0.0.0.0', port=8000)