"""
FILE: SenseVisionBackend.py
MÔ TẢ:
    Module quản lý hệ thống Thị giác Máy tính (Computer Vision) thời gian thực.
    Nhiệm vụ:
    1. Nhận luồng ảnh (từ Camera vật lý hoặc Base64 từ Unity).
    2. Phát hiện khuôn mặt (MediaPipe) và đánh giá độ tin cậy (ánh sáng, độ mờ).
    3. Nhận diện cảm xúc (HSEmotion) và ước lượng hướng nhìn (Head Pose).
    4. Phân luồng dữ liệu vào 2 bộ đệm (MoodManager) độc lập dựa trên tín hiệu VAD (Đang nói / Đang im lặng).
"""

import cv2
import time
import json
import torch
import base64
import numpy as np
import mediapipe as mp
from collections import deque
from hsemotion.facial_emotions import HSEmotionRecognizer
from backend.modules.vision.MoodManager import MoodManager

# Import Console từ rich để đồng bộ màu sắc log
try:
    from rich.console import Console
    console = Console()
except ImportError:
    raise SystemExit("Lỗi: Hãy chạy lệnh 'pip install rich'")

class SenseVisionBackend:
    """
    Trái tim của hệ thống phân tích hình ảnh đa luồng.
    Tích hợp cơ chế tự động giới hạn FPS (Throttling) để chống quá tải CPU.
    """

    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.6)
        
        console.print("[cyan]Đang tải AI Cảm xúc (HSEmotion)...[/cyan]")
        
        # --- BẢN VÁ BẢO MẬT PYTORCH ---
        # Lý do: Các phiên bản PyTorch mới mặc định chặn việc load model cũ (pickle) để chống mã độc.
        # Tuy nhiên, model HSEmotion được huấn luyện từ trước nên cần mở khóa tạm thời.
        original_torch_load = torch.load 
        
        def patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False 
            return original_torch_load(*args, **kwargs)
        
        torch.load = patched_torch_load 
        
        # Khởi tạo mô hình trên CPU để nhường toàn bộ VRAM GPU cho Whisper và LLM
        self.fer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device='cpu')
        
        # Khôi phục lại hàm gốc ngay lập tức để không tạo ra lỗ hổng bảo mật cho toàn hệ thống Python
        torch.load = original_torch_load 

        # --- HAI BỂ CHỨA CẢM XÚC ĐỘC LẬP ---
        # Lý do: Tâm lý con người khi đang nói (bộc lộ) và khi đang nghe/im lặng (tiếp thu) 
        # mang ý nghĩa hoàn toàn khác nhau. Cần tách riêng để LLM dễ dàng phân tích ngữ cảnh.
        self.silence_buffer = MoodManager(mode_name="1_Min_Silence")
        self.speaking_buffer = MoodManager(mode_name="5_Min_Speaking")
        
        # Tối ưu hiệu năng: Giới hạn xử lý 3 khung hình/giây.
        # Khuôn mặt con người thường không chuyển đổi cảm xúc quá 3 lần trong 1 giây,
        # việc xử lý 30-60 FPS là cực kỳ lãng phí tài nguyên CPU máy chủ.
        self.FPS_LIMIT = 3
        self.FRAME_INTERVAL = 1.0 / self.FPS_LIMIT 
        self.last_process_time = 0
        
        self.is_speaking = False
        
        # Giới hạn thời gian kích hoạt bộ đệm (Đổi thành 10s và 30s để test demo cho nhanh)
        self.SILENCE_LIMIT = 60  
        self.SPEAKING_LIMIT = 300 
        
        self.last_silence_report = time.time()
        self.speak_start_time = None

    def calculate_reliability(self, gray_face: np.ndarray) -> tuple[float, float, float]:
        """
        Đánh giá "Chất lượng" của khuôn mặt vừa bắt được.
        Loại bỏ các frame bị mờ (do lia máy nhanh) hoặc quá tối/sáng.
        
        Returns:
            tuple: (Độ sáng, Độ mờ, Điểm tin cậy tổng hợp từ 0.0 đến 1.0)
        """
        brightness = np.mean(gray_face)
        # Điểm sáng tối ưu là 128. Càng lệch về 0 (tối thui) hoặc 255 (cháy sáng) thì điểm càng thấp.
        w_light = max(0.0, 1.0 - ((brightness - 128.0) / 128.0) ** 2)
        
        # Dùng toán tử Laplacian để đo độ sắc nét của các đường viền trên mặt
        variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        w_blur = min(1.0, variance / 100.0) 
        
        return brightness, variance, w_light * w_blur

    def estimate_head_pose(self, detection) -> str:
        """
        Ước lượng hướng nhìn sơ bộ dựa trên sự tương quan tọa độ của 2 mắt, mũi và miệng.
        Giúp AI biết người dùng có đang tập trung nhìn vào màn hình (Frontal) hay không.
        """
        keypoints = detection.location_data.relative_keypoints
        eye_center_y = (keypoints[0].y + keypoints[1].y) / 2.0
        pitch = (keypoints[2].y - eye_center_y) / ((keypoints[3].y - keypoints[2].y) + 1e-6)
        
        if pitch < 0.75: 
            return "Looking Down"
        elif pitch > 1.5: 
            return "Looking Up"
        else: 
            return "Frontal"

    def receive_audio_trigger(self, is_speaking_now: bool):
        """
        Nhận tín hiệu báo động từ hệ thống VAD (Bên module Audio).
        Đóng vai trò "Công tắc" điều hướng luồng dữ liệu hình ảnh vào đúng bể chứa.
        """
        if is_speaking_now and not self.is_speaking:
            console.print("\n[yellow][VAD] Bắt đầu NÓI. Chuyển nhánh -> Lưu mảng 5 phút![/yellow]")
            self.is_speaking = True
            self.speak_start_time = time.time()
            self.speaking_buffer.clear_log() 
            
        elif not is_speaking_now and self.is_speaking:
            console.print("\n[yellow][VAD] Ngừng nói. Kích hoạt kết hợp Module Âm thanh![/yellow]")
            self.is_speaking = False
            
            # Đóng gói toàn bộ cảm xúc khuôn mặt trong lúc nói và sẵn sàng đẩy cho LLM
            json_out = self.speaking_buffer.generate_json_payload(trigger_reason="User Audio Finished")
            console.print(f"\n[bold cyan]--- [JSON GIAO TIẾP] ---[/bold cyan]\n{json_out}\n")
            
            # Reset lại đồng hồ đếm ngược của nhánh Im lặng để không bị kích hoạt lỗi (nổ cò)
            # do thời gian nói chuyện đã chiếm mất khoảng thời gian đó.
            self.last_silence_report = time.time()
            self.silence_buffer.clear_log()

    def process_base64_frame(self, b64_string: str) -> dict | None:
        """
        Hàm chính được Socket Server gọi liên tục mỗi khi Client Unity gửi ảnh lên.
        Tiến hành giải mã, cắt mặt và phân tích cảm xúc.
        """
        # ==========================================
        # 1. GIẢI MÃ BASE64 THÀNH ẢNH OPENCV
        # ==========================================
        try:
            if "," in b64_string:
                b64_string = b64_string.split(",")[1]
            
            img_data = base64.b64decode(b64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return None
        except Exception as e:
            console.print(f"[red][Vision] Lỗi giải mã Base64: {e}[/red]")
            return None

        # ==========================================
        # 2. XỬ LÝ KHUNG HÌNH (Lọc & Phân Tích)
        # ==========================================
        current_time = time.time()
        
        if current_time - self.last_process_time < self.FRAME_INTERVAL:
            return None
            
        self.last_process_time = current_time

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        # Lọc bước 1: Đảm bảo có mặt người trong khung hình
        if results.detections:
            # Chọn khuôn mặt to nhất (Gần camera nhất) để làm đối tượng phân tích chính
            largest_detection = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
            bboxC = largest_detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            
            x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            w, h = int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Cắt ảnh khuôn mặt (Tránh lỗi tọa độ âm văng ra ngoài khung hình)
            x, y = max(0, x), max(0, y)
            face_img = frame[y:min(y+h, ih), x:min(x+w, iw)]

            if face_img.size > 0:
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                
                # Lọc bước 2: Đảm bảo ảnh mặt đủ sáng và đủ nét
                _, _, w_total = self.calculate_reliability(gray_face)
                
                if w_total > 0.3:
                    head_pose = self.estimate_head_pose(largest_detection)
                    raw_emotion, scores = self.fer.predict_emotions(face_img, logits=False)
                    confidence = max(scores)
                    
                    # Lọc bước 3: Phân luồng cảm xúc dựa trên VAD
                    if self.is_speaking:
                        self.speaking_buffer.add_emotion(raw_emotion, confidence, w_total, head_pose)
                    else:
                        self.silence_buffer.add_emotion(raw_emotion, confidence, w_total, head_pose)
                    
                    console.print(f"[magenta]👁️ [Vision][/magenta] Nhìn thấy mặt: {raw_emotion} (Tin cậy: {confidence:.2f} | Pose: {head_pose})")
                    
                    return {"emotion": raw_emotion, "confidence": float(confidence)}
        
        return None

    def run_simulation(self):
        """
        Hàm chạy độc lập dùng Camera máy tính để Test cục bộ,
        không cần bật toàn bộ hệ thống Socket hay Client Unity.
        """
        cap = cv2.VideoCapture(0)
        console.print("\n[bold green][HỆ THỐNG] Đã nạp logic Flowchart mới nhất.[/bold green]")
        console.print(f"- Nhánh Im lặng: Báo cáo mỗi {self.SILENCE_LIMIT}s.")
        console.print(f"- Nhánh Giao tiếp: Tối đa {self.SPEAKING_LIMIT}s. (Nhấn giữ phím 's' để giả lập lúc đang nói)\n")

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            current_time = time.time()
            display_frame = frame.copy()
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'): self.receive_audio_trigger(True)
            else: self.receive_audio_trigger(False)

            # ==========================================
            # KIỂM TRA ĐIỀU KIỆN THỜI GIAN CỦA 2 NHÁNH
            # ==========================================
            if self.is_speaking:
                # Ép đóng gói dữ liệu nếu người dùng nói lải nhải không ngừng quá 5 phút
                if current_time - self.speak_start_time >= self.SPEAKING_LIMIT:
                    console.print("\n[yellow][VAD] Đã chạm mốc 5 phút nói liên tục. Ép đẩy JSON![/yellow]")
                    json_out = self.speaking_buffer.generate_json_payload(trigger_reason="Max Speaking Time Reached (5m)")
                    console.print(f"\n[bold cyan]--- [JSON GIAO TIẾP ÉP BUỘC] ---[/bold cyan]\n{json_out}\n")
                    self.speak_start_time = time.time() 
                    self.speaking_buffer.clear_log()
            else:
                # Ép hệ thống báo cáo tình trạng tâm lý của người dùng nếu họ im lặng nhìn màn hình quá 1 phút
                if current_time - self.last_silence_report >= self.SILENCE_LIMIT:
                    json_out = self.silence_buffer.generate_json_payload(trigger_reason=f"Silent for {self.SILENCE_LIMIT}s")
                    console.print(f"\n[bold green]--- [JSON HỌC TĨNH 1 PHÚT] ---[/bold green]\n{json_out}\n")
                    self.last_silence_report = time.time()
                    self.silence_buffer.clear_log()

            # ==========================================
            # XỬ LÝ KHUNG HÌNH (Chạy chung cho cả 2 nhánh)
            # ==========================================
            if current_time - self.last_process_time >= self.FRAME_INTERVAL:
                self.last_process_time = current_time
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb_frame)

                if results.detections:
                    largest_detection = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
                    bboxC = largest_detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    
                    x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                    w, h = int(bboxC.width * iw), int(bboxC.height * ih)
                    face_img = frame[max(0, y):min(y+h, ih), max(0, x):min(x+w, iw)]

                    if face_img.size > 0:
                        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                        
                        _, _, w_total = self.calculate_reliability(gray_face)
                        
                        if w_total > 0.3:
                            head_pose = self.estimate_head_pose(largest_detection)
                            raw_emotion, scores = self.fer.predict_emotions(face_img, logits=False)
                            
                            if self.is_speaking:
                                self.speaking_buffer.add_emotion(raw_emotion, max(scores), w_total, head_pose)
                            else:
                                self.silence_buffer.add_emotion(raw_emotion, max(scores), w_total, head_pose)
                            
                            # Vẽ khung hình xanh lá và in chữ nhận diện lên màn hình test
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"{raw_emotion} (W:{w_total:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Cập nhật UI cho màn hình giả lập
            status = f"SPEAKING (Limit {self.SPEAKING_LIMIT}s)" if self.is_speaking else f"SILENT (Limit {self.SILENCE_LIMIT}s)"
            cv2.putText(display_frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.is_speaking else (0, 255, 255), 2)
            cv2.imshow('Flowchart Implementation', display_frame)

        cap.release()
        cv2.destroyAllWindows()