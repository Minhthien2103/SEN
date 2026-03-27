import cv2
import time
import json
import torch
import base64
import numpy as np
import mediapipe as mp
from collections import deque
from hsemotion.facial_emotions import HSEmotionRecognizer
from Vision.core.MoodManager import MoodManager

class SenseVisionBackend:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.6)
        
        print("Đang tải AI Cảm xúc...")
        original_torch_load = torch.load # Lưu lại hàm load gốc
        
        def patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False # Ép tắt bảo mật chặn mã độc
            return original_torch_load(*args, **kwargs)
        
        torch.load = patched_torch_load # Tráo hàm của PyTorch
        
        self.fer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device='cpu')
        
        torch.load = original_torch_load # Khôi phục lại hàm gốc để an toàn cho các module khác

        # self.fer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device='cpu')
        
        # TÁCH LÀM 2 BỂ CHỨA ĐỘC LẬP THEO FLOWCHART
        self.silence_buffer = MoodManager(mode_name="1_Min_Silence")
        self.speaking_buffer = MoodManager(mode_name="5_Min_Speaking")
        
        self.FPS_LIMIT = 3
        self.FRAME_INTERVAL = 1.0 / self.FPS_LIMIT 
        self.last_process_time = 0
        
        self.is_speaking = False
        
        # Giới hạn thời gian (Đổi thành 10s và 30s để test demo cho nhanh)
        self.SILENCE_LIMIT = 60  # Thực tế là 60 (1 phút)
        self.SPEAKING_LIMIT = 300 # Thực tế là 300 (5 phút)
        
        self.last_silence_report = time.time()
        self.speak_start_time = None

    def calculate_reliability(self, gray_face):
        brightness = np.mean(gray_face)
        w_light = max(0.0, 1.0 - ((brightness - 128.0) / 128.0) ** 2)
        variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        w_blur = min(1.0, variance / 100.0) 
        return brightness, variance, w_light * w_blur

    def estimate_head_pose(self, detection):
        # ... (Giữ nguyên) ...
        keypoints = detection.location_data.relative_keypoints
        eye_center_y = (keypoints[0].y + keypoints[1].y) / 2.0
        pitch = (keypoints[2].y - eye_center_y) / ((keypoints[3].y - keypoints[2].y) + 1e-6)
        if pitch < 0.75: return "Looking Down"
        elif pitch > 1.5: return "Looking Up"
        else: return "Frontal"

    def receive_audio_trigger(self, is_speaking_now: bool):
        if is_speaking_now and not self.is_speaking:
            print("\n[VAD] Bắt đầu NÓI. Chuyển nhánh -> Lưu mảng 5 phút!")
            self.is_speaking = True
            self.speak_start_time = time.time()
            self.speaking_buffer.clear_log() 
            
        elif not is_speaking_now and self.is_speaking:
            print("\n[VAD] Ngừng nói. Kích hoạt kết hợp Module Âm thanh!")
            self.is_speaking = False
            
            # Xuất JSON của nhánh Speaking
            json_out = self.speaking_buffer.generate_json_payload(trigger_reason="User Audio Finished")
            print(f"\n--- [JSON GIAO TIẾP] ---\n{json_out}\n")
            
            # Reset lại đồng hồ nhánh Silence để không bị nổ cò súng lỗi
            self.last_silence_report = time.time()
            self.silence_buffer.clear_log()

    def process_base64_frame(self, b64_string: str):
        """
        Hàm hứng ảnh Base64 từ Unity, giải mã, cắt mặt và nhận diện cảm xúc.
        """
        # ==========================================
        # 1. GIẢI MÃ BASE64 THÀNH ẢNH OPENCV
        # ==========================================
        try:
            # Xóa đoạn râu ria header nếu Unity có đính kèm (vd: "data:image/jpeg;base64,...")
            if "," in b64_string:
                b64_string = b64_string.split(",")[1]
            
            img_data = base64.b64decode(b64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return None
        except Exception as e:
            print(f"❌ [Vision] Lỗi giải mã Base64: {e}")
            return None

        # ==========================================
        # 2. XỬ LÝ KHUNG HÌNH (Ráp logic cũ của sếp vào đây)
        # ==========================================
        current_time = time.time()
        
        # Giới hạn FPS để không làm cháy CPU máy chủ
        if current_time - self.last_process_time < self.FRAME_INTERVAL:
            return None
            
        self.last_process_time = current_time

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        # Nút: Kiểm tra mặt -> Bắt khuôn to nhất
        if results.detections:
            largest_detection = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
            bboxC = largest_detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            
            x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            w, h = int(bboxC.width * iw), int(bboxC.height * ih)
            
            # An toàn: Tránh tọa độ âm làm crash numpy array
            x, y = max(0, x), max(0, y)
            face_img = frame[y:min(y+h, ih), x:min(x+w, iw)]

            if face_img.size > 0:
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                
                # Nút: Tính trọng số tin cậy
                _, _, w_total = self.calculate_reliability(gray_face)
                
                # Nút: Tin cậy > 0.3 mới đẩy vào model AI
                if w_total > 0.3:
                    head_pose = self.estimate_head_pose(largest_detection)
                    raw_emotion, scores = self.fer.predict_emotions(face_img, logits=False)
                    confidence = max(scores)
                    
                    # RẼ NHÁNH VÀO 2 BỂ CHỨA
                    if self.is_speaking:
                        self.speaking_buffer.add_emotion(raw_emotion, confidence, w_total, head_pose)
                    else:
                        self.silence_buffer.add_emotion(raw_emotion, confidence, w_total, head_pose)
                    
                    # In log nhỏ cho sếp dễ theo dõi trên Terminal
                    print(f"👁️ [Vision] Nhìn thấy mặt: {raw_emotion} (Tin cậy: {confidence:.2f} | Pose: {head_pose})")
                    
                    # Trả về payload tức thời
                    return {"emotion": raw_emotion, "confidence": float(confidence)}
        
        return None

    def run_simulation(self):
        cap = cv2.VideoCapture(0)
        print("\n[HỆ THỐNG] Đã nạp logic Flowchart mới nhất.")
        print(f"- Nhánh Im lặng: Báo cáo mỗi {self.SILENCE_LIMIT}s.")
        print(f"- Nhánh Giao tiếp: Tối đa {self.SPEAKING_LIMIT}s. (Nhấn giữ 's' để giả lập nói)\n")

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
                # Ép xuất JSON nếu nói vượt quá giới hạn 5 phút
                if current_time - self.speak_start_time >= self.SPEAKING_LIMIT:
                    print("\n[VAD] Đã chạm mốc 5 phút nói liên tục. Ép đẩy JSON!")
                    json_out = self.speaking_buffer.generate_json_payload(trigger_reason="Max Speaking Time Reached (5m)")
                    print(f"\n--- [JSON GIAO TIẾP ÉP BUỘC] ---\n{json_out}\n")
                    self.speak_start_time = time.time() # Reset đếm lại
                    self.speaking_buffer.clear_log()
            else:
                # Xuất JSON nếu im lặng đủ 1 phút
                if current_time - self.last_silence_report >= self.SILENCE_LIMIT:
                    json_out = self.silence_buffer.generate_json_payload(trigger_reason=f"Silent for {self.SILENCE_LIMIT}s")
                    print(f"\n--- [JSON HỌC TĨNH 1 PHÚT] ---\n{json_out}\n")
                    self.last_silence_report = time.time()
                    self.silence_buffer.clear_log()

            # ==========================================
            # XỬ LÝ KHUNG HÌNH (Chạy chung cho cả 2 nhánh)
            # ==========================================
            if current_time - self.last_process_time >= self.FRAME_INTERVAL:
                self.last_process_time = current_time
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb_frame)

                # Nút: Kiểm tra mặt (MediaPipe) -> Không -> Bỏ frame
                if results.detections:
                    largest_detection = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
                    bboxC = largest_detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                    w, h = int(bboxC.width * iw), int(bboxC.height * ih)
                    face_img = frame[max(0, y):min(y+h, ih), max(0, x):min(x+w, iw)]

                    if face_img.size > 0:
                        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                        
                        # Nút: Tính độ sáng, độ nét -> Trọng số tin cậy
                        _, _, w_total = self.calculate_reliability(gray_face)
                        
                        # Nút: Tin cậy > 0.3 mới xử lý tiếp
                        if w_total > 0.3:
                            head_pose = self.estimate_head_pose(largest_detection)
                            raw_emotion, scores = self.fer.predict_emotions(face_img, logits=False)
                            
                            # Nút: Có đang nói không -> Rẽ nhánh lưu mảng
                            if self.is_speaking:
                                self.speaking_buffer.add_emotion(raw_emotion, max(scores), w_total, head_pose)
                            else:
                                self.silence_buffer.add_emotion(raw_emotion, max(scores), w_total, head_pose)
                            
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"{raw_emotion} (W:{w_total:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Cập nhật UI
            status = f"SPEAKING (Limit {self.SPEAKING_LIMIT}s)" if self.is_speaking else f"SILENT (Limit {self.SILENCE_LIMIT}s)"
            cv2.putText(display_frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.is_speaking else (0, 255, 255), 2)
            cv2.imshow('Flowchart Implementation', display_frame)

        cap.release()
        cv2.destroyAllWindows()