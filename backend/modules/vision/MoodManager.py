"""
FILE: MoodManager.py
MÔ TẢ:
    Module quản lý và tổng hợp trạng thái cảm xúc theo chuỗi thời gian (Time-series) cho Vision.
    Sử dụng thuật toán Discount Factor để ưu tiên cảm xúc gần nhất và cơ chế 
    Sustained Peaks để lọc nhiễu, giúp LLM nhận định đúng tâm lý người dùng
    mà không bị đánh lừa bởi các biểu cảm thoáng qua (ví dụ: chớp mắt, nhăn mặt vô thức).
"""

import json

# Import bộ cấu hình chuẩn đã thống nhất từ file config
from core.emotion_config import EmotionConfig

class MoodManager:
    """
    Bộ quản lý bộ đệm cảm xúc (Emotion Buffer).
    Tích lũy kết quả từ nhiều frame camera để tính toán ra một phân phối cảm xúc ổn định nhất.
    """

    def __init__(self, mode_name="Silence"):
        self.mode_name = mode_name
        
        # Lưu trữ dưới dạng List để duy trì thứ tự thời gian thực, 
        # bắt buộc phải có để tính toán hàm suy giảm (Discount Factor) từ cũ đến mới.
        self.emotions_log = [] 
        
        self.PEAK_THRESHOLD = 0.70
        
        # Lọc nhiễu (Debounce): Đòi hỏi cảm xúc mạnh phải duy trì liên tục tối thiểu 3 frames (~1 giây)
        # để loại trừ các lỗi nhận diện sai đột xuất từ mô hình AI.
        self.SUSTAINED_FRAMES = 3 
        
        # Danh sách các trạng thái tâm lý tiêu cực cần theo dõi sát sao để AI chủ động an ủi
        self.TARGET_PEAKS = [
            EmotionConfig.VISION_FORMAT["sad"],
            EmotionConfig.VISION_FORMAT["fear"],
            EmotionConfig.VISION_FORMAT["angry"],
            EmotionConfig.VISION_FORMAT["disgust"]
        ]


    def _normalize_emotion(self, raw_emotion: str) -> str:
        """
        Quy đổi nhãn cảm xúc thô từ mô hình AI về định dạng "ID: Tên" chuẩn của Client Unity.
        """
        # Bước 1: Ánh xạ từ nhãn raw (happiness, anger...) về chuẩn chung (happy, angry...)
        standard_key = EmotionConfig.RAW_TO_STANDARD.get(raw_emotion.lower())
        
        # Đảm bảo hệ thống không bị crash nếu model trả về một nhãn lạ ngoài từ điển
        if not standard_key:
            standard_key = "neutral"
            
        # Bước 2: Đẩy ra định dạng bắt buộc cho UI (VD: "0: Enjoyment / happiness")
        return EmotionConfig.VISION_FORMAT.get(standard_key, EmotionConfig.VISION_FORMAT["neutral"])


    def add_emotion(self, raw_emotion: str, score: float, w_total: float, head_pose: str):
        """
        Nạp dữ liệu của một khung hình mới vào bộ đệm lịch sử.
        
        Args:
            raw_emotion (str): Tên cảm xúc gốc trả về từ mô hình HSEmotion.
            score (float): Điểm xác suất của cảm xúc đó.
            w_total (float): Trọng số tin cậy dựa trên chất lượng khuôn mặt (độ nghiêng, độ sáng).
            head_pose (str): Hướng quay của đầu (Trái, Phải, Thẳng).
        """
        mapped_emotion = self._normalize_emotion(raw_emotion)
        self.emotions_log.append({
            "emotion": mapped_emotion,
            "score": float(score),
            "head_pose": head_pose,
            "reliability": float(w_total)
        })


    def clear_log(self):
        self.emotions_log.clear()


    def _calculate_discounted_distribution(self) -> tuple[dict, str]:
        """
        Tính toán phân phối cảm xúc dựa trên lịch sử tích lũy.
        Áp dụng thuật toán Discount Factor để ưu tiên các khung hình mới nhất.
        
        Returns:
            tuple: (Từ điển phần trăm phân phối, Cảm xúc chiếm tỷ trọng cao nhất)
        """
        # Hệ số suy giảm (Gamma): Khung hình càng cũ trong quá khứ thì giá trị tác động 
        # lên quyết định hiện tại càng giảm đi 5%. Phản ánh đúng bản chất tâm lý thay đổi theo thời gian.
        gamma = 0.95 
        N = len(self.emotions_log)
        
        weighted_counts = {}
        total_weight = 0.0

        for i, item in enumerate(self.emotions_log):
            weight = (gamma ** (N - 1 - i)) * item["reliability"]
            emo = item["emotion"]
            
            weighted_counts[emo] = weighted_counts.get(emo, 0.0) + weight
            total_weight += weight

        if total_weight == 0: 
            return {}, EmotionConfig.VISION_FORMAT["neutral"]

        distribution = {k: round(v / total_weight, 2) for k, v in weighted_counts.items()}
        dominant_emotion = max(weighted_counts, key=weighted_counts.get)
        
        return distribution, dominant_emotion


    def _find_sustained_peaks(self) -> list:
        """
        Truy quét và lọc ra các "Đỉnh cảm xúc" tiêu cực kéo dài.
        """
        peaks = set()
        consecutive_count = 0
        current_tracking_emo = None

        for item in self.emotions_log:
            emo = item["emotion"]
            score = item["score"]
            
            if score >= self.PEAK_THRESHOLD and emo in self.TARGET_PEAKS:
                if emo == current_tracking_emo:
                    consecutive_count += 1
                else:
                    current_tracking_emo = emo
                    consecutive_count = 1
                    
                # Chỉ ghi nhận Peak khi cảm xúc đó duy trì liên tục vượt qua ngưỡng chịu đựng (SUSTAINED_FRAMES)
                if consecutive_count >= self.SUSTAINED_FRAMES:
                    peaks.add(emo)
            else:
                consecutive_count = 0
                current_tracking_emo = None
                
        return list(peaks)


    def generate_json_payload(self, trigger_reason: str) -> str:
        """
        Đóng gói toàn bộ kết quả phân tích thành chuỗi JSON chuẩn để đẩy sang LLM hoặc Client.
        """
        if not self.emotions_log:
            return json.dumps({
                "status": "idle", 
                "trigger": trigger_reason, 
                "message": "Bỏ qua do không đủ độ tin cậy hoặc mất dấu khuôn mặt."
            }, ensure_ascii=False)
            
        total_frames = len(self.emotions_log)
        
        distribution, dominant_emotion = self._calculate_discounted_distribution()
        detected_peaks = self._find_sustained_peaks()
        
        poses = [item["head_pose"] for item in self.emotions_log]
        dominant_pose = max(set(poses), key=poses.count)
        avg_reliability = sum(item["reliability"] for item in self.emotions_log) / total_frames

        payload = {
            "status": "active",
            "trigger_reason": trigger_reason,
            "mode": self.mode_name,
            "analyzed_frames": total_frames,
            "dominant_mood": dominant_emotion,
            "mood_distribution": distribution,
            "critical_sustained_peaks": detected_peaks, 
            "dominant_head_pose": dominant_pose,
            "avg_reliability": round(avg_reliability, 2)
        }
        
        return json.dumps(payload, indent=2, ensure_ascii=False)