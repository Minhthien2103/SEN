import json

class MoodManager:
    def __init__(self, mode_name="Silence"):
        self.mode_name = mode_name
        self.emotions_log = [] # Chuyển sang List để dễ duyệt xuôi/ngược cho Discount Factor
        self.PEAK_THRESHOLD = 0.70
        self.SUSTAINED_FRAMES = 3 # Cần 3 frame liên tiếp (1 giây) để xác nhận là Peak
        
        # Danh sách các cảm xúc tiêu cực cần theo dõi Peak
        self.TARGET_PEAKS = ["1: Sadness", "2: Fear", "3: Anger", "4: Disgust"]

        self.EMOTION_MAP = {
            "Happy": "0: Enjoyment / happiness",
            "Sad": "1: Sadness",
            "Fear": "2: Fear",
            "Angry": "3: Anger",
            "Disgust": "4: Disgust",
            "Surprise": "5: Surprise",
            "Neutral": "6: Neutral"
        }

    def _normalize_emotion(self, raw_emotion: str) -> str:
        mapping = {
            "happy": "Happy", "happiness": "Happy",
            "sad": "Sad", "sadness": "Sad",
            "fear": "Fear",
            "angry": "Angry", "anger": "Angry",
            "disgust": "Disgust",
            "surprise": "Surprise",
            "neutral": "Neutral"
        }
        standard_key = mapping.get(raw_emotion.lower(), "Neutral")
        return self.EMOTION_MAP.get(standard_key, "6: Neutral")

    def add_emotion(self, raw_emotion: str, score: float, w_total: float, head_pose: str):
        mapped_emotion = self._normalize_emotion(raw_emotion)
        self.emotions_log.append({
            "emotion": mapped_emotion,
            "score": float(score),
            "head_pose": head_pose,
            "reliability": float(w_total)
        })

    def clear_log(self):
        self.emotions_log.clear()

    def _calculate_discounted_distribution(self):
        """Tính toán Phân phối cảm xúc có áp dụng Discount Factor (Hệ số suy giảm)"""
        gamma = 0.95 # Hệ số suy giảm: Cảm xúc càng cũ, sức nặng càng giảm đi 5%
        N = len(self.emotions_log)
        
        weighted_counts = {}
        total_weight = 0.0

        for i, item in enumerate(self.emotions_log):
            # Công thức: Trọng số = (0.95 ^ Độ cũ) * Độ tin cậy của frame
            weight = (gamma ** (N - 1 - i)) * item["reliability"]
            emo = item["emotion"]
            
            weighted_counts[emo] = weighted_counts.get(emo, 0.0) + weight
            total_weight += weight

        if total_weight == 0: return {}, "6: Neutral"

        # Tính phần trăm phân phối
        distribution = {k: round(v / total_weight, 2) for k, v in weighted_counts.items()}
        # Tìm cảm xúc chủ đạo (có trọng số cao nhất)
        dominant_emotion = max(weighted_counts, key=weighted_counts.get)
        
        return distribution, dominant_emotion

    def _find_sustained_peaks(self):
        """Lọc Đỉnh cảm xúc (Peak) tiêu cực / mạnh KÉO DÀI"""
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
                    
                # Nếu kéo dài đủ số frame quy định -> Xác nhận là Peak thật
                if consecutive_count >= self.SUSTAINED_FRAMES:
                    peaks.add(emo)
            else:
                consecutive_count = 0
                current_tracking_emo = None
                
        return list(peaks)

    def generate_json_payload(self, trigger_reason: str) -> str:
        if not self.emotions_log:
            return json.dumps({"status": "idle", "trigger": trigger_reason, "message": "Bỏ qua do không đủ độ tin cậy hoặc mất dấu khuôn mặt."})
            
        total_frames = len(self.emotions_log)
        
        # 1. Gọi hàm tính Toán học suy giảm (Discount Factor)
        distribution, dominant_emotion = self._calculate_discounted_distribution()
        
        # 2. Gọi hàm bắt Đỉnh cảm xúc kéo dài (Sustained Peaks)
        detected_peaks = self._find_sustained_peaks()
        
        # 3. Hướng nhìn và Độ tin cậy trung bình
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
            "critical_sustained_peaks": detected_peaks, # Đã đổi tên để phản ánh tính 'kéo dài'
            "dominant_head_pose": dominant_pose,
            "avg_reliability": round(avg_reliability, 2)
        }
        
        return json.dumps(payload, indent=2, ensure_ascii=False)