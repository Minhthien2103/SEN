"""
FILE: emotion_config.py
MÔ TẢ: 
    File cấu hình cốt lõi quản lý toàn bộ các hằng số và bộ quy đổi (Mapper) 
    liên quan đến hệ thống nhận diện cảm xúc. Đảm bảo tính nhất quán dữ liệu 
    trước khi đưa vào LLM hoặc trả về cho Client/Unity.
"""

class EmotionConfig:
    # 1. Các nhãn cảm xúc chuẩn hóa (Standard Labels) dùng chung cho toàn hệ thống
    STANDARD_LABELS = [
        "angry", "calm", "disgust", "fearful", "happy", 
        "neutral", "sad", "surprised", "uncertain"
    ]

    # 2. Bộ quy đổi (Mapper) từ RAW output của các Model (Audio & Vision) về chuẩn chung
    RAW_TO_STANDARD = {
        # -- HuBERT (Audio) raw labels --
        "neu": "neutral",
        "hap": "happy",
        "exc": "happy",
        "sad": "sad",
        "ang": "angry",
        "fea": "fearful",
        
        # -- Vision raw labels (bao gồm cả các biến thể) --
        "happiness": "happy",
        "anger": "angry",
        "fear": "fearful",
        "surprise": "surprised",
    }

    # 3. Định dạng Output Tiếng Việt (Dùng cho UI / Log / LLM Prompt)
    VI_MAP = {
        "angry":     "Tức giận",
        "calm":      "Bình tĩnh",
        "disgust":   "Khó chịu",
        "fearful":   "Sợ hãi",
        "happy":     "Vui vẻ",
        "neutral":   "Bình thường",
        "sad":       "Buồn bã",
        "surprised": "Ngạc nhiên",
        "uncertain": "Không rõ",
    }

    # 4. Định dạng Output cho module Vision (Chuẩn Client/API)
    VISION_FORMAT = {
        "happy":     "0: Enjoyment / happiness",
        "sad":       "1: Sadness",
        "fearful":   "2: Fear",
        "angry":     "3: Anger",
        "disgust":   "4: Disgust",
        "surprised": "5: Surprise",
        "neutral":   "6: Neutral",
        # Fallback các cảm xúc khác về Neutral cho Vision nếu cần
        "calm":      "6: Neutral",
        "uncertain": "6: Neutral"
    }