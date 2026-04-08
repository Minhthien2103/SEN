"""
FILE: SER.py
MÔ TẢ:
    Module Phân tích Cảm xúc qua Giọng nói (Speech Emotion Recognition).
    Sử dụng mô hình HuBERT Large để trích xuất đặc trưng và dự đoán tone giọng.
    Kết quả dự đoán sẽ được đối chiếu và chuẩn hóa bằng EmotionConfig 
    để đảm bảo đồng bộ với các module khác trong hệ thống.
"""

import numpy as np
import torch
import torchaudio.functional as F_audio
from transformers import AutoFeatureExtractor, HubertForSequenceClassification
from rich.console import Console

# Import bộ cấu hình chuẩn đã thống nhất từ file config
from core.emotion_config import EmotionConfig

console = Console()


class AudioToneAnalyzer:
    """
    Bộ phân tích cảm xúc âm thanh, hoạt động theo ngưỡng tin cậy (confidence threshold)
    để lọc bỏ các kết quả dự đoán thiếu chắc chắn.
    """

    CONFIDENCE_THRESHOLD = 0.40

    def __init__(self, debug: bool = False, confidence_threshold: float = None):
        """
        Khởi tạo mô hình HuBERT và tải trọng số vào bộ nhớ.

        Args:
            debug (bool): Bật chế độ in log chi tiết cho mục đích gỡ lỗi.
            confidence_threshold (float): Ngưỡng xác suất tối thiểu để chấp nhận một cảm xúc (thay vì uncertain).
        """
        self.debug = debug
        if confidence_threshold is not None:
            self.CONFIDENCE_THRESHOLD = confidence_threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"[cyan]Đang tải HuBERT Large lên {self.device.type.upper()}...[/cyan]")

        self.model_name      = "superb/hubert-large-superb-er"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)

        # Tối ưu hóa VRAM trên GPU bằng half-precision (float16) giúp model chạy nhẹ và nhanh hơn
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model  = HubertForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label

        console.print("[green]AudioToneAnalyzer (HuBERT Large SER) đã sẵn sàng![/green]")

    def analyze(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Phân tích mảng dữ liệu âm thanh thô để trích xuất cảm xúc chi phối.

        Args:
            audio (np.ndarray): Mảng numpy chứa tín hiệu âm thanh đầu vào.
            sample_rate (int): Tần số lấy mẫu hiện tại của đoạn âm thanh.

        Returns:
            dict: Từ điển chứa xác suất của từng cảm xúc chuẩn hóa, kèm theo
                  nhãn cảm xúc chi phối (dominant), tên tiếng Việt và độ tự tin (confidence).
        """
        # Khởi tạo vector phân phối xác suất mặc định dựa trên bộ khung chuẩn
        probs_dict = {label: 0.0 for label in EmotionConfig.STANDARD_LABELS}

        try:
            t = torch.from_numpy(audio).float()
            
            # Gộp kênh (mixdown) về mono do mô hình HuBERT chỉ hỗ trợ đầu vào 1 kênh
            if t.ndim > 1:
                t = torch.mean(t, dim=0)

            # Đồng bộ tần số lấy mẫu về mức chuẩn 16kHz để mô hình trích xuất feature chính xác
            if sample_rate != 16000:
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                t = F_audio.resample(t, sample_rate, 16000)
                audio_16k = t.squeeze(0).numpy()
            else:
                audio_16k = t.numpy()

            # Bỏ qua các đoạn âm thanh quá ngắn (dưới 0.5s) để tránh model đưa ra dự đoán nhiễu, đoán mò
            if len(audio_16k) < 8000:
                probs_dict["neutral"] = 1.0
                return {**probs_dict, "dominant": "neutral", "human_readable": EmotionConfig.VI_MAP["neutral"], "confidence": 0.0}

            inputs = self.feature_extractor(
                audio_16k,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(self.device).to(self.model.dtype)

            with torch.no_grad():
                outputs       = self.model(input_values)
                probabilities = torch.softmax(outputs.logits, dim=1).squeeze(0).cpu().float().numpy()

            if self.debug:
                raw = {self.id2label[i]: round(float(probabilities[i]), 4) for i in range(len(probabilities))}
                console.print(f"[yellow]HuBERT raw: {raw}[/yellow]")

            # Ánh xạ từ nhãn thô của mô hình sang chuẩn chung của toàn hệ thống
            for i, score in enumerate(probabilities):
                label_raw = self.id2label[i].lower()
                std_label = EmotionConfig.RAW_TO_STANDARD.get(label_raw)
                if std_label and std_label in probs_dict:
                    probs_dict[std_label] += float(score)

            # Chuẩn hóa lại tổng các xác suất về 1.0 (do có trường hợp nhiều nhãn thô trỏ về cùng một nhãn chuẩn)
            total = sum(probs_dict.values())
            if total > 0:
                probs_dict = {k: round(v / total, 4) for k, v in probs_dict.items()}

            dominant  = max(probs_dict, key=probs_dict.get)
            top_score = probs_dict[dominant]

            # Lọc bỏ các dự đoán mập mờ, thiếu cơ sở để LLM không bị tiêm prompt sai lệch cảm xúc
            if top_score < self.CONFIDENCE_THRESHOLD:
                if self.debug:
                    console.print(f"[yellow]Confidence thấp ({top_score:.3f} < {self.CONFIDENCE_THRESHOLD}) -> uncertain[/yellow]")
                dominant_out = "uncertain"
            else:
                dominant_out = dominant

            return {
                **probs_dict,
                "dominant":       dominant_out,
                "human_readable": EmotionConfig.VI_MAP.get(dominant_out, dominant_out),
                "confidence":     top_score,
            }

        except Exception as e:
            console.print(f"[red]Lỗi khi phân tích giọng nói: {e}[/red]")
            probs_dict["neutral"] = 1.0
            return {**probs_dict, "dominant": "neutral", "human_readable": "Bình thường (Lỗi)", "confidence": 0.0}