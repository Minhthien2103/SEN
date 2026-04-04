import numpy as np
import torch
import torchaudio.functional as F_audio
from transformers import AutoFeatureExtractor, HubertForSequenceClassification
from rich.console import Console

console = Console()


class AudioToneAnalyzer:
    """
    Phân tích tone giọng dùng HuBERT Large (superb/hubert-large-superb-er).
    Nếu confidence thấp hơn ngưỡng → trả về "uncertain".
    """

    CONFIDENCE_THRESHOLD = 0.40

    LABEL_MAP = {
        "neu": "neutral",
        "hap": "happy",
        "exc": "happy",
        "sad": "sad",
        "ang": "angry",
        "fea": "fearful",
    }

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

    LABELS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

    def __init__(self, debug: bool = False, confidence_threshold: float = None):
        self.debug = debug
        if confidence_threshold is not None:
            self.CONFIDENCE_THRESHOLD = confidence_threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"[cyan]⏳ Đang tải HuBERT Large lên {self.device.type.upper()}...[/cyan]")

        self.model_name      = "superb/hubert-large-superb-er"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)

        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model  = HubertForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label

        console.print("[green]✅ AudioToneAnalyzer (HuBERT Large SER) đã sẵn sàng![/green]")

    def analyze(self, audio: np.ndarray, sample_rate: int) -> dict:
        probs_dict = {label: 0.0 for label in self.LABELS}

        try:
            t = torch.from_numpy(audio).float()
            if t.ndim > 1:
                t = torch.mean(t, dim=0)

            if sample_rate != 16000:
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                t = F_audio.resample(t, sample_rate, 16000)
                audio_16k = t.squeeze(0).numpy()
            else:
                audio_16k = t.numpy()

            if len(audio_16k) < 8000:
                probs_dict["neutral"] = 1.0
                return {**probs_dict, "dominant": "neutral", "human_readable": self.VI_MAP["neutral"], "confidence": 0.0}

            inputs = self.feature_extractor(
                audio_16k,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(self.device).to(self.model.dtype)

            with torch.no_grad():
                outputs      = self.model(input_values)
                probabilities = torch.softmax(outputs.logits, dim=1).squeeze(0).cpu().float().numpy()

            if self.debug:
                raw = {self.id2label[i]: round(float(probabilities[i]), 4) for i in range(len(probabilities))}
                console.print(f"[yellow]🔍 HuBERT raw: {raw}[/yellow]")

            for i, score in enumerate(probabilities):
                label_raw = self.id2label[i].lower()
                std_label = self.LABEL_MAP.get(label_raw)
                if std_label and std_label in probs_dict:
                    probs_dict[std_label] += float(score)

            total = sum(probs_dict.values())
            if total > 0:
                probs_dict = {k: round(v / total, 4) for k, v in probs_dict.items()}

            dominant  = max(probs_dict, key=probs_dict.get)
            top_score = probs_dict[dominant]

            if top_score < self.CONFIDENCE_THRESHOLD:
                if self.debug:
                    console.print(f"[yellow]🔍 Confidence thấp ({top_score:.3f} < {self.CONFIDENCE_THRESHOLD}) → uncertain[/yellow]")
                dominant_out = "uncertain"
            else:
                dominant_out = dominant

            return {
                **probs_dict,
                "dominant":       dominant_out,
                "human_readable": self.VI_MAP.get(dominant_out, dominant_out),
                "confidence":     top_score,
            }

        except Exception as e:
            console.print(f"[red]⚠️ Lỗi khi phân tích giọng nói: {e}[/red]")
            probs_dict["neutral"] = 1.0
            return {**probs_dict, "dominant": "neutral", "human_readable": "Bình thường (Lỗi)", "confidence": 0.0}