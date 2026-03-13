from transformers import pipeline
import torch


class EmotionPredictor:

    # Canonical emotion ids and names (required output order)
    ID2EMOTION = {
        0: "Enjoyment",
        1: "Sadness",
        2: "Fear",
        3: "Anger",
        4: "Disgust",
        5: "Surprise",
        6: "Neutral",
    }
    EMOTION2ID = {v.lower(): k for k, v in ID2EMOTION.items()}

    def __init__(self, model_path="vsmec_emotion_model/best"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"⏳ Loading PhoBERT emotion model...")

        self.pipe = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=0 if self.device == "cuda" else -1,
            top_k=None
        )

        # Lấy nhãn trực tiếp từ model đã fine-tune để khớp với output thật sự
        config = self.pipe.model.config
        self.labels = [config.id2label[i] for i in range(config.num_labels)]

        print("✅ Emotion model ready!")

    @staticmethod
    def _norm_label(label: str) -> str:
        return (label or "").strip().lower()

    def _label_to_id(self, label: str) -> int | None:
        """
        Map model label -> canonical emotion id (0..6).
        Falls back to Neutral for common 'other'/'neutral' variants.
        """
        low = self._norm_label(label)
        if not low:
            return None

        if low in self.EMOTION2ID:
            return self.EMOTION2ID[low]

        # Common variants from fine-tuned datasets
        alias = {
            "enjoyment": 0,
            "joy": 0,
            "happy": 0,
            "happiness": 0,
            "sadness": 1,
            "sad": 1,
            "fear": 2,
            "angry": 3,
            "anger": 3,
            "disgust": 4,
            "surprise": 5,
            "surprised": 5,
            "neutral": 6,
            "other": 6,
        }
        return alias.get(low)

    def predict(self, text: str):

        if not text or not text.strip():
            return self._neutral_result()

        results = self.pipe(text)[0]

        # Keep raw per-model-label probs (for debugging/compat)
        raw_probs = {l: 0.0 for l in self.labels}

        # Canonical probs by id 0..6 (required output)
        probs_by_id = {i: 0.0 for i in range(7)}

        for res in results:
            label = res.get("label")
            score = round(float(res.get("score", 0.0)), 4)

            if label in raw_probs:
                raw_probs[label] = score

            cid = self._label_to_id(label)
            if cid is not None:
                probs_by_id[cid] = score

        dominant_id = max(probs_by_id, key=probs_by_id.get)

        return {
            **raw_probs,
            "probs_by_id": probs_by_id,
            "dominant_id": dominant_id,
            "confident": probs_by_id[dominant_id] >= 0.5,
        }

    def _neutral_result(self):

        raw = {e: 0.0 for e in self.labels}
        probs_by_id = {i: 0.0 for i in range(7)}
        probs_by_id[6] = 1.0  # Neutral

        # Keep backward-compatible raw label defaults
        if "neutral" in raw:
            raw["neutral"] = 1.0
        elif "other" in raw:
            raw["other"] = 1.0
        elif self.labels:
            raw[self.labels[0]] = 1.0

        return {
            **raw,
            "probs_by_id": probs_by_id,
            "dominant_id": 6,
            "confident": True,
        }