from transformers import pipeline
import torch


class EmotionPredictor:

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

    def predict(self, text: str):

        if not text or not text.strip():
            return self._neutral_result()

        results = self.pipe(text)[0]

        probs = {l: 0.0 for l in self.labels}

        for res in results:
            label = res["label"]
            if label in probs:
                probs[label] = round(float(res["score"]), 4)

        dominant = max(probs, key=probs.get)

        return {
            **probs,
            "dominant": dominant,
            "confident": probs[dominant] >= 0.5
        }

    def _neutral_result(self):

        result = {e: 0.0 for e in self.labels}
        if "other" in result:
            result["other"] = 1.0
            result["dominant"] = "other"
        else:
            # Nếu model không có nhãn "other" thì chọn nhãn đầu tiên làm mặc định
            first = self.labels[0] if self.labels else "other"
            result[first] = 1.0
            result["dominant"] = first

        result["confident"] = True

        return result