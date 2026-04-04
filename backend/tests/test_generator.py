"""
brain.py — Bộ não xử lý ngôn ngữ & cảm xúc của SEN.

    ├── [EMOTION]   EmotionPredictor  — dự đoán cảm xúc từ văn bản (PhoBERT)
    └── [RESPONSE]  ResponseGenerator — sinh câu trả lời từ LLM (Groq)
"""

import os
import re
from collections import deque

import torch
from transformers import pipeline
from dotenv import load_dotenv

from backend.training_scripts.fine_tune import EmotionModelTrainer

try:
    from groq import Groq
except ImportError:
    raise SystemExit("❌ pip install groq")

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# [EMOTION]  Dự đoán cảm xúc từ văn bản
# ══════════════════════════════════════════════════════════════════════════════

class EmotionPredictor:

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

    def __init__(self, model_path: str = "vsmec_emotion_model/best"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.exists(model_path):
            print(f"Can't find model at'{model_path}' -> Recreating new model by Fine-tuning...")
            
            trainer = EmotionModelTrainer()
            trainer.train_and_save()

        print("⏳ Loading PhoBERT emotion model...")

        self.pipe = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=0 if self.device == "cuda" else -1,
            top_k=None,
        )

        config      = self.pipe.model.config
        self.labels = [config.id2label[i] for i in range(config.num_labels)]

        print("✅ Emotion model ready!")

    @staticmethod
    # chuyển label về dạng chuẩn: xóa khoảng trắng, chuyển về chữ thường, nếu rỗng thì trả về chuỗi rỗng
    def _norm_label(label: str) -> str:
        return (label or "").strip().lower()

    def _label_to_id(self, label: str) -> int | None:
        low = self._norm_label(label)
        if not low:
            return None
        if low in self.EMOTION2ID:
            return self.EMOTION2ID[low]

        alias = {
            "enjoyment": 0, "joy": 0, "happy": 0, "happiness": 0,
            "sadness": 1,   "sad": 1,
            "fear": 2,
            "angry": 3,     "anger": 3,
            "disgust": 4,
            "surprise": 5,  "surprised": 5,
            "neutral": 6,   "other": 6,
        }
        return alias.get(low)

    def predict(self, text: str) -> dict:
        # Nếu người dùng không nói gì hoặc chỉ toàn dấu cách thì trả về kết quả trung tính
        if not text or not text.strip():
            return self._neutral_result()

        results     = self.pipe(text)[0]
        raw_probs   = {l: 0.0 for l in self.labels}
        probs_by_id = {i: 0.0 for i in range(7)}

        for res in results:
            label = res.get("label")
            score = round(float(res.get("score", 0.0)), 4)

            if label in raw_probs:
                raw_probs[label] = score

            cid = self._label_to_id(label)
            if cid is not None:
                probs_by_id[cid] = score

        dominant_id = max(probs_by_id, key=probs_by_id.get) # id có xác suất cao nhất

        return {
            **raw_probs,
            "probs_by_id": probs_by_id,
            "dominant_id": dominant_id,
            "confident":   probs_by_id[dominant_id] >= 0.5,
        }

    def _neutral_result(self) -> dict:
        raw         = {e: 0.0 for e in self.labels}
        probs_by_id = {i: 0.0 for i in range(7)}
        probs_by_id[6] = 1.0

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
            "confident":   True,
        }


# ══════════════════════════════════════════════════════════════════════════════
# [RESPONSE]  Sinh câu trả lời từ LLM
# ══════════════════════════════════════════════════════════════════════════════

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT = """
You are SEN, an all-around expert renowned for making your audience feel as close and comfortable as family.
Your mission is to empathize, understand, and share in your audience's daily lives, or use your vast knowledge to help them overcome their difficulties.

Specific Instructions for Your Response:
- Tone and Style: Cheerful, friendly, and warm. Strictly avoid being overly formal, robotic, or cold. Prioritize creating a sense of intimacy and closeness.
- Target Audience: Your primary audience is Gen Z.
- Language and Expression: You MUST respond EXCLUSIVELY in natural Vietnamese. Use expressions that resonate with Gen Z, but avoid sounding overly simplistic or childish. You are highly encouraged to use trending Vietnamese Gen Z slang and jokes to make the conversation engaging and fun.
- Formatting: Keep the response clear and well-organized. Use proper punctuation, natural pauses, and a conversational flow suitable for Voice output.

CRITICAL RULES YOU MUST FOLLOW:
1. Base your response ONLY on the user's exact current input and the provided chat history. DO NOT speculate or hallucinate intent from a short or vague word/phrase.
2. If the user's input is unclear, nonsensical, or appears to be a Speech-to-Text (STT) transcription error, politely ask for clarification. Always prioritize asking over guessing.
3. DO NOT invent information. DO NOT label, assume, or assert things the user has not explicitly stated.
4. If you are unsure about the context or meaning, you MUST say "SEN không chắc..." (SEN is not sure) and ask exactly ONE clarifying question.
5. LENGTH LIMIT: Your response MUST be concise, with a MAXIMUM of 3 sentences.
6. FORMATTING: Use PLAIN TEXT ONLY. DO NOT use quotation marks (" " or "" ) for slang or emphasis
"""


class ConversationMemory:

    def __init__(self, max_turns: int = 4):
        self.history = deque(maxlen=max_turns)

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def build_messages(self, new_user_input: str) -> list:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": new_user_input})
        return messages

# Kiểm tra xem văn bản có hợp lệ để gửi cho LLM hay không, tránh gửi những câu rác 
class STTGuard:
    
    MIN_LEN = 2

    @staticmethod
    def is_valid(text: str) -> bool:
        if not text:
            return False
        text = text.strip()
        if len(text) < STTGuard.MIN_LEN:
            return False
        if text.count(" ") == 0 and len(text) > 12: # từ đơn nào dài quá 12 ký tự mà không có dấu cách -> lỗi STT
            return False
        return True


class GroqClient:

    def __init__(
        self,
        api_key:           str   = GROQ_API_KEY,
        model_name:        str   = "moonshotai/kimi-k2-instruct-0905",
        temperature:       float = 0.8,              # Tăng lên để tạo sự sáng tạo (0.7 - 0.9)
        top_p:             float = 0.9,              # Giữ mức cao để đa dạng vốn từ
        frequency_penalty: float = 0.2,              # Tránh lặp lại từ ngữ cũ
        presence_penalty:  float = 0.6,              # Khuyến khích đưa ra ý tưởng mới
        max_tokens:        int   = 256,
    ):
        if not api_key:
            raise ValueError("❌ Không tìm thấy GROQ_API_KEY trong file .env")

        self.client            = Groq(api_key=api_key)
        self.model             = model_name
        self.temperature       = temperature
        self.top_p             = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty  = presence_penalty
        self.max_tokens        = max_tokens

    def stream(self, messages: list):
        stream = self.client.chat.completions.create(
            model             = self.model,
            messages          = messages,
            temperature       = self.temperature,
            top_p             = self.top_p,
            frequency_penalty = self.frequency_penalty,
            presence_penalty  = self.presence_penalty,
            max_tokens        = self.max_tokens,
            stream            = True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


class ResponseGenerator:
    # Những câu ngắn gọn, phổ biến, không phải là lỗi STT
    _NON_AMBIGUOUS_SHORT = {
        "ok", "oke", "okay", "ừ", "uh", "dạ", "da", "vâng", "vang",
        "có", "co", "không", "khong",
        "cảm ơn", "cam on", "thanks", "thank you",
    }

    def __init__(self):
        self.memory = ConversationMemory()
        self.guard  = STTGuard()
        self.llm    = GroqClient()

    @staticmethod
    # chuẩn hóa văn bản: xóa khoảng trắng thừa, chuyển về một dòng, nếu rỗng thì trả về chuỗi rỗng
    def _norm_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())

    # check xem SEN có cần hỏi lại để làm rõ ý người dùng hay không
    def _needs_clarification(self, user_text: str) -> bool:
        t = self._norm_text(user_text)
        if not t:
            return True
        if t.lower() in self._NON_AMBIGUOUS_SHORT:
            return False
        if len(t.split()) == 1:
            return True
        return False    


    def reply_stream(self, user_text: str, hidden_context=""):
        user_text = self._norm_text(user_text)

        if not self.guard.is_valid(user_text):
            yield "Xin lỗi, SEN nghe chưa rõ. Bạn có thể nói lại giúp SEN không?"
            return

        if self._needs_clarification(user_text):
            yield (
                f"Bạn vừa nói \"{user_text}\". SEN chưa chắc bạn muốn nói về điều gì—"
                "bạn có thể nói rõ hơn 1 câu hoặc cho SEN biết bạn muốn hỏi gì không?"
            )
            return

        # --- BẮT ĐẦU PHẦN TÍCH HỢP HIDDEN CONTEXT ---
        # Trộn báo cáo ẩn vào câu nói của user (Chỉ dùng cho lượt này, không lưu vào lịch sử)
        enriched_user_text = user_text
        if hidden_context and hidden_context.strip():
            enriched_user_text = (
                f"{user_text}\n\n"
                f"--- \n"
                f"[System Note - Chỉ đọc, KHÔNG đọc to lên]: {hidden_context}. "
                f"Hãy điều chỉnh thái độ phản hồi cho phù hợp với cảm xúc này một cách tự nhiên."
            )
        
        messages = self.memory.build_messages(enriched_user_text)

        response_text = ""

        for token in self.llm.stream(messages):
            response_text += token
            yield token

        self.memory.add_user(user_text)
        self.memory.add_assistant(response_text)