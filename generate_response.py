"""
Response Generator cho Voice Assistant
- Groq API
- Streaming token
- Conversation memory
- Anti-hallucination guard
- Low latency
"""

import os
import re
from collections import deque

try:
    from groq import Groq
except ImportError:
    raise SystemExit("❌ pip install groq")

try:
    from dotenv import load_dotenv
except ImportError:
    raise SystemExit("❌ pip install python-dotenv")


# ================= LOAD ENV =================

load_dotenv()   # đọc file .env


# ================= CONFIG =================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT = """
Bạn là SEN, một trợ lý voice assistant.
Bạn trò chuyện với người dùng về cuộc sống hằng ngày, không giới hạn chủ đề.
Trả lời thân thiện, ấm áp, gần gũi như đang nói chuyện với bạn, tránh quá trang trọng hay lạnh lùng
Trả lời bằng tiếng Việt, khi trả lời thì xưng hô là SEN

QUY TẮC:
- Chỉ dựa trên nội dung người dùng vừa nói và lịch sử hội thoại; không suy đoán ý định từ một từ/cụm từ ngắn.
- Nếu câu hỏi không rõ hoặc có vẻ bị sai do speech-to-text, hãy hỏi lại lịch sự (ưu tiên hỏi làm rõ thay vì đoán).
- Không tự bịa thông tin, không gán nhãn/khẳng định điều người dùng chưa nói.
- Nếu không chắc chắn, hãy nói "SEN không chắc" và hỏi 1 câu để làm rõ.
- Trả lời tối đa 3 câu, ngắn gọn.
"""


# ================= MEMORY =================

class ConversationMemory:
    """
    Lưu lịch sử hội thoại (sliding window)
    """

    def __init__(self, max_turns: int = 6):
        self.history = deque(maxlen=max_turns)

    def add_user(self, text: str):
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.history.append({"role": "assistant", "content": text})

    def build_messages(self, new_user_input: str):

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for msg in self.history:
            messages.append(msg)

        messages.append({"role": "user", "content": new_user_input})

        return messages


# ================= GUARD =================

class STTGuard:

    MIN_LEN = 2

    @staticmethod
    def is_valid(text: str):

        if not text:
            return False

        text = text.strip()

        if len(text) < STTGuard.MIN_LEN:
            return False

        if text.count(" ") == 0 and len(text) > 12:
            return False

        return True


# ================= GROQ CLIENT =================

class GroqClient:

    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        model_name: str = "llama-3.1-8b-instant",
        # thấp hơn để giảm "bịa"/suy đoán
        temperature: float = 0.15,
        top_p: float = 0.85,
        # nhẹ để hạn chế lặp/lan man
        frequency_penalty: float = 0.2,
        presence_penalty: float = 0.0,
        max_tokens: int = 256,
    ):

        if not api_key:
            raise ValueError("❌ Không tìm thấy GROQ_API_KEY trong file .env")

        self.client = Groq(api_key=api_key)

        self.model = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens

    # ================= NORMAL =================

    def generate(self, messages):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content.strip()

    # ================= STREAM =================

    def stream(self, messages):

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            max_tokens=self.max_tokens,
            stream=True,
        )

        for chunk in stream:

            delta = chunk.choices[0].delta

            if delta and delta.content:
                yield delta.content


# ================= RESPONSE GENERATOR =================

class ResponseGenerator:

    def __init__(self):

        self.memory = ConversationMemory()
        self.guard = STTGuard()
        self.llm = GroqClient()
        self._non_ambiguous_short = {
            "ok", "oke", "okay", "ừ", "uh", "dạ", "da", "vâng", "vang", "có", "co", "không", "khong",
            "cảm ơn", "cam on", "thanks", "thank you",
        }

    @staticmethod
    def _norm_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())

    def _needs_clarification(self, user_text: str) -> bool:
        """
        Chặn case input quá ngắn/1 từ (dễ khiến LLM tự suy đoán).
        """
        t = self._norm_text(user_text)
        if not t:
            return True

        low = t.lower()
        if low in self._non_ambiguous_short:
            return False

        # 1 token (vd: "Cristo") rất mơ hồ -> hỏi lại thay vì đoán
        if len(t.split()) == 1:
            return True

        return False

    # ================= STREAM =================

    def reply_stream(self, user_text: str):

        user_text = self._norm_text(user_text)

        if not self.guard.is_valid(user_text):
            yield "Xin lỗi, SEN nghe chưa rõ. Bạn có thể nói lại giúp SEN không?"
            return

        if self._needs_clarification(user_text):
            yield f"Bạn vừa nói “{user_text}”. SEN chưa chắc bạn muốn nói về điều gì—bạn có thể nói rõ hơn 1 câu hoặc cho SEN biết bạn muốn hỏi gì không?"
            return

        messages = self.memory.build_messages(user_text)

        response_text = ""

        for token in self.llm.stream(messages):

            response_text += token
            yield token

        self.memory.add_user(user_text)
        self.memory.add_assistant(response_text)

    # ================= NORMAL =================

    def reply(self, user_text: str):

        user_text = self._norm_text(user_text)

        if not self.guard.is_valid(user_text):
            return "Xin lỗi, SEN nghe chưa rõ. Bạn có thể nói lại giúp SEN không?"

        if self._needs_clarification(user_text):
            return f"Bạn vừa nói “{user_text}”. SEN chưa chắc bạn muốn nói về điều gì—bạn có thể nói rõ hơn 1 câu hoặc cho SEN biết bạn muốn hỏi gì không?"

        messages = self.memory.build_messages(user_text)

        response = self.llm.generate(messages)

        self.memory.add_user(user_text)
        self.memory.add_assistant(response)

        return response