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
6. FORMATTING: Use PLAIN TEXT ONLY. DO NOT use quotation marks (" " or “” ) for slang or emphasis
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
        model_name: str = "moonshotai/kimi-k2-instruct-0905",
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