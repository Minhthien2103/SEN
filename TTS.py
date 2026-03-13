import os
import queue
import threading
import re
import base64

import sounddevice as sd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

try:
    import requests
except ImportError:
    raise SystemExit("❌ pip install requests")

try:
    from rich.console import Console
    console = Console()
except ImportError:
    raise SystemExit("❌ pip install rich")

# ===================== CONFIG =====================

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_VOICE_NAME_ENV = "GEMINI_VOICE_NAME"

# Dùng model hỗ trợ Audio output (gemini-2.0-flash hoặc gemini-1.5-flash)
DEFAULT_MODEL_ID = "gemini-2.5-flash-preview-tts"
# Gemini thường trả về PCM ở mức 24000Hz
DEFAULT_SAMPLE_RATE = 24000
# Các giọng đọc: Puck, Charon, Kore, Fenrir, Aoede
DEFAULT_VOICE_NAME = "Puck"


# ===================== AUDIO =====================

def _play_audio_pcm16le(audio_bytes: bytes, samplerate: int = DEFAULT_SAMPLE_RATE) -> None:
    """Phát raw PCM signed 16-bit little-endian."""
    audio_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
    if audio_i16.size == 0:
        return

    audio_f32 = (audio_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    sd.play(audio_f32, samplerate)
    sd.wait()


# ===================== SYNTH =====================

def _synthesize_gemini(
    text: str,
    voice_name: str,
    model_id: str,
    api_key: str,
) -> bytes:
    """
    Gọi Google AI Studio (Gemini API) để tạo giọng nói.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "contents": [
            {
                "role": "user",
                # Rất quan trọng: Phải yêu cầu model CHỈ đọc lại text, không trò chuyện thêm
                "parts": [{"text": f"Read the following text exactly as written, without adding any conversational filler or extra words:\n\n{text}"}]
            }
        ],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice_name
                    }
                }
            }
        }
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if not r.ok:
        raise RuntimeError(f"Gemini API error {r.status_code}: {r.text}")

    res_json = r.json()
    
    try:
        # Lấy dữ liệu âm thanh base64 từ JSON response
        parts = res_json["candidates"][0]["content"]["parts"]
        for part in parts:
            if "inlineData" in part:
                b64_data = part["inlineData"]["data"]
                # mime_type = part["inlineData"].get("mimeType", "")
                return base64.b64decode(b64_data)
        
        raise ValueError("Không tìm thấy dữ liệu audio trong phản hồi của API.")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response format: {res_json}")


# ===================== MAIN CLASS =====================

class TextToSpeech:
    """
    TTS dùng Google AI Studio (Gemini API).

    ENV cần:
        GEMINI_API_KEY
        GEMINI_VOICE_NAME (Optional, mặc định: Puck)
    """

    def __init__(
        self,
        voice_name: str | None = None,
        model_id: str = DEFAULT_MODEL_ID,
    ):

        api_key = os.getenv(GEMINI_API_KEY_ENV)
        if not api_key:
            raise ValueError("❌ Không tìm thấy GEMINI_API_KEY")

        v_name = voice_name or os.getenv(GEMINI_VOICE_NAME_ENV) or DEFAULT_VOICE_NAME

        self.api_key = api_key
        self.voice_name = v_name
        self.model_id = model_id
        self.sample_rate = DEFAULT_SAMPLE_RATE

        self._q: queue.Queue[str | None] = queue.Queue()
        self._audio_q: queue.Queue[bytes | None] = queue.Queue()

        self._is_speaking = threading.Event()
        self._running = True

        self._worker_synth = threading.Thread(
            target=self._synth_loop,
            daemon=True,
            name="TTS-synth"
        )

        self._worker_play = threading.Thread(
            target=self._play_loop,
            daemon=True,
            name="TTS-play"
        )

        self._worker_synth.start()
        self._worker_play.start()

        console.print(
            f"[green]✅ TTS (Gemini API) ready — voice: [bold]{self.voice_name}[/bold][/green]"
        )

    # ─────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        return re.sub(r'["“”‘’]', "", text)

    # ─────────────────────────────
    # PUBLIC
    # ─────────────────────────────

    def speak(self, text: str) -> None:
        cleaned = self._clean_text(text)

        if cleaned and cleaned.strip():
            self._q.put(cleaned.strip())

    def speak_wait(self, text: str) -> None:
        cleaned = self._clean_text(text)

        if not cleaned.strip():
            return

        audio = _synthesize_gemini(
            text=cleaned.strip(),
            voice_name=self.voice_name,
            model_id=self.model_id,
            api_key=self.api_key,
        )

        _play_audio_pcm16le(audio, samplerate=self.sample_rate)

    def is_speaking(self) -> bool:
        return self._is_speaking.is_set()

    def stop(self) -> None:

        self._running = False

        self._q.put(None)
        self._audio_q.put(None)

        self._worker_synth.join(timeout=3)
        self._worker_play.join(timeout=3)

        console.print("[red]⏹️ TTS stopped[/red]")

    # ─────────────────────────────
    # INTERNAL
    # ─────────────────────────────

    def _synth_loop(self):

        while self._running:

            text = self._q.get()

            if text is None:
                self._audio_q.put(None)
                break

            try:

                audio = _synthesize_gemini(
                    text=text,
                    voice_name=self.voice_name,
                    model_id=self.model_id,
                    api_key=self.api_key,
                )

                self._audio_q.put(audio)

            except Exception as e:
                console.print(f"[red]TTS synth error: {e}[/red]")

    def _play_loop(self):

        while self._running:

            audio = self._audio_q.get()

            if audio is None:
                break

            try:

                self._is_speaking.set()

                _play_audio_pcm16le(audio, samplerate=self.sample_rate)

            except Exception as e:
                console.print(f"[red]TTS play error: {e}[/red]")

            finally:

                self._is_speaking.clear()