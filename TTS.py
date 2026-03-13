import asyncio
import io
import queue
import threading
import re

import edge_tts
import sounddevice as sd
import soundfile as sf

try:
    from rich.console import Console
    console = Console()
except ImportError:
    raise SystemExit("❌ pip install rich")


# ===================== CONFIG =====================
DEFAULT_VOICE = "vi-VN-HoaiMyNeural"   # Giọng nữ  tiếng Việt
DEFAULT_RATE    = "+30%"    # tốc độ: -50% .. +100%
DEFAULT_VOLUME  = "+0%"    # âm lượng: -50% .. +50%
DEFAULT_PITCH   = "+0Hz"   # cao độ:   -50Hz .. +50Hz


# ===================== HELPER =====================
async def _synthesize(text: str, voice: str, rate: str, volume: str, pitch: str) -> bytes:
    communicate = edge_tts.Communicate(
        text   = text,
        voice  = voice,
        rate   = rate,
        volume = volume,
        pitch  = pitch,
    )
    mp3_buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buf.write(chunk["data"])
    mp3_buf.seek(0)
    return mp3_buf.read()


def _play_audio(audio_bytes: bytes) -> None:
    buf = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(buf, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()


# ===================== MAIN CLASS =====================
class TextToSpeech:
    def __init__(
        self,
        voice:  str = DEFAULT_VOICE,
        rate:   str = DEFAULT_RATE,
        volume: str = DEFAULT_VOLUME,
        pitch:  str = DEFAULT_PITCH,
    ):
        self.voice  = voice
        self.rate   = rate
        self.volume = volume
        self.pitch  = pitch

        self._q:        queue.Queue[str | None] = queue.Queue()
        self._audio_q:  queue.Queue[bytes | None] = queue.Queue()
        self._is_speaking = threading.Event()
        self._running     = True
        
        # Buffer dành riêng cho Streaming LLM
        self.text_buffer = "" 

        self._worker_synth = threading.Thread(target=self._synth_loop, daemon=True, name="TTS-synth")
        self._worker_play  = threading.Thread(target=self._play_loop, daemon=True, name="TTS-play")
        self._worker_synth.start()
        self._worker_play.start()

        console.print(f"[green]✅ TTS sẵn sàng — giọng: [bold]{voice}[/bold][/green]")

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        return re.sub(r'["“”‘’]', "", text)

    # ── MỚI: API cho Streaming ────────────────────────────────

    def stream_token(self, token: str) -> None:
        """
        Nhận từng token từ LLM. Gom lại và cắt câu ngay khi gặp dấu ngắt nghỉ.
        """
        if not token:
            return
            
        self.text_buffer += token

        # Biểu thức chính quy tìm các dấu ngắt nhịp tự nhiên
        match = re.search(r'([.,;!?\n]+)', self.text_buffer)
        if match:
            split_point = match.end()
            chunk = self.text_buffer[:split_point]

            # Đẩy cụm văn bản đi tổng hợp âm thanh ngay
            self.speak(chunk)

            # Giữ lại phần text thừa chưa hoàn thiện cho vòng lặp sau
            self.text_buffer = self.text_buffer[split_point:]

    def flush_stream(self) -> None:
        """
        BẮT BUỘC GỌI sau khi LLM đã stream xong toàn bộ câu trả lời 
        để phát nốt đoạn text cuối cùng không có dấu câu.
        """
        if self.text_buffer.strip():
            self.speak(self.text_buffer)
        self.text_buffer = ""

    # ── public API ────────────────────────────────────────────

    def speak(self, text: str) -> None:
        cleaned = self._clean_text(text)
        # Sửa đổi nhỏ: Tránh đẩy những chunk chỉ có mỗi dấu câu (như ".") vào Edge-TTS gây lỗi
        if cleaned and any(c.isalnum() for c in cleaned):
            self._q.put(cleaned.strip())

    def speak_wait(self, text: str) -> None:
        cleaned = self._clean_text(text)
        if not cleaned or not any(c.isalnum() for c in cleaned):
            return
        audio = asyncio.run(
            _synthesize(cleaned.strip(), self.voice, self.rate, self.volume, self.pitch)
        )
        _play_audio(audio)

    def is_speaking(self) -> bool:
        return self._is_speaking.is_set()

    def stop(self) -> None:
        self._running = False
        self._q.put(None)
        self._audio_q.put(None)
        self._worker_synth.join(timeout=3)
        self._worker_play.join(timeout=3)
        console.print("[red]⏹️  TTS đã dừng[/red]")

    # ── internal ──────────────────────────────────────────────

    def _synth_loop(self) -> None:
        while self._running:
            text = self._q.get()
            if text is None:
                self._audio_q.put(None)
                break
            try:
                audio = asyncio.run(
                    _synthesize(text, self.voice, self.rate, self.volume, self.pitch)
                )
                self._audio_q.put(audio)
            except Exception as e:
                console.print(f"[red]TTS synth error: {e}[/red]")

    def _play_loop(self) -> None:
        while self._running:
            audio = self._audio_q.get()
            if audio is None:
                break
            try:
                self._is_speaking.set()
                _play_audio(audio)
            except Exception as e:
                console.print(f"[red]TTS play error: {e}[/red]")
            finally:
                self._is_speaking.clear()