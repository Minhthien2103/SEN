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

#DEFAULT_VOICE   = "vi-VN-NamMinhNeural"   # Giọng nam   tiếng Việt
DEFAULT_VOICE = "vi-VN-HoaiMyNeural"   # Giọng nữ   tiếng Việt
DEFAULT_RATE    = "+30%"    # tốc độ: -50% .. +100%
DEFAULT_VOLUME  = "+0%"    # âm lượng: -50% .. +50%
DEFAULT_PITCH   = "+0Hz"   # cao độ:   -50Hz .. +50Hz


# ===================== HELPER =====================

async def _synthesize(text: str, voice: str, rate: str, volume: str, pitch: str) -> bytes:
    """Gọi Edge TTS và trả về raw audio bytes (MP3)."""
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
    """Decode MP3 bytes rồi phát qua sounddevice."""
    buf = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(buf, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()


# ===================== MAIN CLASS =====================

class TextToSpeech:
    """
    Text-to-Speech module dùng Microsoft Edge TTS.

    Sử dụng:
        tts = TextToSpeech()
        tts.speak("Xin chào!")          # non-blocking, xếp hàng
        tts.speak_wait("Xin chào!")     # blocking, đợi phát xong
        tts.stop()                       # dừng hẳn
    """

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

        self._worker_synth = threading.Thread(target=self._synth_loop, daemon=True, name="TTS-synth")
        self._worker_play  = threading.Thread(target=self._play_loop, daemon=True, name="TTS-play")
        self._worker_synth.start()
        self._worker_play.start()

        console.print(f"[green]✅ TTS sẵn sàng — giọng: [bold]{voice}[/bold][/green]")

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Loại bỏ dấu ngoặc kép để tránh TTS bị khựng.
        Giữ nguyên phần còn lại.
        """
        if not text:
            return ""
        # Bỏ các dạng ngoặc kép phổ biến: " ” “ ‘ ’
        return re.sub(r'["“”‘’]', "", text)

    # ── public API ────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Xếp text vào hàng chờ phát (non-blocking)."""
        cleaned = self._clean_text(text)
        if cleaned and cleaned.strip():
            self._q.put(cleaned.strip())

    def speak_wait(self, text: str) -> None:
        """Phát ngay và chờ xong (blocking)."""
        cleaned = self._clean_text(text)
        if not cleaned or not cleaned.strip():
            return
        audio = asyncio.run(
            _synthesize(cleaned.strip(), self.voice, self.rate, self.volume, self.pitch)
        )
        _play_audio(audio)

    def is_speaking(self) -> bool:
        """True nếu đang phát âm thanh."""
        return self._is_speaking.is_set()

    def stop(self) -> None:
        """Dừng worker thread."""
        self._running = False
        self._q.put(None)          # unblock get()
        self._audio_q.put(None)    # unblock play get()
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


