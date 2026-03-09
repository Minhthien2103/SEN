import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import queue
import threading
import wave
import tempfile
import numpy as np
import re
from typing import Callable
from dotenv import load_dotenv
from groq import Groq
import torch
import torchaudio.functional as F

from STT_emotion_pred import EmotionPredictor

load_dotenv()

try:
    import sounddevice as sd
except ImportError:
    raise SystemExit("❌ pip install sounddevice")

try:
    import webrtcvad
except ImportError:
    raise SystemExit("❌ pip install webrtcvad-wheels")

try:
    from rich.console import Console
    console = Console()
except ImportError:
    raise SystemExit("❌ pip install rich")


# ================= HELPERS =================

def _resample_np(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:

    if orig_sr == target_sr:
        return audio

    t = torch.from_numpy(audio).unsqueeze(0)
    t = F.resample(t, orig_sr, target_sr)
    return t.squeeze(0).numpy()


# ================= DEEPFILTERNET WRAPPER =================

class DeepFilterDenoiser:
    """
    Wrapper cho DeepFilterNet — khử noise chất lượng cao.
    pip install deepfilternet
    """

    DF_SR = 48000   # DeepFilterNet yêu cầu 48kHz

    def __init__(self):

        try:
            from df.enhance import enhance, init_df
        except ImportError:
            raise SystemExit("❌ pip install deepfilternet")

        self._enhance = enhance
        self._model, self._df_state, _ = init_df()

    def process(self, audio_f32: np.ndarray, sr: int) -> np.ndarray:
        """
        Nhận audio float32 @ sr bất kỳ.
        Resample lên 48kHz → DeepFilterNet → resample về sr ban đầu.
        """
        audio_48k = _resample_np(audio_f32, sr, self.DF_SR)

        tensor    = torch.from_numpy(audio_48k).unsqueeze(0)
        enhanced  = self._enhance(self._model, self._df_state, tensor)
        audio_48k = enhanced.squeeze(0).numpy()

        return _resample_np(audio_48k, self.DF_SR, sr)


# ================= SPEECH TO TEXT =================

class SpeechToText:

    def __init__(
        self,
        config: dict | None = None,
        on_transcript: Callable[[str], None] | None = None,
    ):

        self.on_transcript = on_transcript
        self.console       = Console()

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ Không tìm thấy GROQ_API_KEY trong file .env")
        self.groq_client = Groq(api_key=api_key)

        self.config = config or {
            "sample_rate":        16000,
            "language":           "vi",

            "vad_threshold":      0.45,
            "min_silence_ms":     450,
            "early_transcribe_s": 8.0,

            "webrtc_vad_mode":    1,

            "highpass_hz":        80,
            "lowpass_hz":         7500,

            "save_to_file":       True,
            "output_file":        "transcript.txt",

            "save_audio":         True,
            "audio_output_dir":   "recorded_audio",
        }

        self.block_size        = 512
        self._webrtc_frame_len = 320

        self.audio_q    = queue.Queue()
        self.segment_q  = queue.Queue()
        self.is_running = False

        self.vad_model  = None
        self.webrtc_vad = None
        self.denoiser   = None

        self.collected_audio = []
        self._seg_counter    = 0
        self._session_tag    = time.strftime("%Y%m%d_%H%M%S")

        self.emotion_model = EmotionPredictor()

        self.hallucination_patterns = [
            r"^\s*$",
            r"^\W+$",
            r"^(Thanks for watching|Subscribe|Like and subscribe|Please subscribe"
            r"|Hãy subscribe cho kênh|Hẹn gặp lại|Cảm ơn các bạn"
            r"|Cảm ơn bạn đã xem|Đừng quên đăng ký|Nhớ nhấn like"
            r"|Xin chào các bạn|Chúc các bạn|Tạm biệt)",
            r"^\.{2,}$",
            r"^-{2,}$",
        ]

    # ================= MODEL LOAD =================

    def load_models(self):

        console.print("[cyan]⏳ Khởi tạo WebRTC VAD...[/cyan]")
        self.webrtc_vad = webrtcvad.Vad(self.config["webrtc_vad_mode"])
        console.print("[green]✅ WebRTC VAD OK[/green]")

        console.print("[cyan]⏳ Tải Silero VAD...[/cyan]")
        self.vad_model = self._load_silero_vad_local()
        self.vad_model.eval()
        console.print("[green]✅ Silero VAD OK[/green]")

        console.print("[cyan]⏳ Tải DeepFilterNet...[/cyan]")
        self.denoiser = DeepFilterDenoiser()
        console.print("[green]✅ DeepFilterNet OK[/green]")

    # ================= SILERO CACHE =================

    def _load_silero_vad_local(self):

        hub_dir    = torch.hub.get_dir()
        model_path = os.path.join(hub_dir, "silero_vad_cached.jit")

        if not os.path.exists(model_path):

            console.print("[yellow]📥 Caching Silero VAD...[/yellow]")

            model, _ = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                force_reload=False,
                onnx=False,
            )

            torch.jit.save(model, model_path)
            return model

        return torch.jit.load(model_path, map_location="cpu")

    # ================= AUDIO CALLBACK =================

    def audio_callback(self, indata, frames, time_info, status):
        self.audio_q.put(indata[:, 0].copy().astype(np.float32))

    # ================= WEBRTC VAD =================

    def _webrtc_is_speech(self, samples_f32: np.ndarray) -> bool:

        pcm_int16 = (samples_f32 * 32767).clip(-32768, 32767).astype(np.int16)
        frame_len = self._webrtc_frame_len
        sr        = self.config["sample_rate"]
        n_frames  = len(pcm_int16) // frame_len

        for i in range(n_frames):
            frame = pcm_int16[i * frame_len:(i + 1) * frame_len].tobytes()
            try:
                if self.webrtc_vad.is_speech(frame, sr):
                    return True
            except Exception:
                return True

        return False

    def _vad_prob(self, block: np.ndarray) -> float:

        t = torch.FloatTensor(block)
        with torch.no_grad():
            return self.vad_model(t, 16000).item()

    # ================= VAD THREAD =================

    def vad_thread(self):

        SR             = self.config["sample_rate"]
        min_sil_chunks = int(self.config["min_silence_ms"] / 1000 * SR / self.block_size)
        early_chunks   = int(self.config["early_transcribe_s"] * SR / self.block_size)

        speech_buf = []
        sil_count  = 0
        in_speech  = False
        carry      = np.array([], dtype=np.float32)

        self.vad_model.reset_states()

        while self.is_running:

            try:
                raw = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            combined = np.concatenate([carry, raw])
            n        = len(combined) // self.block_size
            carry    = combined[n * self.block_size:]

            for i in range(n):

                blk = combined[i * self.block_size:(i + 1) * self.block_size]

                webrtc_speech = self._webrtc_is_speech(blk)

                if not webrtc_speech and not in_speech:
                    continue

                prob = self._vad_prob(blk)
                is_v = prob >= self.config["vad_threshold"]

                if is_v:

                    sil_count = 0

                    if not in_speech:
                        in_speech  = True
                        speech_buf = [blk]
                    else:
                        speech_buf.append(blk)

                else:

                    if in_speech:

                        speech_buf.append(blk)
                        sil_count += 1

                        if sil_count >= min_sil_chunks:

                            self.segment_q.put(np.concatenate(speech_buf))

                            speech_buf = []
                            sil_count  = 0
                            in_speech  = False

                            self.vad_model.reset_states()

            if in_speech and len(speech_buf) >= early_chunks:

                self.segment_q.put(np.concatenate(speech_buf))

                speech_buf = []
                in_speech  = False

                self.vad_model.reset_states()

    # ================= CLEAN AUDIO =================

    def clean_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Pipeline:
          1. Highpass / lowpass filter
          2. DeepFilterNet denoise
          3. Peak normalise
        """
        sr = self.config["sample_rate"]
        hp = self.config.get("highpass_hz", 80)
        lp = self.config.get("lowpass_hz", 7500)

        tensor = torch.from_numpy(audio).unsqueeze(0)
        tensor = F.highpass_biquad(tensor, sr, float(hp))
        tensor = F.lowpass_biquad(tensor, sr, float(lp))
        audio  = tensor.squeeze(0).numpy()

        audio = self.denoiser.process(audio, sr)

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9

        return audio.astype(np.float32)

    # ================= GROQ STT =================

    def groq_transcribe(self, audio: np.ndarray) -> str:

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:

            self._save_wav(f.name, audio)

            with open(f.name, "rb") as audio_file:
                transcript = self.groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3-turbo",
                    language="vi",
                )

        return transcript.text

    # ================= HALLUCINATION FILTER =================

    def _is_hallucination(self, text: str) -> bool:

        for pat in self.hallucination_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return True

        return False

    # ================= TRANSCRIBE THREAD =================

    def transcribe_thread(self):

        while self.is_running:

            try:
                audio = self.segment_q.get(timeout=0.5)
            except queue.Empty:
                continue

            min_samples = int(self.config["sample_rate"] * 0.4)

            if len(audio) < min_samples:
                continue

            clean = self.clean_audio(audio)

            self.collected_audio.append(clean)

            if self.config.get("save_audio", False):
                try:
                    self._save_segment_wav(clean)
                except Exception as e:
                    console.print(f"[red]Audio save error: {e}[/red]")

            start = time.time()

            try:
                text = self.groq_transcribe(clean)
            except Exception as e:
                console.print(f"[red]Groq error: {e}[/red]")
                continue

            elapsed = time.time() - start

            if not text:
                continue

            text = text.strip()

            if not text or self._is_hallucination(text):
                continue

            text = text[0].upper() + text[1:]

            console.print(
                f"\n[bold yellow][VI][/bold yellow] "
                f"[white]{text}[/white]  "
                f"[dim]({elapsed:.2f}s | {len(audio)/self.config['sample_rate']:.1f}s audio)[/dim]"
            )

            if self.config["save_to_file"]:
                with open(self.config["output_file"], "a", encoding="utf-8") as f:
                    f.write(f"[VI] {text}\n")

            emotion   = self.emotion_model.predict(text)
            probs_str = ", ".join(
                f"{label}: {emotion.get(label, 0.0):.3f}"
                for label in getattr(self.emotion_model, "labels", [])
            )

            if probs_str:
                console.print(
                    f"[cyan]🧠 Emotion:[/cyan] {emotion['dominant']} "
                    f"[dim]({probs_str})[/dim]"
                )
            else:
                console.print(f"[cyan]🧠 Emotion:[/cyan] {emotion['dominant']}")

            if self.on_transcript:
                try:
                    self.on_transcript(text)
                except Exception as e:
                    console.print(f"[red]Reply error: {e}[/red]")

    # ================= MAIN =================

    def run(self):

        self.load_models()

        self.is_running = True

        threading.Thread(target=self.vad_thread,        daemon=True).start()
        threading.Thread(target=self.transcribe_thread, daemon=True).start()

        console.print("\n[green]🎙️ Đang lắng nghe...[/green]\n")

        try:

            with sd.InputStream(
                samplerate=self.config["sample_rate"],
                channels=1,
                dtype="float32",
                blocksize=self.block_size,
                callback=self.audio_callback,
                latency="low",
            ):

                while True:
                    time.sleep(0.1)

        except KeyboardInterrupt:

            console.print("\n[red]⏹️ Đã dừng[/red]")
            self.is_running = False

            try:
                self._save_full_session()
            except Exception as e:
                console.print(f"[red]Full session save error: {e}[/red]")

    # ================= SAVE WAV =================

    def _save_wav(self, path: str, audio: np.ndarray):

        sr  = self.config["sample_rate"]
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

    def _save_segment_wav(self, audio: np.ndarray) -> str:

        out_dir = self.config.get("audio_output_dir", "recorded_audio")
        os.makedirs(out_dir, exist_ok=True)

        self._seg_counter += 1
        seg_path = os.path.join(out_dir, f"{self._session_tag}_seg{self._seg_counter:03d}.wav")
        self._save_wav(seg_path, audio)

        return seg_path

    def _save_full_session(self):

        if not self.config["save_audio"] or not self.collected_audio:
            return

        out_dir   = self.config["audio_output_dir"]
        os.makedirs(out_dir, exist_ok=True)

        full_path = os.path.join(out_dir, f"{self._session_tag}_full_session.wav")
        combined  = np.concatenate(self.collected_audio)

        self._save_wav(full_path, combined)
        console.print(f"[bold green]✅ Đã lưu toàn bộ phiên: {full_path}[/bold green]")