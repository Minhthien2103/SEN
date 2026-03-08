import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import queue
import threading
import wave
import tempfile
import numpy as np
import math
from collections import deque
import torch
from dotenv import load_dotenv
import torchaudio.functional as F
import noisereduce as nr
import re
from typing import Callable
from STT_emotion_pred import EmotionPredictor
from groq import Groq

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


class SpeechToText:

    def __init__(self, config: dict | None = None, on_transcript: Callable[[str], None] | None = None):

        self.on_transcript = on_transcript
        self.console = Console()

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ Không tìm thấy GROQ_API_KEY trong file .env")
        self.groq_client = Groq(api_key=api_key)

        self.config = config or {
            "sample_rate": 16000,
            "language": "vi",

            "vad_threshold": 0.45,
            "min_speech_chunks": 2,
            "min_silence_ms": 450,
            "speech_pad_ms": 200,
            "min_segment_ms": 400,

            "webrtc_vad_mode": 0,

            "early_transcribe_s": 8.0,

            "save_to_file": True,
            "output_file": "transcript.txt",

            "save_audio": True,
            "audio_output_dir": "recorded_audio",

            # Audio clean (sẽ được override theo noise profile)
            "highpass_hz": 80,
            "lowpass_hz": 7500,
            "nr_stationary": True,
            "nr_prop_decrease": 0.6,
            "nr_n_fft": 512,
        }

        # ================= NOISE PROFILES =================
        # 3 mức: yên tĩnh / bình thường / ồn
        self._env_profiles = {
            "quiet": {
                "webrtc_vad_mode": 0,
                "vad_threshold": 0.40,
                "min_silence_ms": 350,
                "early_transcribe_s": 7.0,
                "highpass_hz": 60,
                "lowpass_hz": 7800,
                "nr_stationary": True,
                "nr_prop_decrease": 0.35,
            },
            "normal": {
                "webrtc_vad_mode": 1,
                "vad_threshold": 0.45,
                "min_silence_ms": 450,
                "early_transcribe_s": 8.0,
                "highpass_hz": 80,
                "lowpass_hz": 7500,
                "nr_stationary": True,
                "nr_prop_decrease": 0.60,
            },
            "noisy": {
                "webrtc_vad_mode": 2,
                "vad_threshold": 0.55,
                "min_silence_ms": 650,
                "early_transcribe_s": 9.0,
                "highpass_hz": 120,
                "lowpass_hz": 6800,
                "nr_stationary": True,
                "nr_prop_decrease": 0.80,
            },
        }

        self._cfg_lock = threading.Lock()
        self._env_level = "normal"

        self.block_size = 512
        self._webrtc_frame_len = 320

        self.audio_q = queue.Queue()
        self.segment_q = queue.Queue()

        self.is_running = False

        self.vad_model = None
        self.webrtc_vad = None

        self._apply_env_profile(self._env_level, announce=False)

        # Noise estimator state
        self._noise_db_hist = deque(maxlen=200)
        self._speech_db_hist = deque(maxlen=120)
        self._last_env_eval_t = 0.0
        self._env_eval_interval_s = 2.0
        self._min_noise_blocks = 25
        self._min_speech_blocks = 8

        self.collected_audio = []
        self._seg_counter = 0
        self._session_tag = time.strftime("%Y%m%d_%H%M%S")

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

    # ================= NOISE EVAL =================

    @staticmethod
    def _rms_dbfs(samples_f32: np.ndarray) -> float:
        """
        Ước lượng RMS theo dBFS cho audio float32 [-1, 1].
        """
        x = np.asarray(samples_f32, dtype=np.float32)
        if x.size == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        return 20.0 * math.log10(rms + 1e-12)

    @staticmethod
    def _env_label_vi(level: str) -> str:
        return {"quiet": "yên tĩnh", "normal": "bình thường", "noisy": "ồn"}.get(level, level)

    def _apply_env_profile(self, level: str, announce: bool = True) -> None:
        profile = self._env_profiles.get(level)
        if not profile:
            return

        with self._cfg_lock:
            for k, v in profile.items():
                self.config[k] = v

            # cập nhật WebRTC VAD mode ngay khi đổi profile
            if self.webrtc_vad is not None:
                try:
                    self.webrtc_vad = webrtcvad.Vad(self.config["webrtc_vad_mode"])
                except Exception:
                    pass

        if announce:
            console.print(f"[magenta]🔊 Môi trường: {self._env_label_vi(level)}[/magenta]")

    def _maybe_eval_environment(self) -> None:
        now = time.time()
        if now - self._last_env_eval_t < self._env_eval_interval_s:
            return

        self._last_env_eval_t = now

        if len(self._noise_db_hist) < self._min_noise_blocks or len(self._speech_db_hist) < self._min_speech_blocks:
            return

        noise_db = float(np.median(np.array(self._noise_db_hist, dtype=np.float32)))
        speech_db = float(np.median(np.array(self._speech_db_hist, dtype=np.float32)))
        snr = speech_db - noise_db

        # Heuristic thresholds (tinh chỉnh nếu cần)
        if noise_db > -35.0 or snr < 10.0:
            level = "noisy"
        elif noise_db < -55.0 and snr > 20.0:
            level = "quiet"
        else:
            level = "normal"

        if level != self._env_level:
            self._env_level = level
            self._apply_env_profile(level, announce=True)

    # ================= MODEL LOAD =================

    def load_models(self):

        console.print("[cyan]⏳ Khởi tạo WebRTC VAD...[/cyan]")
        self.webrtc_vad = webrtcvad.Vad(self.config["webrtc_vad_mode"])
        console.print("[green]✅ WebRTC VAD OK[/green]")

        console.print("[cyan]⏳ Tải Silero VAD...[/cyan]")
        self.vad_model = self._load_silero_vad_local()
        self.vad_model.eval()
        console.print("[green]✅ Silero VAD OK[/green]")

    # ================= SILERO CACHE =================

    def _load_silero_vad_local(self):

        hub_dir = torch.hub.get_dir()
        model_path = os.path.join(hub_dir, "silero_vad_cached.jit")

        if not os.path.exists(model_path):

            console.print("[yellow]📥 Caching Silero VAD...[/yellow]")

            model, _ = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                force_reload=False,
                onnx=False
            )

            torch.jit.save(model, model_path)
            return model

        else:

            model = torch.jit.load(model_path, map_location="cpu")
            return model

    # ================= AUDIO CALLBACK =================

    def audio_callback(self, indata, frames, time_info, status):
        self.audio_q.put(indata[:, 0].copy().astype(np.float32))

    # ================= WEBRTC VAD =================

    def _webrtc_is_speech(self, samples_f32):

        pcm_int16 = (samples_f32 * 32767).clip(-32768, 32767).astype(np.int16)

        frame_len = self._webrtc_frame_len
        sr = self.config["sample_rate"]

        n_frames = len(pcm_int16) // frame_len

        for i in range(n_frames):

            frame = pcm_int16[i * frame_len:(i + 1) * frame_len].tobytes()

            try:
                if self.webrtc_vad.is_speech(frame, sr):
                    return True
            except Exception:
                return True

        return False

    # ================= VAD THREAD =================

    def vad_thread(self):

        SR = self.config["sample_rate"]

        min_sil_chunks = int(self.config["min_silence_ms"] / 1000 * SR / self.block_size)

        early_chunks = int(self.config["early_transcribe_s"] * SR / self.block_size)

        speech_buf = []
        sil_count = 0
        in_speech = False

        carry = np.array([], dtype=np.float32)

        self.vad_model.reset_states()

        while self.is_running:

            try:
                raw = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            combined = np.concatenate([carry, raw])

            n = len(combined) // self.block_size

            carry = combined[n * self.block_size:]

            for i in range(n):

                blk = combined[i * self.block_size:(i + 1) * self.block_size]

                webrtc_speech = self._webrtc_is_speech(blk)

                if not webrtc_speech and not in_speech:
                    # block yên lặng/không nói: dùng để ước lượng noise floor
                    self._noise_db_hist.append(self._rms_dbfs(blk))
                    self._maybe_eval_environment()
                    continue

                prob = self._vad_prob(blk)

                # cập nhật speech stats khi có khả năng là giọng nói
                if webrtc_speech:
                    self._speech_db_hist.append(self._rms_dbfs(blk))
                self._maybe_eval_environment()

                with self._cfg_lock:
                    vad_threshold = self.config["vad_threshold"]

                is_v = prob >= vad_threshold

                if is_v:

                    sil_count = 0

                    if not in_speech:
                        in_speech = True
                        speech_buf = [blk]
                    else:
                        speech_buf.append(blk)

                else:

                    if in_speech:

                        speech_buf.append(blk)
                        sil_count += 1

                        if sil_count >= min_sil_chunks:

                            audio = np.concatenate(speech_buf)

                            self.segment_q.put(audio)

                            speech_buf = []
                            sil_count = 0
                            in_speech = False

                            self.vad_model.reset_states()

                if in_speech and len(speech_buf) >= early_chunks:

                    audio = np.concatenate(speech_buf)

                    self.segment_q.put(audio)

                    speech_buf = []
                    in_speech = False

                    self.vad_model.reset_states()

    def _vad_prob(self, block):

        t = torch.FloatTensor(block)

        with torch.no_grad():
            return self.vad_model(t, 16000).item()

    # ================= CLEAN AUDIO =================

    def clean_audio(self, audio):

        with self._cfg_lock:
            sr = self.config["sample_rate"]
            hp = self.config.get("highpass_hz", 80)
            lp = self.config.get("lowpass_hz", 7500)
            nr_stationary = self.config.get("nr_stationary", True)
            nr_prop = self.config.get("nr_prop_decrease", 0.6)
            nr_n_fft = self.config.get("nr_n_fft", 512)

        tensor = torch.from_numpy(audio).unsqueeze(0)

        tensor = F.highpass_biquad(tensor, sr, float(hp))
        tensor = F.lowpass_biquad(tensor, sr, float(lp))

        audio = tensor.squeeze(0).numpy()

        peak = np.max(np.abs(audio))

        if peak > 0:
            audio = audio / peak * 0.9

        audio = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=bool(nr_stationary),
            prop_decrease=float(nr_prop),
            n_fft=int(nr_n_fft),
        )

        return audio.astype(np.float32)

    # ================= GROQ STT =================

    def groq_transcribe(self, audio):

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

    def _is_hallucination(self, text):

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

            min_samples = int(self.config["sample_rate"] * self.config["min_segment_ms"] / 1000)

            if len(audio) < min_samples:
                continue

            clean = self.clean_audio(audio)

            self.collected_audio.append(clean)

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

            emotion = self.emotion_model.predict(text)

            # In cảm xúc trội + xác suất của tất cả cảm xúc
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

            # In mức môi trường hiện tại để bạn theo dõi
            console.print(f"[dim]🔊 Env: {self._env_label_vi(self._env_level)}[/dim]")

            if self.on_transcript:

                try:
                    self.on_transcript(text)
                except Exception as e:
                    console.print(f"[red]Reply error: {e}[/red]")

    # ================= MAIN =================

    def run(self):

        self.load_models()

        self.is_running = True

        threading.Thread(target=self.vad_thread, daemon=True).start()
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

    # ================= SAVE WAV =================

    def _save_wav(self, path, audio):

        sr = self.config["sample_rate"]

        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)

        with wave.open(path, "wb") as wf:

            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

    def _save_full_session(self):
        """Ghép tất cả đoạn đã nhận diện thành một file WAV duy nhất."""
        if not self.config["save_audio"] or not self.collected_audio:
            return
        out_dir   = self.config["audio_output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        full_path = os.path.join(out_dir, f"{self._session_tag}_full_session.wav")
        combined  = np.concatenate(self.collected_audio)
        self._save_wav(full_path, combined)
        console.print(f"[bold green]✅ Đã lưu toàn bộ phiên: {full_path}[/bold green]")
