"""
EnvironmentClassifier — CLAP zero-shot audio classification.
Model: laion/clap-htsat-fused (HuggingFace Transformers + PyTorch)

pip install transformers accelerate torchaudio rich
"""

import threading
import time

import numpy as np
import torch

try:
    from transformers import pipeline
except ImportError:
    raise SystemExit("❌ pip install transformers accelerate")

try:
    import torchaudio.functional as F_audio
except ImportError:
    raise SystemExit("❌ pip install torchaudio")

try:
    from rich.console import Console
    console = Console()
except ImportError:
    raise SystemExit("❌ pip install rich")


class EnvironmentClassifier:
    """
    Nhận diện môi trường bằng CLAP zero-shot audio classification.
    Chạy nền mỗi eval_interval_s giây. Chỉ callback khi môi trường thay đổi.
    """

    PRESETS = {
        "quiet": {
            "vad_threshold":   0.40,
            "min_silence_ms":  350,
            "webrtc_vad_mode": 0,
            "highpass_hz":     60,
            "lowpass_hz":      7_800,
        },
        "normal": {
            "vad_threshold":   0.55,
            "min_silence_ms":  450,
            "webrtc_vad_mode": 2,
            "highpass_hz":     80,
            "lowpass_hz":      7_500,
        },
        "noisy": {
            "vad_threshold":   0.65,
            "min_silence_ms":  650,
            "webrtc_vad_mode": 3,
            "highpass_hz":     120,
            "lowpass_hz":      6_800,
        },
    }

    CANDIDATE_LABELS = {
        "quiet":  "a very quiet room with almost no background noise or silence",
        "normal": "a quiet indoor environment with light ambient sounds like typing or air conditioning",
        "noisy":  "a noisy environment with crowd noise, traffic, music, or loud background sounds",
    }

    _STT_SR      = 16_000
    _CLAP_SR     = 48_000
    _MIN_AUDIO_S = 2.0
    _MIN_SAMPLES = int(_MIN_AUDIO_S * _STT_SR)
    _VOTE_WINDOW = 3 # Nó lưu lại lịch sử 3 lần dự đoán gần nhất -> lấy nhãn xuất hiện nhiều nhất trong 3 lần đó

    def __init__(
        self,
        on_env_change,
        eval_interval_s: float = 3.0,
        model_id: str = "laion/clap-htsat-fused",
        device: str | None = None,
    ):
        self.on_env_change   = on_env_change
        self.eval_interval_s = eval_interval_s

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._audio_buf = []
        self._buf_lock  = threading.Lock()
        self._current   = "normal"
        self._history   = []
        self._running   = False

        console.print(f"[cyan]⏳ Tải CLAP model ({model_id}, device={device})...[/cyan]")

        try:
            self._pipe = pipeline(
                task="zero-shot-audio-classification",
                model=model_id,
                device=0 if device == "cuda" else -1,
            )
        except Exception as e:
            if device == "cuda":
                console.print(f"[yellow]⚠️  CUDA load thất bại ({e}), fallback CPU...[/yellow]")
                self._pipe = pipeline(
                    task="zero-shot-audio-classification",
                    model=model_id,
                    device=-1,
                )
            else:
                raise

        self._labels     = list(self.CANDIDATE_LABELS.keys())
        self._label_text = list(self.CANDIDATE_LABELS.values())

        console.print("[green]✅ CLAP sẵn sàng[/green]")

    # ── Audio Buffer ──────────────────────────────────────────────────────────

    def push_audio(self, block: np.ndarray) -> None:
        with self._buf_lock:
            self._audio_buf.append(block.copy())

    # ── Resample ──────────────────────────────────────────────────────────────

    @staticmethod
    def _resample(audio: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(audio).unsqueeze(0)
        t = F_audio.resample(t, EnvironmentClassifier._STT_SR, EnvironmentClassifier._CLAP_SR)
        return t.squeeze(0).numpy()

    # ── Classify ──────────────────────────────────────────────────────────────

    def _classify_once(self) -> tuple[str, str]:
        with self._buf_lock:
            if not self._audio_buf:
                return self._current, "N/A"
            audio = np.concatenate(self._audio_buf)
            self._audio_buf.clear()

        if len(audio) < self._MIN_SAMPLES:
            return self._current, "N/A"

        audio_48k = self._resample(audio)

        try:
            results = self._pipe(
                audio_48k,
                candidate_labels=self._label_text,
                sampling_rate=self._CLAP_SR,
            )
        except Exception as e:
            console.print(f"[red]❌ CLAP inference lỗi: {e}[/red]")
            return self._current, "N/A"

        top_text  = results[0]["label"]
        top_score = results[0]["score"]

        winner = self._current
        for preset, text in self.CANDIDATE_LABELS.items():
            if text == top_text:
                winner = preset
                break

        return winner, f"{top_text[:40]}... ({top_score:.2f})"

    #lấy nhãn xuất hiện nhiều nhất trong 3 lần đó
    def _vote(self, new_preset: str) -> str:
        self._history.append(new_preset)
        if len(self._history) > self._VOTE_WINDOW:
            self._history.pop(0)
        return max(set(self._history), key=self._history.count)

    # ── Background Loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            time.sleep(self.eval_interval_s)

            raw_preset, reason = self._classify_once()
            if raw_preset == "N/A":
                continue

            voted_preset = self._vote(raw_preset)
            if voted_preset != self._current:
                old           = self._current
                self._current = voted_preset
                self.on_env_change(voted_preset, old, reason)

    # ── Public ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="EnvClassifier").start()
        console.print("[green]✅ EnvironmentClassifier (CLAP) đang chạy nền[/green]")

    def stop(self) -> None:
        self._running = False

    @property
    def current_preset(self) -> str:
        return self._current