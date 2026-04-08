"""
FILE: Env_classifier.py
MÔ TẢ: 
    Module nhận diện môi trường âm thanh xung quanh (quiet, normal, noisy) 
    bằng mô hình CLAP (Contrastive Language-Audio Pretraining) zero-shot.
    Hệ thống chạy ngầm và cung cấp callback để các thành phần khác tự động 
    điều chỉnh ngưỡng nhạy âm thanh dựa theo môi trường thực tế.
"""

import threading
import time
import numpy as np
import torch

try:
    from transformers import pipeline
except ImportError:
    raise SystemExit(" pip install transformers accelerate")

try:
    import torchaudio.functional as F_audio
except ImportError:
    raise SystemExit(" pip install torchaudio")

try:
    from rich.console import Console
    console = Console()
except ImportError:
    raise SystemExit(" pip install rich")


class EnvironmentClassifier:
    """
    Quản lý luồng đánh giá âm trường độc lập.
    Liên tục thu thập chunk âm thanh và phân loại môi trường định kỳ để tránh ảnh hưởng hiệu suất luồng chính.
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
    _VOTE_WINDOW = 3 

    def __init__(
        self,
        on_env_change,
        eval_interval_s: float = 3.0,
        model_id: str = "laion/clap-htsat-fused",
        device: str | None = None,
    ):
        """
        Khởi tạo bộ phân loại môi trường.

        Args:
            on_env_change (Callable): Hàm callback được gọi khi môi trường có sự thay đổi.
            eval_interval_s (float): Thời gian chờ giữa các lần đánh giá (tính bằng giây).
            model_id (str): Định danh mô hình HuggingFace sử dụng cho CLAP.
            device (str | None): Thiết bị chạy inference ('cuda' hoặc 'cpu').
        """
        self.on_env_change   = on_env_change
        self.eval_interval_s = eval_interval_s

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._audio_buf = []
        self._buf_lock  = threading.Lock()
        self._current   = "normal"
        self._history   = []
        self._running   = False

        console.print(f"[cyan] Tải CLAP model ({model_id}, device={device})...[/cyan]")

        try:
            self._pipe = pipeline(
                task="zero-shot-audio-classification",
                model=model_id,
                device=0 if device == "cuda" else -1,
            )
        except Exception as e:
            if device == "cuda":
                # Đảm bảo hệ thống vẫn hoạt động (fallback) nếu VRAM bị đầy hoặc lỗi driver GPU
                console.print(f"[yellow]  CUDA load thất bại ({e}), fallback CPU...[/yellow]")
                self._pipe = pipeline(
                    task="zero-shot-audio-classification",
                    model=model_id,
                    device=-1,
                )
            else:
                raise

        self._labels     = list(self.CANDIDATE_LABELS.keys())
        self._label_text = list(self.CANDIDATE_LABELS.values())

        console.print("[green] CLAP sẵn sàng[/green]")


    def push_audio(self, block: np.ndarray) -> None:
        """
        Nạp đoạn âm thanh mới vào bộ đệm chờ xử lý.

        Args:
            block (np.ndarray): Mảng numpy chứa dữ liệu âm thanh thô.
        """
        with self._buf_lock:
            self._audio_buf.append(block.copy())


    @staticmethod
    def _resample(audio: np.ndarray) -> np.ndarray:
        # Chuyển đổi sample rate để phù hợp với chuẩn đầu vào bắt buộc của mô hình CLAP (48kHz)
        t = torch.from_numpy(audio).unsqueeze(0)
        t = F_audio.resample(t, EnvironmentClassifier._STT_SR, EnvironmentClassifier._CLAP_SR)
        return t.squeeze(0).numpy()


    def _classify_once(self) -> tuple[str, str]:
        with self._buf_lock:
            if not self._audio_buf:
                return self._current, "N/A"
            audio = np.concatenate(self._audio_buf)
            self._audio_buf.clear()

        # Bỏ qua suy luận nếu thời lượng thu âm chưa đủ dài, tránh việc model đoán mò thiếu cơ sở
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
            console.print(f"[red] CLAP inference lỗi: {e}[/red]")
            return self._current, "N/A"

        top_text  = results[0]["label"]
        top_score = results[0]["score"]

        winner = self._current
        for preset, text in self.CANDIDATE_LABELS.items():
            if text == top_text:
                winner = preset
                break

        return winner, f"{top_text[:40]}... ({top_score:.2f})"


    def _vote(self, new_preset: str) -> str:
        """
        Xác định nhãn môi trường cuối cùng bằng cơ chế bầu chọn (Majority Vote).

        Args:
            new_preset (str): Nhãn môi trường vừa dự đoán được từ lần chạy gần nhất.

        Returns:
            str: Nhãn môi trường xuất hiện nhiều nhất trong khung thời gian lịch sử.
        """
        # Áp dụng cơ chế cửa sổ trượt (sliding window) để lọc nhiễu (debounce).
        # Giúp hệ thống không bị chuyển đổi ngưỡng VAD liên tục do một tiếng động lạ ngẫu nhiên.
        self._history.append(new_preset)
        if len(self._history) > self._VOTE_WINDOW:
            self._history.pop(0)
        return max(set(self._history), key=self._history.count)


    def _loop(self) -> None:
        while self._running:
            time.sleep(self.eval_interval_s)

            raw_preset, reason = self._classify_once()
            if raw_preset == "N/A":
                continue

            voted_preset = self._vote(raw_preset)
            # Chỉ kích hoạt callback khi môi trường thực sự thay đổi để tiết kiệm tài nguyên xử lý
            if voted_preset != self._current:
                old           = self._current
                self._current = voted_preset
                self.on_env_change(voted_preset, old, reason)


    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="EnvClassifier").start()
        console.print("[green] EnvironmentClassifier (CLAP) đang chạy nền[/green]")

    def stop(self) -> None:
        self._running = False

    @property
    def current_preset(self) -> str:
        return self._current