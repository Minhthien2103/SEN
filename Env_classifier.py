import os
os.environ["PANNS_HOME"] = r"C:\Users\PC\panns_data"  # phải set TRƯỚC khi import


import numpy as np
import threading
import time

try:
    from panns_inference import AudioTagging
except ImportError:
    raise SystemExit("❌ pip install panns-inference")

try:
    from rich.console import Console
    console = Console()
except ImportError:
    raise SystemExit("❌ pip install rich")


# ================= ENVIRONMENT CLASSIFIER =================

class EnvironmentClassifier:
    """
    Nhận diện môi trường xung quanh bằng PANNS (AudioSet 527 classes, PyTorch).
    Chạy nền mỗi eval_interval_s giây trên audio im lặng (WebRTC = False).
    Tự động gọi callback on_env_change khi phát hiện môi trường thay đổi.
    """

    # ================= LABEL MAP =================
    # Map PANNS label → preset (quiet / normal / noisy)
    # Chỉnh sửa nếu môi trường thực tế của bạn trả về label khác

    LABEL_MAP = {
        # Quiet
        "Silence":                  "quiet",
        "Inside, small room":       "quiet",
        "Hum":                      "quiet",
        "White noise":              "quiet",
        "Pink noise":               "quiet",

        # Normal
        "Office":                   "normal",
        "Typing":                   "normal",
        "Computer keyboard":        "normal",
        "Printer":                  "normal",
        "Speech":                   "normal",
        "Conversation":             "normal",
        "Female speech, woman speaking": "normal",
        "Male speech, man speaking":     "normal",
        "Inside, large room or hall":    "normal",

        # Noisy
        "Noise":                    "noisy",
        "Background noise":         "noisy",
        "Babbling":                 "noisy",
        "Crowd":                    "noisy",
        "Hubbub, speech noise, speech babble": "noisy",
        "Restaurant":               "noisy",
        "Music":                    "noisy",
        "Traffic noise, roadway noise": "noisy",
        "Car":                      "noisy",
        "Vehicle":                  "noisy",
        "Horn":                     "noisy",
        "Wind":                     "noisy",
        "Rain":                     "noisy",
        "Thunder":                  "noisy",
        "Construction":             "noisy",
        "Drill":                    "noisy",
        "Chatter":                  "noisy",
    }

    # ================= PRESETS =================

    PRESETS = {
        "quiet": {
            "vad_threshold":      0.40,
            "min_silence_ms":     350,
            "webrtc_vad_mode":    0,
            "highpass_hz":        60,
            "lowpass_hz":         7800,
        },
        "normal": {
            "vad_threshold":      0.45,
            "min_silence_ms":     450,
            "webrtc_vad_mode":    1,
            "highpass_hz":        80,
            "lowpass_hz":         7500,
        },
        "noisy": {
            "vad_threshold":      0.60,
            "min_silence_ms":     650,
            "webrtc_vad_mode":    3,
            "highpass_hz":        120,
            "lowpass_hz":         6800,
        },
    }

    # Cần ít nhất N giây audio để classify đáng tin cậy
    _MIN_AUDIO_S   = 2.0
    _SAMPLE_RATE   = 16000
    _MIN_SAMPLES   = int(_MIN_AUDIO_S * _SAMPLE_RATE)

    # Giữ lịch sử N lần classify gần nhất để vote (tránh flip liên tục)
    _VOTE_WINDOW   = 3

    def __init__(
        self,
        on_env_change,
        eval_interval_s: float = 5.0,
        device: str = "cuda",
    ):
        """
        on_env_change : callback(preset: str, label: str) — gọi khi môi trường đổi
        eval_interval_s : bao lâu classify 1 lần (giây)
        device          : "cpu" hoặc "cuda"
        """
        self.on_env_change   = on_env_change
        self.eval_interval_s = eval_interval_s

        self._audio_buf  = []
        self._buf_lock   = threading.Lock()
        self._current    = "normal"
        self._history    = []           # lịch sử preset để vote
        self._running    = False

        console.print("[cyan]⏳ Tải PANNS AudioTagging...[/cyan]")
        self._model      = AudioTagging(checkpoint_path=None, device=device)
        self._labels     = self._load_labels()
        console.print("[green]✅ PANNS OK[/green]")

    # ================= LABEL LOADER =================

    @staticmethod
    def _load_labels() -> list[str]:
        """
        Tải class names AudioSet 527 từ file CSV của PANNS.
        Tự download nếu chưa có.
        """
        import os
        import csv
        import urllib.request

        path = "audioset_class_labels.csv"
        url  = (
            "https://raw.githubusercontent.com/qiuqiangkong/"
            "audioset_tagging_cnn/master/metadata/class_labels_indices.csv"
        )

        if not os.path.exists(path):
            console.print("[yellow]📥 Downloading AudioSet class labels...[/yellow]")
            urllib.request.urlretrieve(url, path)

        with open(path, newline="", encoding="utf-8") as f:
            return [row["display_name"] for row in csv.DictReader(f)]

    # ================= AUDIO BUFFER =================

    def push_audio(self, block: np.ndarray) -> None:
        """
        Nhận block audio im lặng từ VAD thread (WebRTC = False).
        Chỉ gọi khi không có giọng nói để tránh classify nhầm giọng thành noise.
        """
        with self._buf_lock:
            self._audio_buf.append(block.copy())

    # ================= CLASSIFY =================

    def _classify_once(self) -> tuple[str, str]:

        with self._buf_lock:
            if not self._audio_buf:
                return self._current, "N/A"

            audio = np.concatenate(self._audio_buf)
            self._audio_buf.clear()

        if len(audio) < self._MIN_SAMPLES:
            return self._current, "N/A"

        inp = audio[np.newaxis, :].astype(np.float32)

        _, clipwise = self._model.inference(inp)
        scores      = clipwise[0]

        # Giới hạn index không vượt quá số labels thực tế
        n_classes   = min(len(scores), len(self._labels))
        scores      = scores[:n_classes]

        top_indices = np.argsort(scores)[-10:][::-1]
        top_labels  = [(self._labels[i], float(scores[i])) for i in top_indices]

        for label, score in top_labels:
            if label in self.LABEL_MAP:
                return self.LABEL_MAP[label], label

        return self._current, top_labels[0][0]

    def _vote(self, new_preset: str) -> str:
        """
        Giữ lịch sử _VOTE_WINDOW lần gần nhất.
        Trả về preset thắng đa số — tránh flip liên tục.
        """
        self._history.append(new_preset)

        if len(self._history) > self._VOTE_WINDOW:
            self._history.pop(0)

        return max(set(self._history), key=self._history.count)

    # ================= BACKGROUND LOOP =================

    def _loop(self) -> None:

        while self._running:

            time.sleep(self.eval_interval_s)

            raw_preset, top_label = self._classify_once()

            if raw_preset == "N/A":
                continue

            voted_preset = self._vote(raw_preset)

            if voted_preset != self._current:
                self._current = voted_preset
                self.on_env_change(voted_preset, top_label)

    # ================= PUBLIC =================

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self) -> None:
        self._running = False

    @property
    def current_preset(self) -> str:
        return self._current