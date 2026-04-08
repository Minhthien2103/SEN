"""
FILE: speech.py
MÔ TẢ:
    Module tổng hợp toàn bộ chức năng âm thanh của hệ thống SEN.
    Quản lý luồng dữ liệu song song (Pipeline) bao gồm:
    - Nhận diện và khử nhiễu giọng nói (DeepFilterNet + VAD).
    - Chuyển đổi giọng nói thành văn bản (Groq Whisper STT).
    - Phân tích cảm xúc âm thanh (SER / HuBERT).
    - Trích xuất dữ liệu khẩu hình miệng (Rhubarb Lip-sync).
    - Tổng hợp văn bản thành giọng nói (Edge-TTS).
"""

import os
# Ngăn chặn thư viện HuggingFace in ra các cảnh báo symlink không cần thiết làm trôi log hệ thống
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import asyncio
import io
import queue
import re
import tempfile
import threading
import time
import wave
from collections import deque
from typing import Callable
from pydub import AudioSegment

import numpy as np
import soundfile as sf
import edge_tts
import torch
import torchaudio.functional as F
from dotenv import load_dotenv
from groq import Groq

# Import bộ xử lý não bộ và cấu hình chuẩn đã thống nhất
from backend.core.brain import EmotionPredictor
from backend.core.emotion_config import EmotionConfig
from backend.modules.audio.Env_classifier import EnvironmentClassifier
from backend.modules.audio.SER import AudioToneAnalyzer

import json
import subprocess

load_dotenv()

try:
    from rich.console import Console
    console = Console()
except ImportError:
    raise SystemExit("Lỗi: Hãy chạy lệnh 'pip install rich'")

try:
    import sounddevice as sd
except ImportError:
    console.print("[red]Lỗi: Hãy chạy lệnh 'pip install sounddevice'[/red]")
    raise SystemExit(1)

try:
    import webrtcvad
except ImportError:
    console.print("[red]Lỗi: Hãy chạy lệnh 'pip install webrtcvad-wheels'[/red]")
    raise SystemExit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resample_np(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Chuyển đổi tần số lấy mẫu (Sample Rate) của mảng âm thanh.

    Args:
        audio (np.ndarray): Mảng âm thanh thô.
        orig_sr (int): Tần số lấy mẫu gốc.
        target_sr (int): Tần số lấy mẫu mục tiêu.

    Returns:
        np.ndarray: Mảng âm thanh sau khi chuyển đổi tần số.
    """
    # Bỏ qua quá trình tính toán tensor tốn tài nguyên nếu tần số đã khớp
    if orig_sr == target_sr:
        return audio
    
    t = torch.from_numpy(audio).unsqueeze(0)
    t = F.resample(t, orig_sr, target_sr)
    return t.squeeze(0).numpy()


# ── Khẩu Hình Miệng (Lip-sync) ────────────────────────────────────────────────

def generate_mouth_cues(audio_bytes: bytes, rhubarb_path: str = "./rhubarb.exe") -> list:
    """
    Phân tích file âm thanh để tạo dữ liệu đồng bộ khẩu hình miệng (Lip-sync) cho Engine Unity.

    Args:
        audio_bytes (bytes): Dữ liệu âm thanh thô dạng byte.
        rhubarb_path (str): Đường dẫn đến file thực thi Rhubarb.

    Returns:
        list: Danh sách các khung hình miệng (mouthCues) đã được canh thời gian.
    """
    # Gắn timestamp vào tên file tạm để tránh xung đột I/O khi hệ thống xử lý nhiều request âm thanh cùng lúc
    timestamp = int(time.time() * 1000)
    temp_wav = f"temp_rhubarb_{timestamp}.wav"
    temp_json = f"temp_rhubarb_{timestamp}.json"
    
    mouth_cues = []
    
    try:
        # Rhubarb engine bắt buộc đọc luồng âm thanh từ file vật lý thay vì buffer trên RAM
        with open(temp_wav, "wb") as f:
            f.write(audio_bytes)
            
        command = [
            rhubarb_path,
            "-f", "json",
            "--machineReadable",
            "-o", temp_json,
            temp_wav
        ]
        
        # Chạy quy trình ẩn (DEVNULL) để Rhubarb không đẩy rác log ra màn hình Console chính
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        with open(temp_json, 'r', encoding='utf-8') as f:
            rhubarb_data = json.load(f)
            
        # Key "mouthCues" được hardcode theo đúng chuẩn camelCase mà C# Unity Client đang mong đợi
        mouth_cues = rhubarb_data.get("mouthCues", [])
        
    except Exception as e:
        console.print(f"[red]Lỗi tạo khẩu hình miệng (Rhubarb): {e}[/red]")
    finally:
        # Giải phóng ổ cứng ngay lập tức để tránh tràn đĩa khi Server chạy thời gian dài
        if os.path.exists(temp_wav): os.remove(temp_wav)
        if os.path.exists(temp_json): os.remove(temp_json)
        
    return mouth_cues


# ── Bộ Lọc Nhiễu ──────────────────────────────────────────────────────────────

class DeepFilterDenoiser:
    """
    Bộ lọc nhiễu âm thanh sử dụng mạng nơ-ron học sâu (DeepFilterNet).
    Tối ưu để hoạt động thời gian thực với độ trễ thấp, tăng cường độ rõ 
    của giọng nói trước khi đưa vào module STT.
    """
    
    DF_SR = 48_000

    def __init__(self):
        try:
            from df.enhance import enhance, init_df
        except ImportError:
            console.print("[red]Lỗi: Hãy chạy lệnh 'pip install deepfilternet'[/red]")
            raise SystemExit(1)

        self._enhance = enhance
        self._model, self._df_state, _ = init_df()

    def process(self, audio_f32: np.ndarray, sr: int) -> np.ndarray:
        """
        Khử tiếng ồn nền cho đoạn âm thanh đầu vào.

        Args:
            audio_f32 (np.ndarray): Mảng âm thanh dạng float32.
            sr (int): Tần số lấy mẫu gốc của âm thanh.

        Returns:
            np.ndarray: Mảng âm thanh đã được làm sạch tiếng ồn.
        """
        # DeepFilterNet bắt buộc nhận đầu vào 48kHz để có thể phân tích toàn bộ phổ âm thanh
        audio_48k = _resample_np(audio_f32, sr, self.DF_SR)
        tensor    = torch.from_numpy(audio_48k).unsqueeze(0)
        
        enhanced  = self._enhance(self._model, self._df_state, tensor)
        
        audio_48k = enhanced.squeeze(0).numpy()
        # Ép tần số lấy mẫu về lại chuẩn ban đầu để không làm gián đoạn luồng của module STT
        return _resample_np(audio_48k, self.DF_SR, sr)


# ══════════════════════════════════════════════════════════════════════════════
# STT — SPEECH TO TEXT & AUDIO PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class SpeechToText:
    """
    Module quản lý toàn bộ vòng đời của việc nhận diện giọng nói:
    Lấy dữ liệu từ Mic -> Khử ồn -> Kiểm tra giọng nói (VAD) -> 
    Dịch thành văn bản (Groq) -> Phân tích cảm xúc -> Đẩy cho LLM.
    """

    def __init__(
        self,
        config: dict | None = None,
        on_transcript: Callable[[str], None] | None = None,
        tts=None,
    ):
        """
        Khởi tạo hệ thống STT và các model phụ trợ.

        Args:
            config (dict): Từ điển chứa các cấu hình cho hệ thống âm thanh (như sample rate, ngưỡng VAD...).
            on_transcript (Callable): Callback được gọi khi có văn bản mới được dịch ra.
            tts (Object): Tham chiếu đến đối tượng TextToSpeech để chặn ghi âm khi bot đang nói (chống tiếng vọng).
        """
        self.on_transcript = on_transcript
        self.console       = Console()
        self.tts           = tts

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key: 
            raise ValueError("Không tìm thấy GROQ_API_KEY trong file .env")
        self.groq_client = Groq(api_key=api_key)

        self.config = config or {
            # Tần số lấy mẫu 16kHz là mức tiêu chuẩn tối ưu cho các mô hình AI âm thanh hiện tại (VAD, Whisper, HuBERT).
            "sample_rate":        16_000, 
            "language":           "vi",
            
            # Ngưỡng VAD: Quyết định độ nhạy của việc cắt câu. Càng cao càng ít bị cắt nhầm bởi tiếng ồn, 
            # nhưng nếu cao quá có thể bỏ sót tiếng thì thầm. Sẽ được module Env_classifier tự động điều chỉnh.
            "vad_threshold":      0.50, 
            
            # Thời gian yên lặng tối thiểu để hệ thống hiểu là người dùng đã ngắt câu và bắt đầu dịch.
            "min_silence_ms":     650, 
            
            # Loại bỏ các tiếng tặc lưỡi, ho hay tiếng ồn ngắn (dưới 250ms) không phải là lời nói có ý nghĩa.
            "min_speech_ms":      250,
            
            # Nếu người dùng nói một tràng quá dài (hơn 8 giây), cắt ngang để dịch trước giúp hệ thống phản hồi mượt hơn.
            "early_transcribe_s": 8.0, 
            
            "webrtc_vad_mode":    2,
            "highpass_hz":        80,
            "lowpass_hz":         7_500,
            "save_to_file":       False,
            "output_file":        "transcript.txt",
            "save_audio":         False,
            "audio_output_dir":   "recorded_audio",
        }

        self.block_size        = 512
        self._webrtc_frame_len = 320

        self.audio_q    = queue.Queue()
        self.segment_q  = queue.Queue()
        self.is_running = False

        self._mic_muted = threading.Event()

        self.vad_model  = None
        self.webrtc_vad = None
        self.denoiser   = None

        self.collected_audio = []
        self._seg_counter    = 0
        self._session_tag    = time.strftime("%Y%m%d_%H%M%S")
        self._prev_transcripts = deque(maxlen=6)

        self.emotion_model    = EmotionPredictor()
        self.tone_model       = AudioToneAnalyzer(debug=True)
        self.env_classifier   = EnvironmentClassifier(on_env_change=self._on_env_change)
        self._pending_env_log = []
        self._env_log_lock    = threading.Lock()

        # Bộ lọc Regular Expression dùng để loại bỏ hiện tượng "ảo giác" (hallucination) cực kỳ phổ biến 
        # của model Whisper khi nó cố gắng chèn các câu kết thúc YouTube ngẫu nhiên vào đoạn âm thanh nhiễu.
        self.hallucination_patterns = [
            r"^\s*$",
            r"^\W+$",
            r"^(Thanks for watching|Subscribe|Like and subscribe|Please subscribe"
            r"|Hãy subscribe cho kênh|Hẹn gặp lại|Cảm ơn các bạn"
            r"|Cảm ơn bạn đã xem|Đừng quên đăng ký|Nhớ nhấn like"
            r"|Xin chào các bạn|Chúc các bạn|Tạm biệt)",
            r"^\.{2,}$",
            r"^-{2,}$",
            r"ghiền mì gô", 
            r"đăng ký kênh", 
            r"nhớ đăng ký", 
            r"subscribe", 
            r"bấm sub",
            r"follow", 
            r"theo dõi", 
            r"cảm ơn các bạn", 
            r"hẹn gặp lại", 
            r"chào mừng các bạn",
            r"ấn chuông", 
            r"like và share", 
            r"ủng hộ mình",
            r"xem tiếp phần", 
            r"tập \d+",
            r"bản tin",
            r"thời sự",
            r"chúc các bạn"
        ]

    # ── Mic Gate ──────────────────────────────────────────────────────────────

    def mute_mic(self) -> None:
        """Kích hoạt cờ khóa luồng dữ liệu từ microphone."""
        self._mic_muted.set()

    def unmute_mic(self) -> None:
        """Xóa sạch hàng đợi âm thanh cũ để tránh dịch nhầm tiếng vọng, sau đó mở khóa mic."""
        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break
        self._mic_muted.clear()

    # ── Model Load ────────────────────────────────────────────────────────────

    def load_models(self) -> None:
        self.console.print("[cyan]Khởi tạo WebRTC VAD...[/cyan]")
        self.webrtc_vad = webrtcvad.Vad(self.config["webrtc_vad_mode"])
        self.console.print("[green]WebRTC VAD OK[/green]")

        self.console.print("[cyan]Tải Silero VAD...[/cyan]")
        self.vad_model = self._load_silero_vad_local()
        self.vad_model.eval()
        self.console.print("[green]Silero VAD OK[/green]")

        self.console.print("[cyan]Tải DeepFilterNet...[/cyan]")
        self.denoiser = DeepFilterDenoiser()
        self.console.print("[green]DeepFilterNet OK[/green]")

        self.env_classifier.start()

    # ── Silero Cache ──────────────────────────────────────────────────────────
    
    def _load_silero_vad_local(self):
        """
        Lưu cache mô hình Silero VAD vào ổ đĩa nội bộ (JIT Compiled).
        Mục đích: Tăng tốc độ khởi động ở các lần chạy sau và cho phép hệ thống
        vẫn hoạt động kể cả khi mất kết nối mạng.
        """
        hub_dir    = torch.hub.get_dir()
        model_path = os.path.join(hub_dir, "silero_vad_cached.jit")

        if not os.path.exists(model_path):
            self.console.print("[yellow]Caching Silero VAD...[/yellow]")
            model, _ = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                force_reload=False,
                onnx=False,
            )
            torch.jit.save(model, model_path)
            return model

        return torch.jit.load(model_path, map_location="cpu")

    # ── Audio Callback ────────────────────────────────────────────────────────
    
    def audio_callback(self, indata, frames, time_info, status) -> None:
        """
        Hàm callback bất đồng bộ của thư viện sounddevice.
        Trực tiếp nhận từng gói dữ liệu nhỏ từ phần cứng Microphone.
        """
        block = indata[:, 0].copy().astype(np.float32)

        # Chuyển tiếp âm thanh sang luồng phân tích môi trường để auto-tune ngưỡng VAD.
        if not self._mic_muted.is_set() and hasattr(self, "env_classifier"):
            self.env_classifier.push_audio(block)

        if self._mic_muted.is_set():
            return

        self.audio_q.put(block)

    # ── VAD Logic ─────────────────────────────────────────────────────────────
    
    def _webrtc_is_speech(self, samples_f32: np.ndarray) -> bool:
        """
        Sử dụng thuật toán kinh điển WebRTC VAD (cực nhẹ và nhanh) để lọc bước 1.
        Nếu WebRTC kết luận không có tiếng người, hệ thống sẽ bỏ qua đoạn âm thanh này
        để không tốn tài nguyên chạy mô hình AI Silero VAD nặng hơn.
        """
        pcm_int16 = (samples_f32 * 32767).clip(-32768, 32767).astype(np.int16)
        sr        = self.config["sample_rate"]

        if len(pcm_int16) >= 480:
            frame = pcm_int16[-480:].tobytes()
        else:
            return False

        try:
            return self.webrtc_vad.is_speech(frame, sr)
        except Exception:
            return True
            
    def _vad_prob(self, block: np.ndarray) -> float:
        """Lọc bước 2: Dùng mạng nơ-ron Silero VAD để tính xác suất thực sự là tiếng người."""
        t = torch.FloatTensor(block)
        with torch.no_grad():
            return self.vad_model(t, 16000).item()

    def vad_thread(self) -> None:
        """
        Luồng nền liên tục giám sát và cắt các khối âm thanh.
        Nó quyết định khi nào người dùng bắt đầu nói, và khi nào họ đã nói xong
        để đóng gói lại thành một đoạn ghi âm hoàn chỉnh gửi đi dịch.
        """
        SR                = self.config["sample_rate"]
        min_sil_chunks    = int(self.config["min_silence_ms"] / 1000 * SR / self.block_size)
        early_chunks      = int(self.config["early_transcribe_s"] * SR / self.block_size)
        min_speech_blocks = int(self.config.get("min_speech_ms", 250) / 1000 * SR / self.block_size)

        speech_buf          = []
        sil_count           = 0
        speech_blocks_count = 0
        in_speech           = False
        carry               = np.array([], dtype=np.float32)
        
        # Buffer đệm: Lưu trữ 0.4 giây âm thanh ngay trước khi VAD phát hiện ra tiếng người.
        # Lý do: Con người thường nói âm tiết đầu tiên khá nhỏ hoặc bị nhiễu, nếu không có đệm này
        # chữ cái đầu tiên của câu sẽ luôn luôn bị cắt mất.
        pre_speech_pad      = deque(maxlen=int(0.4 * SR / self.block_size))

        self.vad_model.reset_states()

        while self.is_running:
            try:
                raw = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # Anti-echo: Tạm dừng lắng nghe khi SEN đang phát giọng nói qua loa.
            if self.tts and self.tts.is_speaking():
                continue

            combined = np.concatenate([carry, raw])
            n        = len(combined) // self.block_size
            carry    = combined[n * self.block_size:]

            for i in range(n):
                blk           = combined[i * self.block_size:(i + 1) * self.block_size]
                webrtc_speech = self._webrtc_is_speech(blk) 

                if not webrtc_speech and not in_speech: 
                    pre_speech_pad.append(blk)
                    continue

                prob = self._vad_prob(blk)
                is_v = prob >= self.config["vad_threshold"]

                if is_v:
                    sil_count = 0
                    if not in_speech: 
                        in_speech           = True
                        speech_buf          = list(pre_speech_pad) + [blk]
                        speech_blocks_count = len(speech_buf)
                    else:
                        speech_buf.append(blk) 
                        speech_blocks_count += 1
                else:
                    if not in_speech:
                        pre_speech_pad.append(blk)

                    if in_speech:
                        speech_buf.append(blk)
                        sil_count += 1

                        is_early_cut = (len(speech_buf) >= early_chunks and sil_count >= (min_sil_chunks // 3))
                        is_force_cut = len(speech_buf) >= int(early_chunks * 1.5)

                        # Quyết định cắt câu khi: Đạt giới hạn thời gian im lặng, hoặc câu quá dài (early/force cut)
                        if sil_count >= min_sil_chunks or is_early_cut or is_force_cut:
                            if speech_blocks_count >= min_speech_blocks:
                                self.segment_q.put(np.concatenate(speech_buf))

                            speech_buf          = []
                            sil_count           = 0
                            speech_blocks_count = 0
                            in_speech           = False
                            pre_speech_pad.clear()
                            self.vad_model.reset_states()

    # ── Audio Processing ──────────────────────────────────────────────────────

    def clean_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Xử lý làm sạch tín hiệu âm thanh trước khi đưa vào mô hình STT.
        Sử dụng kết hợp bộ lọc tần số (Biquad Filter) và AI (DeepFilterNet).
        """
        sr = self.config["sample_rate"]
        hp = self.config.get("highpass_hz", 80)
        lp = self.config.get("lowpass_hz", 7_500)

        original = audio.copy()

        # Áp dụng Highpass để loại bỏ tiếng ồn trầm (tiếng quạt máy, tiếng gõ bàn).
        # Áp dụng Lowpass để loại bỏ tần số siêu âm không cần thiết.
        tensor = torch.from_numpy(audio).unsqueeze(0)
        tensor = F.highpass_biquad(tensor, sr, float(hp))
        tensor = F.lowpass_biquad(tensor, sr, float(lp))
        filtered = tensor.squeeze(0).numpy()

        cleaned = self.denoiser.process(filtered, sr)

        # Trộn (Blend) lại một phần âm thanh gốc. 
        # Lý do: DeepFilterNet đôi khi lọc quá mạnh làm giọng bị "méo" như robot. 
        # Giữ lại một chút bản gốc sẽ giúp giọng nói tự nhiên hơn.
        blend_ratio = 0.75
        mixed       = (cleaned * blend_ratio) + (original * (1.0 - blend_ratio))

        # Chuẩn hóa âm lượng (Normalization) để âm thanh không bị rè (clip).
        peak = np.max(np.abs(mixed))
        if peak > 0:
            mixed = mixed / peak * 0.9

        return mixed.astype(np.float32)

    # ── Groq STT ──────────────────────────────────────────────────────────────

    def _build_stt_prompt(self) -> str:
        """
        Xây dựng prompt mồi (prompt injection) cho mô hình Whisper.
        Mục đích: Cung cấp lịch sử hội thoại gần nhất giúp Whisper đoán đúng 
        ngữ cảnh, từ lóng hoặc tên riêng mà người dùng vừa đề cập.
        """
        if not self._prev_transcripts:
            return "Bạn là SEN, một trợ lý ảo tiếng Việt, trò chuyện hằng ngày tự nhiên."

        prev = "\n".join(f"- {t}" for t in list(self._prev_transcripts)[-4:])
        return (
            "Bạn là SEN, một trợ lý ảo tiếng Việt, trò chuyện hằng ngày tự nhiên.\n"
            "Ngữ cảnh (các câu trước đó của người dùng):\n"
            f"{prev}\n"
            "Hãy ưu tiên phiên âm đúng tên riêng/thuật ngữ theo ngữ cảnh."
        )

    def groq_transcribe(self, audio: np.ndarray) -> str:
        """
        Gửi đoạn âm thanh lên API Groq để chuyển đổi thành văn bản.
        Sử dụng tempfile vì SDK của Groq yêu cầu đọc luồng từ một file vật lý.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            self._save_wav(f.name, audio)
            with open(f.name, "rb") as audio_file:
                transcript = self.groq_client.audio.transcriptions.create(
                    file        = audio_file,
                    model       = "whisper-large-v3-turbo",
                    language    = "vi",
                    temperature = 0.0, # Giữ 0.0 để kết quả dịch chính xác nhất, không bịa từ (hallucination)
                    prompt      = self._build_stt_prompt(),
                )
        return transcript.text

    # ── Env Callback ──────────────────────────────────────────────────────────
    
    def _on_env_change(self, preset: str, old: str, label: str) -> None:
        """Tự động cập nhật các thông số độ nhạy VAD khi môi trường âm thanh thay đổi."""
        for k, v in EnvironmentClassifier.PRESETS[preset].items():
            self.config[k] = v
        msg = f"[magenta]Môi trường: {old} -> {preset} ({label})[/magenta]"
        with self._env_log_lock:
            self._pending_env_log.append(msg)

    def _flush_env_log(self) -> None:
        """In log môi trường ra màn hình. Dùng lock để tránh đụng độ luồng khi in."""
        with self._env_log_lock:
            logs, self._pending_env_log = self._pending_env_log, []
        for msg in logs:
            self.console.print(msg)

    # ── Hallucination Filter ──────────────────────────────────────────────────

    def _is_hallucination(self, text: str) -> bool:
        """Quét và chặn các câu văn bản vô nghĩa do lỗi mô hình Whisper sinh ra."""
        return any(re.search(pat, text, re.IGNORECASE) for pat in self.hallucination_patterns)

    # ── Transcribe Thread ─────────────────────────────────────────────────────
    
    def transcribe_thread(self) -> None:
        """
        Luồng trung tâm xử lý khối lượng công việc nặng nhất:
        Nhận âm thanh -> Làm sạch -> Dịch (STT) -> Phân tích Cảm xúc -> Gọi Callback.
        """
        while self.is_running:
            try:
                audio = self.segment_q.get(timeout=0.5)
            except queue.Empty:
                continue
            
            # Lọc bỏ các tiếng click, tiếng thở quá ngắn (dưới 0.4s) 
            # để không lãng phí API request lên Groq.
            if len(audio) < int(self.config["sample_rate"] * 0.4):
                continue

            clean = self.clean_audio(audio)
            self.collected_audio.append(clean)

            if self.config.get("save_audio", False):
                try:
                    self._save_segment_wav(clean)
                except Exception as e:
                    self.console.print(f"[red]Audio save error: {e}[/red]")

            start = time.time()
            try:
                text = self.groq_transcribe(clean)
            except Exception as e:
                self.console.print(f"[red]Groq error: {e}[/red]")
                continue

            elapsed = time.time() - start

            if not text:
                continue

            text = text.strip()

            # Lọc rác trước khi đưa vào trí nhớ LLM
            if not text or self._is_hallucination(text):
                continue

            # Chuẩn hóa viết hoa chữ cái đầu
            text = text[0].upper() + text[1:]
            self._prev_transcripts.append(text)

            self.console.print(
                f"\n[bold yellow][VI][/bold yellow] "
                f"[white]{text}[/white]  "
                f"[dim]({elapsed:.2f}s | {len(audio) / self.config['sample_rate']:.1f}s audio)[/dim]"
            )

            if self.config["save_to_file"]:
                with open(self.config["output_file"], "a", encoding="utf-8") as f:
                    f.write(f"[VI] {text}\n")

            # ── Nhận Diện Cảm Xúc (Văn Bản) ──
            emotion     = self.emotion_model.predict(text)
            probs_by_id = emotion.get("probs_by_id")

            if isinstance(probs_by_id, dict) and all(i in probs_by_id for i in range(7)):
                id2name      = getattr(self.emotion_model, "ID2EMOTION", {})
                probs_str    = ", ".join(
                    f"{i}.{id2name.get(i, str(i))}: {float(probs_by_id.get(i, 0.0)):.3f}"
                    for i in range(7)
                )
                dominant_out = str(emotion.get("dominant_id", 6))
            else:
                probs_str    = ", ".join(
                    f"{label}: {emotion.get(label, 0.0):.3f}"
                    for label in getattr(self.emotion_model, "labels", [])
                )
                dominant_out = str(emotion.get("dominant", "neutral"))

            # ── Nhận Diện Cảm Xúc (Âm Thanh) ──
            tone = self.tone_model.analyze(audio, self.config["sample_rate"])

            # Chuẩn hóa nhãn tiếng Việt bằng File Config chung
            dominant_vi = EmotionConfig.VI_MAP.get(dominant_out, dominant_out)

            if probs_str:
                self.console.print(f"[cyan]Text Emotion:[/cyan] {dominant_vi} [dim]({probs_str})[/dim]")
            else:
                self.console.print(f"[cyan]Text Emotion:[/cyan] {dominant_vi}")

            tone_probs_str = ", ".join(
                f"{k}: {v:.3f}" for k, v in tone.items()
                if k not in ("dominant", "human_readable", "confidence")
            )
            self.console.print(
                f"[magenta]Audio Tone:[/magenta] {tone['human_readable']} "
                f"[dim]({tone_probs_str})[/dim]"
            )

            self._flush_env_log()

            if self.on_transcript:
                try:
                    self.on_transcript(text)
                except Exception as e:
                    self.console.print(f"[red]Reply error: {e}[/red]")

            self._flush_env_log()

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Khởi động toàn bộ các thành phần AI và mở kết nối Microphone."""
        self.load_models()
        self.is_running = True

        threading.Thread(target=self.vad_thread,        daemon=True).start()
        threading.Thread(target=self.transcribe_thread, daemon=True).start()

        self.console.print("\n[green]Đang lắng nghe...[/green]\n")

        try:
            with sd.InputStream(
                samplerate = self.config["sample_rate"],
                channels   = 1,
                dtype      = "float32",
                blocksize  = self.block_size,
                callback   = self.audio_callback,
                latency    = "low",
            ):
                while True:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            self.console.print("\n[red]Đã dừng hệ thống STT[/red]")
            self.is_running = False
            try:
                self._save_full_session()
            except Exception as e:
                self.console.print(f"[red]Full session save error: {e}[/red]")

    # ── Save WAV Utils ────────────────────────────────────────────────────────

    def _save_wav(self, path: str, audio: np.ndarray) -> None:
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

    def _save_full_session(self) -> None:
        if not self.config["save_audio"] or not self.collected_audio:
            return
        out_dir   = self.config["audio_output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        full_path = os.path.join(out_dir, f"{self._session_tag}_full_session.wav")
        combined  = np.concatenate(self.collected_audio)
        self._save_wav(full_path, combined)
        self.console.print(f"[green]Đã lưu toàn bộ phiên thu âm tại: {full_path}[/green]")


# ══════════════════════════════════════════════════════════════════════════════
# TTS — TEXT TO SPEECH
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_VOICE  = "vi-VN-HoaiMyNeural"
DEFAULT_RATE   = "+30%"
DEFAULT_VOLUME = "+0%"
DEFAULT_PITCH  = "+0Hz"


async def _synthesize(text: str, voice: str, rate: str, volume: str, pitch: str) -> bytes:
    """
    Gọi API Edge-TTS để chuyển đổi văn bản thành âm thanh (định dạng MP3).
    Sau đó, bắt buộc phải dịch mã (transcode) từ MP3 sang chuẩn PCM 16-bit WAV.
    Lý do: Cả engine phân tích khẩu hình miệng (Rhubarb) và Client Unity 
    đều yêu cầu chuẩn âm thanh không nén (WAV) để xử lý chính xác thời gian thực.
    """
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
    
    try:
        audio = AudioSegment.from_mp3(mp3_buf)
        
        # Ép chuẩn WAV 16-bit (Đảm bảo độ phân giải âm thanh tiêu chuẩn, tránh lỗi đọc file từ hệ thống khác)
        audio = audio.set_sample_width(2) 
        
        wav_buf = io.BytesIO()
        audio.export(wav_buf, format="wav")
        
        return wav_buf.getvalue()
    except Exception as e:
        console.print(f"[red]Lỗi khi dịch mã MP3 sang WAV: {e}[/red]")
        # Fallback: Trả về luồng MP3 gốc nếu quá trình ép chuẩn thất bại để không làm sập luồng đọc
        return mp3_buf.read()


def _play_audio(audio_bytes: bytes) -> None:
    """Đẩy trực tiếp luồng byte âm thanh ra loa hệ thống và chặn luồng cho đến khi phát xong."""
    buf  = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(buf, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()


# ── TextToSpeech ──────────────────────────────────────────────────────────────

class TextToSpeech:
    """
    Quản lý luồng tổng hợp và phát âm thanh đa luồng (Multi-threading).
    Đảm bảo việc tải âm thanh từ Internet không làm đứng giao diện hoặc luồng nhận diện giọng nói.
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

        self._stt         = None
        self._q:          queue.Queue[str | None]   = queue.Queue()
        self._audio_q:    queue.Queue[bytes | None] = queue.Queue()
        self._is_speaking = threading.Event()
        self._running     = True
        self.text_buffer  = ""

        self._worker_synth = threading.Thread(target=self._synth_loop, daemon=True, name="TTS-synth")
        self._worker_play  = threading.Thread(target=self._play_loop,  daemon=True, name="TTS-play")
        self._worker_synth.start()
        self._worker_play.start()

        console.print(f"[green]TTS sẵn sàng - Giọng: {voice}[/green]")

    # ── STT Link ──────────────────────────────────────────────────────────────

    def set_stt(self, stt) -> None:
        """Kết nối module TTS với module STT để đồng bộ trạng thái khóa mic (chống tiếng vọng)."""
        self._stt = stt

    # ── Streaming API ─────────────────────────────────────────────────────────
    
    def stream_token(self, token: str) -> None:
        """
        Nhận từng token văn bản từ LLM và dồn vào bộ đệm.
        Mục đích: Tăng tốc độ phản hồi. Thay vì đợi LLM sinh xong toàn bộ câu trả lời,
        hệ thống sẽ kích hoạt đọc ngay lập tức mỗi khi gặp dấu ngắt câu (chấm, phẩy...).
        """
        if not token:
            return
        self.text_buffer += token
        match = re.search(r'([.,;!?\n]+)', self.text_buffer)
        if match:
            split_point      = match.end()
            chunk            = self.text_buffer[:split_point]
            self.text_buffer = self.text_buffer[split_point:]
            self.speak(chunk)

    def flush_stream(self) -> None:
        """Xả toàn bộ nội dung còn sót lại trong bộ đệm ra loa khi LLM kết thúc câu trả lời."""
        if self.text_buffer.strip():
            self.speak(self.text_buffer)
        self.text_buffer = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Đẩy một câu văn bản vào hàng đợi để worker chạy nền tải âm thanh về."""
        cleaned = self._clean_text(text)
        if cleaned and any(c.isalnum() for c in cleaned):
            self._q.put(cleaned.strip())

    def speak_wait(self, text: str) -> None:
        """
        Phát âm thanh ở luồng chính (Main Thread) và buộc hệ thống chờ cho đến khi phát xong.
        Thường dùng cho các câu chào cố định lúc hệ thống vừa khởi động.
        """
        cleaned = self._clean_text(text)
        if not cleaned or not any(c.isalnum() for c in cleaned):
            return
        audio = asyncio.run(
            _synthesize(cleaned.strip(), self.voice, self.rate, self.volume, self.pitch)
        )
        _play_audio(audio)

    def is_speaking(self) -> bool:
        """Kiểm tra xem hệ thống có đang bận phát âm thanh hay không."""
        return self._is_speaking.is_set()

    def stop(self) -> None:
        self._running = False
        self._q.put(None)
        self._audio_q.put(None)
        self._worker_synth.join(timeout=3)
        self._worker_play.join(timeout=3)
        console.print("[red]Hệ thống TTS đã dừng[/red]")

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Lọc bỏ các ký tự đặc biệt (như dấu ngoặc kép) để tránh API Edge-TTS báo lỗi cú pháp."""
        if not text:
            return ""
        return re.sub(r'["""'']', "", text)

    def _synth_loop(self) -> None:
        """
        Luồng nền 1 (Worker): Chuyên trách gọi API mạng để tải file âm thanh.
        Tách biệt với luồng phát để tránh làm giật/lag tiếng khi mạng chậm.
        """
        while self._running:
            text = self._q.get()
            if text is None:
                self._audio_q.put(None)
                break
            
            # Khóa mic STT ngay từ khi bắt đầu tổng hợp để chuẩn bị phát loa
            if self._stt:
                self._stt.mute_mic() 
                
            try:
                audio = asyncio.run(
                    _synthesize(text, self.voice, self.rate, self.volume, self.pitch)
                )
                self._audio_q.put(audio)
            except Exception as e:
                console.print(f"[red]Lỗi tổng hợp TTS: {e}[/red]")

    def _play_loop(self) -> None:
        """
        Luồng nền 2 (Worker): Chuyên trách đẩy byte âm thanh ra phần cứng loa.
        """
        while self._running:
            audio = self._audio_q.get()
            if audio is None:
                break
            try:
                self._is_speaking.set()
                _play_audio(audio)

            except Exception as e:
                console.print(f"[red]Lỗi phát TTS: {e}[/red]")
            finally:
                self._is_speaking.clear()
                # Chỉ mở khóa mic khi toàn bộ hàng đợi âm thanh đã được phát hết
                if self._audio_q.empty() and self._stt:
                    self._stt.unmute_mic()