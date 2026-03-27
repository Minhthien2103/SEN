"""
speech.py — Module tổng hợp toàn bộ chức năng âm thanh của SEN.

  ┌──────────────────────────────────────────────────────────┐
  │  STT (Speech-To-Text)                                    │
  │  ├─ DeepFilterDenoiser   — khử nhiễu DeepFilterNet       │
  │  └─ SpeechToText         — VAD + Groq Whisper            │
  ├──────────────────────────────────────────────────────────┤
  │  TTS (Text-To-Speech)                                    │
  │  └─ TextToSpeech         — Edge-TTS + streaming token    │
  └──────────────────────────────────────────────────────────┘
"""


# ══════════════════════════════════════════════════════════════════════════════
# STT — SPEECH TO TEXT
# ══════════════════════════════════════════════════════════════════════════════

import os
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

from brain import EmotionPredictor
from core.Env_classifier import EnvironmentClassifier
from core.SER import AudioToneAnalyzer

# Rhubard
import json
import subprocess

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


# ── Helpers ───────────────────────────────────────────────────────────────────
# Thay đổi tốc độ lấy mẫu
# Vd: chuyển từ 16kHz → 48kHz để phù hợp với yêu cầu của DeepFilterNet
def _resample_np(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    t = torch.from_numpy(audio).unsqueeze(0)
    t = F.resample(t, orig_sr, target_sr)
    return t.squeeze(0).numpy()


# ── DeepFilterNet Wrapper ─────────────────────────────────────────────────────
def generate_mouth_cues(audio_bytes, rhubarb_path="./rhubarb.exe"):
    # Tạo tên file tạm thời không bị trùng lặp
    timestamp = int(time.time() * 1000)
    temp_wav = f"temp_rhubarb_{timestamp}.wav"
    temp_json = f"temp_rhubarb_{timestamp}.json"
    
    mouth_cues = []
    
    try:
        # 1. Ghi byte âm thanh ra file vật lý để Rhubarb có thể đọc
        with open(temp_wav, "wb") as f:
            f.write(audio_bytes)
            
        # 2. Gọi Rhubarb
        command = [
            rhubarb_path,
            "-f", "json",
            "--machineReadable",
            "-o", temp_json,
            temp_wav
        ]
        
        # Chạy ẩn không in log rác ra màn hình (stdout=subprocess.DEVNULL)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 3. Đọc kết quả
        with open(temp_json, 'r', encoding='utf-8') as f:
            rhubarb_data = json.load(f)
            
        # Lưu ý: Trả về key "mouthCues" viết hoa chữ C để khớp 100% với C# Unity
        mouth_cues = rhubarb_data.get("mouthCues", [])
        
    except Exception as e:
        print(f"❌ [Rhubarb Error]: Lỗi tạo khẩu hình miệng: {e}")
    finally:
        # 4. Luôn luôn dọn dẹp rác dù thành công hay thất bại
        if os.path.exists(temp_wav): os.remove(temp_wav)
        if os.path.exists(temp_json): os.remove(temp_json)
        
    return mouth_cues


# ── DeepFilterNet Wrapper ─────────────────────────────────────────────────────
# Lọc Nhiễu
class DeepFilterDenoiser:

    DF_SR = 48_000

    def __init__(self):
        try:
            from df.enhance import enhance, init_df
        except ImportError:
            raise SystemExit("❌ pip install deepfilternet")

        self._enhance = enhance
        self._model, self._df_state, _ = init_df()

    def process(self, audio_f32: np.ndarray, sr: int) -> np.ndarray:
        audio_48k = _resample_np(audio_f32, sr, self.DF_SR)
        tensor    = torch.from_numpy(audio_48k).unsqueeze(0)
        enhanced  = self._enhance(self._model, self._df_state, tensor)
        audio_48k = enhanced.squeeze(0).numpy()
        return _resample_np(audio_48k, self.DF_SR, sr)


# ── SpeechToText ──────────────────────────────────────────────────────────────

class SpeechToText:

    def __init__(
        self,
        config: dict | None = None,
        on_transcript: Callable[[str], None] | None = None,
        tts=None,
    ):
        self.on_transcript = on_transcript
        self.console       = Console()
        self.tts           = tts

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key: 
            raise ValueError("❌ Không tìm thấy GROQ_API_KEY trong file .env")
        self.groq_client = Groq(api_key=api_key)

        self.config = config or {
            "sample_rate":        16_000, # tốc độ lấy mẫu -> mỗi giây thực hiện 16,000 phép đo để ghi lại cường độ của âm thanh
            "language":           "vi",
            "vad_threshold":      0.50, # độ nhạy của silero VAD, tăng độ khắt khe cho bộ lọc. Giá trị càng cao thì càng ít âm thanh được coi là "speech"
            "min_silence_ms":     650, # Khoảng lặng để máy hiểu bạn đã kết thúc câu
            "min_speech_ms":      250,# Một tiếng động phải kéo dài ít nhất 250ms thì mới được coi là lời nói.
            "early_transcribe_s": 8.0, # nói liên tục quá 8 giây mà không nghỉ, máy sẽ tự động cắt đoạn đó ra để xử lý trước
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

        # bộ lọc chặn để ngăn SEN không hiển thị những câu rác này
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
        self._mic_muted.set()

    def unmute_mic(self) -> None:
        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break
        self._mic_muted.clear()

    # ── Model Load ────────────────────────────────────────────────────────────

    def load_models(self) -> None:
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

        self.env_classifier.start()

    # ── Silero Cache ──────────────────────────────────────────────────────────
    # Tải mô hình Silero VAD và lưu nó vào ổ cứng máy tính để dùng lại thay vì phải tải từ trên mạng mỗi khi bạn mở chương trình.
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

    # ── Audio Callback ────────────────────────────────────────────────────────
    # lấy âm thanh trực tiếp từ microphone của bạn và phân phối đến các bộ phận xử lý khác nhau
    def audio_callback(self, indata, frames, time_info, status) -> None:
        block = indata[:, 0].copy().astype(np.float32)

        # Nếu không tắt mic (not _mic_muted) và có bộ phân loại môi trường, nó sẽ đẩy đoạn âm thanh này vào bộ Environment Classifier 
        if not self._mic_muted.is_set() and hasattr(self, "env_classifier"):
            self.env_classifier.push_audio(block)

        if self._mic_muted.is_set():
            return

        self.audio_q.put(block)

    # ── VAD ───────────────────────────────────────────────────────────────────
    # bộ lọc giọng nói sơ cấp: check xem có phải người đang nói hay không
    def _webrtc_is_speech(self, samples_f32: np.ndarray) -> bool:
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
            
    # tính toán xem đoạn âm thanh đó có bao nhiêu phần trăm là tiếng người nói
    # Là check 2 lớp: Nếu WebRTC nghi ngờ có tiếng người, hàm gọi _vad_prob để dùng AI tính xác suất thực tế.
    def _vad_prob(self, block: np.ndarray) -> float:
        t = torch.FloatTensor(block)
        with torch.no_grad():
            return self.vad_model(t, 16000).item()

    def vad_thread(self) -> None:
        SR                = self.config["sample_rate"]
        min_sil_chunks    = int(self.config["min_silence_ms"] / 1000 * SR / self.block_size)
        early_chunks      = int(self.config["early_transcribe_s"] * SR / self.block_size)
        min_speech_blocks = int(self.config.get("min_speech_ms", 250) / 1000 * SR / self.block_size)

        speech_buf          = []
        sil_count           = 0
        speech_blocks_count = 0
        in_speech           = False
        carry               = np.array([], dtype=np.float32)
        pre_speech_pad      = deque(maxlen=int(0.4 * SR / self.block_size))

        self.vad_model.reset_states()

        while self.is_running:
            try:
                raw = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # Nếu SEN đang nói, tắt mic
            if self.tts and self.tts.is_speaking():
                continue

            combined = np.concatenate([carry, raw])
            n        = len(combined) // self.block_size
            carry    = combined[n * self.block_size:]

            for i in range(n):
                # chia thành từng khối để xử lí
                blk           = combined[i * self.block_size:(i + 1) * self.block_size]
                webrtc_speech = self._webrtc_is_speech(blk) # check xem có phải tiếng người hay không bằng WebRTC VAD

                # nếu WebRTC không nghĩ đó là tiếng người và hiện tại cũng không đang trong một đoạn nói nào, thì bỏ qua khối này
                if not webrtc_speech and not in_speech: 
                    pre_speech_pad.append(blk)
                    continue

                prob = self._vad_prob(blk)
                is_v = prob >= self.config["vad_threshold"] # vượt qua webrtc thì tới vad

                if is_v:
                    sil_count = 0
                    if not in_speech: # nếu đang không nói thì lấy cả phần đệm trước đó để tránh bị cắt lời lúc đầu
                        in_speech           = True
                        speech_buf          = list(pre_speech_pad) + [blk]
                        speech_blocks_count = len(speech_buf)
                    else:
                        speech_buf.append(blk) # nếu đang nói rồi thì cứ tiếp tục thêm vào buffer
                        speech_blocks_count += 1
                else:
                    if not in_speech:
                        pre_speech_pad.append(blk)

                    if in_speech:
                        speech_buf.append(blk)
                        sil_count += 1

                        is_early_cut = (len(speech_buf) >= early_chunks and sil_count >= (min_sil_chunks // 3))
                        is_force_cut = len(speech_buf) >= int(early_chunks * 1.5)

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
        sr = self.config["sample_rate"]
        hp = self.config.get("highpass_hz", 80)
        lp = self.config.get("lowpass_hz", 7_500)

        original = audio.copy()

        tensor = torch.from_numpy(audio).unsqueeze(0)
        tensor = F.highpass_biquad(tensor, sr, float(hp))
        tensor = F.lowpass_biquad(tensor, sr, float(lp))
        filtered = tensor.squeeze(0).numpy()

        cleaned = self.denoiser.process(filtered, sr)

        blend_ratio = 0.75
        mixed       = (cleaned * blend_ratio) + (original * (1.0 - blend_ratio))

        peak = np.max(np.abs(mixed))
        if peak > 0:
            mixed = mixed / peak * 0.9

        return mixed.astype(np.float32)

    # ── Groq STT ──────────────────────────────────────────────────────────────

    def _build_stt_prompt(self) -> str:
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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            self._save_wav(f.name, audio)
            with open(f.name, "rb") as audio_file:
                transcript = self.groq_client.audio.transcriptions.create(
                    file        = audio_file,
                    model       = "whisper-large-v3-turbo",
                    language    = "vi",
                    temperature = 0.0,
                    prompt      = self._build_stt_prompt(),
                )
        return transcript.text

    # ── Env Callback ──────────────────────────────────────────────────────────
    # Detect môi trường
    def _on_env_change(self, preset: str, old: str, label: str) -> None:
        for k, v in EnvironmentClassifier.PRESETS[preset].items():
            self.config[k] = v
        msg = f"[magenta]🌍 Môi trường: {old} → {preset} ({label})[/magenta]"
        with self._env_log_lock:
            self._pending_env_log.append(msg)

    def _flush_env_log(self) -> None:
        with self._env_log_lock:
            logs, self._pending_env_log = self._pending_env_log, []
        for msg in logs:
            console.print(msg)

    # ── Hallucination Filter ──────────────────────────────────────────────────

    def _is_hallucination(self, text: str) -> bool:
        return any(re.search(pat, text, re.IGNORECASE) for pat in self.hallucination_patterns)

    # ── Transcribe Thread ─────────────────────────────────────────────────────
    # Làm sạch, Dịch chữ và Phân tích cảm xúc.
    def transcribe_thread(self) -> None:
        while self.is_running:
            try:
                audio = self.segment_q.get(timeout=0.5)
            except queue.Empty:
                continue
            
            # Nếu đoạn âm thanh ngắn hơn 0.4 giây -> bỏ
            if len(audio) < int(self.config["sample_rate"] * 0.4):
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

            # lọc những câu rác
            if not text or self._is_hallucination(text):
                continue

            text = text[0].upper() + text[1:]
            self._prev_transcripts.append(text)

            console.print(
                f"\n[bold yellow][VI][/bold yellow] "
                f"[white]{text}[/white]  "
                f"[dim]({elapsed:.2f}s | {len(audio) / self.config['sample_rate']:.1f}s audio)[/dim]"
            )

            if self.config["save_to_file"]:
                with open(self.config["output_file"], "a", encoding="utf-8") as f:
                    f.write(f"[VI] {text}\n")

            # Nhận diện cảm xúc
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
                dominant_out = str(emotion.get("dominant", ""))

            tone = self.tone_model.analyze(audio, self.config["sample_rate"])

            if probs_str:
                console.print(f"[cyan]📝 Text Emotion:[/cyan] {dominant_out} [dim]({probs_str})[/dim]")
            else:
                console.print(f"[cyan]📝 Text Emotion:[/cyan] {dominant_out}")

            tone_probs_str = ", ".join(
                f"{k}: {v:.3f}" for k, v in tone.items()
                if k not in ("dominant", "human_readable", "confidence")
            )
            console.print(
                f"[magenta]🎙️ Audio Tone:[/magenta] {tone['human_readable']} "
                f"[dim]({tone_probs_str})[/dim]"
            )

            self._flush_env_log()

            if self.on_transcript:
                try:
                    self.on_transcript(text)
                except Exception as e:
                    console.print(f"[red]Reply error: {e}[/red]")

            self._flush_env_log()

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.load_models()
        self.is_running = True

        threading.Thread(target=self.vad_thread,        daemon=True).start()
        threading.Thread(target=self.transcribe_thread, daemon=True).start()

        console.print("\n[green]🎙️ Đang lắng nghe...[/green]\n")

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
            console.print("\n[red]⏹️ Đã dừng[/red]")
            self.is_running = False
            try:
                self._save_full_session()
            except Exception as e:
                console.print(f"[red]Full session save error: {e}[/red]")

    # ── Save WAV ──────────────────────────────────────────────────────────────

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
        console.print(f"[bold green]✅ Đã lưu toàn bộ phiên: {full_path}[/bold green]")


# ══════════════════════════════════════════════════════════════════════════════
# TTS — TEXT TO SPEECH
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_VOICE  = "vi-VN-HoaiMyNeural"
DEFAULT_RATE   = "+30%"
DEFAULT_VOLUME = "+0%"
DEFAULT_PITCH  = "+0Hz"


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
    
    # --- BẢN VÁ: DỊCH MP3 SANG CHUẨN WAV (PCM 16-bit) ---
    try:
        # Đọc MP3 từ RAM
        audio = AudioSegment.from_mp3(mp3_buf)
        
        # Ép chuẩn WAV 16-bit (Để Rhubarb và C# đọc mượt 100%)
        audio = audio.set_sample_width(2) 
        
        # Lưu ngược lại thành WAV trên RAM
        wav_buf = io.BytesIO()
        audio.export(wav_buf, format="wav")
        
        return wav_buf.getvalue()
    except Exception as e:
        print(f"❌ [LỖI TTS] Không thể dịch MP3 sang WAV: {e}")
        return mp3_buf.read() # Fallback trả về cục cũ nếu lỗi


def _play_audio(audio_bytes: bytes) -> None:
    buf  = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(buf, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()


# ── TextToSpeech ──────────────────────────────────────────────────────────────

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

        console.print(f"[green]✅ TTS sẵn sàng — giọng: [bold]{voice}[/bold][/green]")

    # ── STT Link ──────────────────────────────────────────────────────────────

    def set_stt(self, stt) -> None:
        self._stt = stt

    # ── Streaming API ─────────────────────────────────────────────────────────
    # streaming text token by token, mỗi khi có dấu câu hoặc xuống dòng thì sẽ gọi speak() để đọc đoạn đó lên, còn nếu chưa có dấu câu thì cứ tiếp tục lưu vào buffer
    def stream_token(self, token: str) -> None:
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
        if self.text_buffer.strip():
            self.speak(self.text_buffer)
        self.text_buffer = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        cleaned = self._clean_text(text)
        if cleaned and any(c.isalnum() for c in cleaned):
            self._q.put(cleaned.strip())

    # phát âm thanh và bắt chương trình phải đợi cho đến khi nói xong
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
        console.print("[red]⏹️ TTS đã dừng[/red]")

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        return re.sub(r'["""'']', "", text)

    def _synth_loop(self) -> None:
        while self._running:
            text = self._q.get()
            if text is None:
                self._audio_q.put(None)
                break
            if self._stt:
                self._stt.mute_mic() # khi SEN đang nói, tắt mic để tránh bị thu lại tiếng của chính nó
            try:
                # để chuyển chữ thành các đoạn mã âm thanh
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
                if self._audio_q.empty() and self._stt:
                    self._stt.unmute_mic()