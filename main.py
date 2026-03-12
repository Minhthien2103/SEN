from STT import SpeechToText
from generate_response import ResponseGenerator
from TTS import TextToSpeech
from rich.console import Console

console = Console()
generator = ResponseGenerator()
tts = TextToSpeech()


def on_new_transcript(text: str) -> None:
    """Mỗi khi STT có câu mới: gọi Gemini → in reply → đọc to qua TTS."""
    reply = generator.reply(text)
    if reply:
        console.print(f"\n[bold green][SEN][/bold green] [white]{reply}[/white]\n")
        tts.speak(reply)


# STT + callback: khi nói xong → lưu transcript → gọi Gemini → TTS đọc
stt = SpeechToText(on_transcript=on_new_transcript)
try:
    stt.run()
finally:
    tts.stop()
