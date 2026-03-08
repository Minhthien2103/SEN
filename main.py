from STT import SpeechToText
from generate_response import ResponseGenerator
from rich.console import Console

console = Console()
generator = ResponseGenerator()


def on_new_transcript(text: str) -> None:
    """Mỗi khi STT có câu mới: gọi Gemini trả lời và in ra ngay (realtime)."""
    reply = generator.reply(text)
    if reply:
        console.print(f"\n[bold green][SEN][/bold green] [white]{reply}[/white]\n")


# STT + callback: khi nói xong → lưu transcript → gọi Gemini → in reply ngay
stt = SpeechToText(on_transcript=on_new_transcript)
stt.run()
