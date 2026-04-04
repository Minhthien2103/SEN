import sys
from backend.modules.audio.speech import SpeechToText, TextToSpeech
from backend.core.brain import ResponseGenerator
from rich.console import Console

console   = Console()
generator = ResponseGenerator()
tts       = TextToSpeech()


def on_new_transcript(text: str) -> None:
    console.print("[bold yellow][SEN][/bold yellow] ", end="")

    for token in generator.reply_stream(text):
        sys.stdout.write(token)
        sys.stdout.flush()
        tts.stream_token(token)

    tts.flush_stream()
    print()


stt = SpeechToText(on_transcript=on_new_transcript, tts=tts)
tts.set_stt(stt)

try:
    stt.run()
finally:
    tts.stop()