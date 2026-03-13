from STT import SpeechToText
from generate_response import ResponseGenerator
from TTS import TextToSpeech
from rich.console import Console

console = Console()
generator = ResponseGenerator()
tts = TextToSpeech()


import sys

def on_new_transcript(text: str) -> None:
    """Mỗi khi STT có câu mới: gọi Gemini streaming → in chữ nào ra chữ đó → đọc to qua TTS từng câu."""
    console.print(f"\n[bold green][SEN][/bold green] ", end="")
    
    current_sentence = ""
    for token in generator.reply_stream(text):
        # In trực tiếp token ra màn hình ngay lập tức mà không phá tag rich
        sys.stdout.write(token)
        sys.stdout.flush()
        
        current_sentence += token
        
        # Ngắt câu nếu kết thúc bằng dấu câu cộng với khoảng trắng hoặc xuống dòng
        if any(current_sentence.endswith(p) for p in ['. ', '! ', '? ', '.\n', '!\n', '?\n', '\n']):
            sentence_to_speak = current_sentence.strip()
            if sentence_to_speak:
                tts.speak(sentence_to_speak)
            current_sentence = ""
            
    # Đọc phần còn lại nếu có
    if current_sentence.strip():
        tts.speak(current_sentence.strip())
    
    print() # Xuống dòng khi kết thúc output của SEN


# STT + callback: khi nói xong → lưu transcript → gọi Gemini → TTS đọc
stt = SpeechToText(on_transcript=on_new_transcript, tts=tts)
try:
    stt.run()
finally:
    tts.stop()
