import sys
from STT import SpeechToText
from generate_response import ResponseGenerator
from TTS import TextToSpeech
from rich.console import Console

console = Console()
generator = ResponseGenerator()
tts = TextToSpeech()


def on_new_transcript(text: str) -> None:
    # In prefix [SEN] màu vàng giống [VI]
    console.print("[bold yellow][SEN][/bold yellow] ", end="")
    
    # Bắt đầu nhận luồng stream từ AI
    for token in generator.reply_stream(text):
        # 1. In ra terminal ngay lập tức để mắt nhìn thấy
        sys.stdout.write(token)
        sys.stdout.flush()
        
        # 2. Đẩy thẳng token vào bộ đệm của TTS
        # Hàm này sẽ tự động lo việc kiểm tra dấu câu (phẩy, chấm, hỏi...) 
        # và gọi tts.speak() ngay khi cắt được 1 cụm có nghĩa.
        tts.stream_token(token)
        
    # 3. Khi AI đã sinh xong toàn bộ text, gọi flush để phát nốt 
    # những chữ cái cuối cùng còn sót lại trong bộ đệm (không có dấu câu kết thúc)
    tts.flush_stream()
    
    print() # Xuống dòng khi kết thúc output của SEN


# STT + callback: khi nói xong → lưu transcript → gọi Gemini → TTS đọc
stt = SpeechToText(on_transcript=on_new_transcript, tts=tts)
try:
    stt.run()
finally:
    tts.stop()