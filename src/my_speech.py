import speech_recognition as sr  # Thư viện nhận diện giọng nói

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Chuyển đổi âm thanh từ dạng bytes thành văn bản sử dụng SpeechRecognition.
    """
    r = sr.Recognizer()
    try:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)  # Lưu file âm thanh tạm thời
        with sr.AudioFile("temp_audio.wav") as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data, language="vi-VN")  # Hỗ trợ tiếng Việt
            return text
    except Exception as e:
        return f"Lỗi nhận diện giọng nói: {str(e)}"
