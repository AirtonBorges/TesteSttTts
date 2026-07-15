from faster_whisper import WhisperModel
import sounddevice as sounddevice
import numpy as np
from scipy.io.wavfile import write
import wave
from piper import PiperVoice


def main():

    frequecy = 16000
    number_channels = 1

    duration = 4.0
    print("gravando...")
    myrecording = sounddevice.rec(int(duration * frequecy), samplerate=frequecy, channels=number_channels)
    sounddevice.wait()
    print("finalizado gravação...")

    int16_array = (myrecording * 32767).astype(np.int16)

    write("recorded_audio.wav", frequecy, int16_array)
    print("Áudio salvo em recorded_audio.wav")
    print("Saved to recorded_audio.mp3")

    # Inicaliza o modelo
    model_size = "small"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # Ajusta array do audio
    myrecording = myrecording.squeeze()
    myrecording = np.ascontiguousarray(myrecording, dtype=np.float32)
    
    # passa array do audio para o modelo
    segments, info = model.transcribe(myrecording, beam_size=5)
    
    # Mostra precisao
    print(
        "Linguagem detectada '%s' com probabilidade %f"
        % (info.language, info.language_probability)
    )

    texto = ""
    # Mostra o que voltou do modelo
    for segment in segments:
        texto += ("%s" % (segment.text))

    print("[%.2fs -> %.2fs]" % (segment.start, segment.end), texto)

    voice = PiperVoice.load("pt_BR-faber-medium.onnx")
    with wave.open("piper.wav", "wb") as wav_file:
        voice.synthesize_wav(texto, wav_file)

if __name__ == "__main__":
    main()
