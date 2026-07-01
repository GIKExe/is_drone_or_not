import sounddevice as sd
import numpy as np
import librosa
import matplotlib.pyplot as plt


def create_spectrogram(audio_array, sample_rate):
	mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sample_rate, n_mels=128)

	# Тихий шум останется тихим, а дрон будет громким.
	mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0)
		
	# ИСПРАВЛЕНИЕ 2: Глобальный диапазон. Всё, что тише -80 дБ, становится черным (0).
	mel_spec_db = np.clip(mel_spec_db, a_min=-80.0, a_max=0.0)
		
	# Переводим в 0-255
	mel_spec_normalized = ((mel_spec_db + 80.0) / 80.0 * 255).astype(np.uint8)

	# для отображения в реал тайме (сохранение картинки)
	mel_spec_normalized = np.flipud(mel_spec_normalized)
	plt.imsave('micro.png', mel_spec_normalized, cmap='gray')


def main():
	sample_rate = 16000
	duration = 5
	frames_to_read = int(duration * sample_rate)

	with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:    
		while True:
			audio, overflowed = stream.read(frames_to_read)
			audio = audio.flatten()

			create_spectrogram(audio, sample_rate)


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		pass
	except:
		raise