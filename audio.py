from time import time
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np


SAMPLE_RATE = 16_000


def to_spectrogram(path):
	audio_array, sr = librosa.load(path, sr=SAMPLE_RATE)

	mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=SAMPLE_RATE, n_mels=512)
	mel_spec = librosa.power_to_db(mel_spec, ref=1.0)
	mel_spec = np.flipud(mel_spec)

	# # Размеры
	# n_mels, total_frames = mel_spec.shape  # n_mels = 128
	# chunk_width = 16

	# # Количество полных кусков (остаток отбрасываем)
	# n_chunks = total_frames // chunk_width

	audio_name = int(time()*1000)
	# for i in range(n_chunks):
	# 	start = i * chunk_width
	# 	end = start + chunk_width
	# 	chunk = mel_spec[:, start:end]  # shape: (128, 16)
	# 	plt.imsave(f'audio/audio_{audio_name}_chunk_{i:03d}.png', chunk, cmap='gray')
	plt.imsave(f'audio/audio_{audio_name}.png', mel_spec[428:460, :], cmap='gray')


def main():
	os.makedirs('audio', exist_ok=True)
	while True:
		path = input('Путь до файла: ').strip('"')

		if not os.path.exists(path):
			print(f"Файл {path} не найден!")
			continue

		to_spectrogram(path)


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		pass