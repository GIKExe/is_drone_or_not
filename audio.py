from time import time
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np


def to_spectrogram(path):
	sr = 16000
	audio_array, sr = librosa.load(path, sr=sr)

	mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_mels=128)
	mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

	# Нормализация
	mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
	mel_spec_normalized = (mel_spec_normalized * 255).astype(np.uint8)
	
	# Размеры
	n_mels, total_frames = mel_spec_normalized.shape  # n_mels = 128
	chunk_width = 16

	# Количество полных кусков (остаток отбрасываем)
	n_chunks = total_frames // chunk_width

	audio_name = int(time()*1000)
	for i in range(n_chunks):
		start = i * chunk_width
		end = start + chunk_width
		chunk = mel_spec_normalized[:, start:end]  # shape: (128, 16)
		plt.imsave(f'audio/audio_{audio_name}_chunk_{i:03d}.png', chunk, cmap='gray')


def main():
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
	except:
		raise