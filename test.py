from collections import deque
import os
import shutil
from time import time

from ultralytics import YOLO
import sounddevice as sd
import numpy as np
import librosa
import matplotlib.pyplot as plt


def create_spectrogram(audio_array, sample_rate):
	mel_spec = librosa.feature.melspectrogram(
		y=audio_array, sr=sample_rate, n_mels=128)
	mel_spec = librosa.power_to_db(mel_spec, ref=1.0)
	plt.imsave('micro0.png', np.flipud(mel_spec), cmap='gray')

	# === АДАПТИВНАЯ НОРМАЛИЗАЦИЯ (как в plt.imsave) ===
	vmin = mel_spec.min()
	vmax = mel_spec.max()
	
	# Избегаем деления на ноль
	if vmax - vmin < 1e-6:
		vmax = vmin + 1
	
	# Нормализуем к 0-255
	mel_spec = ((mel_spec - vmin) / (vmax - vmin) * 255).astype(np.uint8)
	mel_spec = np.flipud(mel_spec)

	plt.imsave('micro.png', mel_spec, cmap='gray')
	return np.stack((mel_spec,) * 3, axis=-1)


def main():
	os.makedirs('micro', exist_ok=True)

	list_size = 5
	median_list = deque(maxlen=list_size)
   
	index = 1
	paths = {}
	print('Выберите модель: ')
	for filename in os.listdir():
		if filename.startswith('v') and filename.endswith('.pt'):
			print(f'{index}) {filename}')
			paths[index] = filename
			index += 1
	try:
		i = int(input(' --> '))
		print('Выбрана модель:', paths[i])
	except:  # noqa: E722
		print('Неверный ввод, выбрана последняя модель: ', end='')
		i = index - 1
		print(paths[i])

	model = YOLO(paths[i])

	sample_rate = 16000
	duration = 0.5
	frames_to_read = int(duration * sample_rate)

	with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:    
		while True:
			audio, overflowed = stream.read(frames_to_read)
			audio = audio.flatten()

			# Спектрограмма
			img_array = create_spectrogram(audio, sample_rate)
			
			# Передаем numpy-массив напрямую в модель
			model.predict
			results = model(img_array, save=False, verbose=False, rect=True)[0]
			
			q = float(results.probs.data[0])

			if q > 0.1:
				shutil.copy('micro.png', f'micro/micro_{int(time()*1000)}.png')
			
			median_list.append(q)
			aq = sum(median_list) / list_size
			mq = min(median_list)
			text = 'ДРОН' if (aq > 0.6) and (mq > 0.2) else '    '
			print(f'{text}  Вероятность: {f"{q*100:.2f}":>6}%  Среднее: {f"{aq*100:.2f}":>6}%')


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		pass
	except:
		raise