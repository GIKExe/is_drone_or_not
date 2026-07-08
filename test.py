from collections import deque
import os
import shutil
from time import time

from ultralytics import YOLO
import sounddevice as sd
import numpy as np
from PIL import Image

from spectrogram import single_chunk_to_image, SAMPLE_RATE


def create_spectrogram(audio_array):
    """
    Раньше здесь была "адаптивная" нормализация вручную (по min/max всего
    буфера), а потом ЕЩЁ РАЗ через plt.imsave по обрезанному куску —
    двойная нормализация, которая не совпадала с тем, что видела модель
    на этапе обучения (main.py нормализовал каждый чанк один раз, по
    фиксированному диапазону).

    Теперь используется ТА ЖЕ функция chunk_to_image (через
    single_chunk_to_image), что и в main.py — с фиксированным
    VMIN_DB/VMAX_DB из spectrogram.py. Инференс и обучение видят
    статистически одинаковые картинки.
    """
    img_array = single_chunk_to_image(audio_array, SAMPLE_RATE)
    Image.fromarray(img_array).save('micro.png')
    return img_array


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
    except Exception:
        print('Неверный ввод, выбрана последняя модель: ', end='')
        i = index - 1
        print(paths[i])

    model = YOLO(paths[i])

    duration = 0.5
    frames_to_read = int(duration * SAMPLE_RATE)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
        while True:
            audio, overflowed = stream.read(frames_to_read)
            audio = audio.flatten()

            img_array = create_spectrogram(audio)

            results = model(img_array, save=False, verbose=False, rect=True)[0]
            q = float(results.probs.data[0])

            if q > 0.5:
                shutil.copy('micro.png', f'micro/micro_{int(time()*1000)}.png')

            median_list.append(q)
            aq = sum(median_list) / list_size
            mq = min(median_list)
            text = 'ДРОН' if (aq > 0.6) and (mq > 0.2) else ' '
            print(f'{text} Вероятность: {f"{q*100:.2f}":>6}% Среднее: {f"{aq*100:.2f}":>6}%')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
