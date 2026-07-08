from time import time
import os
import librosa
from PIL import Image

from spectrogram import audio_to_images, SAMPLE_RATE


def to_spectrogram(path):
    audio_array, sr = librosa.load(path, sr=SAMPLE_RATE)

    # Та же нарезка на чанки (CHUNK_WIDTH/CHUNK_STEP) и та же фиксированная
    # нормализация (chunk_to_image), что и в main.py — раньше audio.py
    # сохранял ОДНУ картинку на весь файл без нарезки и без нормализации
    # датасета, из-за чего эти файлы не были сопоставимы с тем, что видит
    # модель на обучении.
    images = audio_to_images(audio_array, sr)

    audio_name = int(time() * 1000)
    for i, img_array in enumerate(images):
        Image.fromarray(img_array).save(f'audio/audio_{audio_name}_chunk_{i:04d}.png')

    print(f"Сохранено {len(images)} чанков для {path}")


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
