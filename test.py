from ultralytics import YOLO
import sounddevice as sd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from time import time
import os


def create_spectrogram(audio_array, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sample_rate, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Нормализация
    mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    mel_spec_normalized = (mel_spec_normalized * 255).astype(np.uint8)

    # для отображения в реал тайме
    plt.imsave('micro.png', mel_spec_normalized, cmap='gray')

    # Конвертируем в 3-канальное изображение (RGB)
    img_rgb = np.stack((mel_spec_normalized,) * 3, axis=-1)

    return img_rgb, mel_spec_normalized


def main():
    debug = input("Включить дебаг? [да/НЕТ] > ").lower() in ['да', 'д', 'y', 'yes']
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
        print('Неверный ввод, выбрана последняя')
        i = index - 1

    model = YOLO(paths[i])
    
    sample_rate = 16000
    duration = 1
    frames_to_read = int(duration * sample_rate)

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:    
        while True:
            # Функция read() блокирует выполнение
            audio, overflowed = stream.read(frames_to_read)
            audio = audio.flatten()

            # Спектрограмма
            img_array, mel_spec_normalized = create_spectrogram(audio, sample_rate)
            
            # Передаем numpy-массив напрямую в модель
            results = model(img_array, save=False, verbose=False)[0]
            
            q = float(results.probs.data[0])
            
            # сохранение записей с высокой вероятностью
            if (q > 0.8) and debug:
                plt.imsave(f'micro/micro_{int(time()*1000)}.png', mel_spec_normalized, cmap='gray')
                
            print(f'Вероятность что это дрон: {q*100:.2f}%')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nЗавершено пользователем.")
    except:
        raise