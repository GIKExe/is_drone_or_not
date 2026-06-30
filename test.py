from ultralytics import YOLO
import sounddevice as sd
import numpy as np
import librosa
import matplotlib.pyplot as plt


def create_spectrogram(audio_array, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sample_rate, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Нормализация
    mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    mel_spec_normalized = (mel_spec_normalized * 255).astype(np.uint8)

    plt.imsave('micro.png', mel_spec_normalized, cmap='gray')

    # Конвертируем в 3-канальное изображение (RGB), 
    # так как plt.imsave с cmap='gray' сохраняет именно 3 канала (R=G=B)
    img_rgb = np.stack((mel_spec_normalized,) * 3, axis=-1)
    
    return img_rgb, mel_spec_normalized


def main():
    model = YOLO("best.pt")

    index = 0
    
    while True:
        # Запись и обработка
        duration=1
        sample_rate=16000
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        # Обработка
        # Удаление тишины в начале/конце
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Нормализация громкости
        # audio = librosa.util.normalize(audio) * 0.9
        
        # Можно добавить шумоподавление
        # audio = librosa.decompose.nn_filter(audio, aggregate=np.median)

        # Спектрограмма
        img_array, mel_spec_normalized = create_spectrogram(audio, sample_rate)
        
        # Передаем numpy-массив напрямую в модель
        results = model(img_array, save=False, verbose=False)[0]
        
        q = float(results.probs.data[0])
        # if q > 0.8:
        #     plt.imsave(f'micro/micro_{index:0>6}.png', mel_spec_normalized, cmap='gray')
        #     index += 1
        print(f'Вероятность что это дрон: {q*100:.2f}%')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nЗавершено пользователем.")
    except:
        raise