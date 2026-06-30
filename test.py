from ultralytics import YOLO
import sounddevice as sd
import numpy as np
import librosa
import matplotlib.pyplot as plt


def record_audio(duration=3, sample_rate=16000):
    """Запись аудио с микрофона"""
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten(), sample_rate


def preprocess_audio(audio, sample_rate):
    # Удаление тишины в начале/конце
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Нормализация громкости
    # audio = librosa.util.normalize(audio) * 0.9
    
    # Можно добавить шумоподавление
    # audio = librosa.decompose.nn_filter(audio, aggregate=np.median)
    
    return audio


def create_spectrogram(audio_array, sample_rate):
    """Создание чб спектрограммы 128xN в памяти (без сохранения на диск)"""
    mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sample_rate, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Нормализация
    mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    mel_spec_normalized = (mel_spec_normalized * 255).astype(np.uint8)

    plt.imsave('micro.png', mel_spec_normalized, cmap='gray')

    # Конвертируем в 3-канальное изображение (RGB), 
    # так как plt.imsave с cmap='gray' сохраняет именно 3 канала (R=G=B)
    img_rgb = np.stack((mel_spec_normalized,) * 3, axis=-1)
    
    return img_rgb


def main():
    model = YOLO("best.pt")
    
    while True:
        # Ввод количества секунд
        # user_input = input("\nВведите кол-во секунд записи → ").strip()
        
        # try:
        #     duration = int(user_input) if user_input else 3
        # except ValueError:
        #     print("Некорректный ввод, используется 3 секунд")
        #     duration = 3

        # if duration < 1:
        #     duration = 1
        #     print("Минимум 1 секунда записи")
        # if duration > 10:
        #     duration = 10
        #     print('Максимум 10 секунд записи')

        # Запись и обработка
        audio, sr = record_audio(duration=1)
        audio = preprocess_audio(audio, sr)
        img_array = create_spectrogram(audio, sr)
        
        # Передаем numpy-массив напрямую в модель
        results = model(img_array, save=False, verbose=False)[0]
        
        q = float(results.probs.data[0])
        # nq = float(results.probs.data[1])
        print(f'Вероятность что это дрон: {q*100:.2f}%')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nЗавершено пользователем.")
    except:
        raise