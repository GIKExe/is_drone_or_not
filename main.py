import os
import io
import soundfile as sf
import numpy as np
import librosa
import matplotlib.pyplot as plt
from datasets import load_dataset, Audio  # Импортируем класс Audio напрямую

# Словарь для преобразования числовых меток в названия папок
LABEL_MAP = {
    0: "no_drone",
    1: "drone"
}

# 1. Загрузка датасета
dataset = load_dataset("parquet", data_files={'train': 'data/*.parquet'})

# Правильный способ отключить автодекодирование аудио
dataset = dataset.cast_column("audio", Audio(decode=False))

# 2. Корневая директория для датасета
base_dataset_dir = "out"


# 3. Функция генерации спектрограммы
def save_melspectrogram(audio_array, sample_rate, save_path):
    mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sample_rate, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Нормализация
    mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    mel_spec_normalized = (mel_spec_normalized * 255).astype(np.uint8)

    # Размеры
    n_mels, total_frames = mel_spec_normalized.shape  # n_mels = 128
    chunk_width = 16

    # Количество полных кусков (остаток отбрасываем)
    n_chunks = total_frames // chunk_width

    for i in range(n_chunks):
        # Вырезаем кусок 128x16
        start = i * chunk_width
        end = start + chunk_width
        chunk = mel_spec_normalized[:, start:end]  # shape: (128, 16)
        plt.imsave(save_path+f'_chunk_{i:03d}.png', chunk, cmap='gray')


# 4. Обработка данных
for index, item in enumerate(dataset['train']):
    try:
        # Извлекаем сырые байты аудиофайла
        audio_bytes = item['audio']['bytes'] # type: ignore

        # Декодируем аудио в массив numpy
        with io.BytesIO(audio_bytes) as byte_io:
            audio_data, sample_rate = sf.read(byte_io)

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        label_id = item['label'] # type: ignore
        class_name = LABEL_MAP.get(label_id, str(label_id))

        # Разделение выборки: 20% в val, 80% в train
        split_type = "val" if index % 5 == 0 else "train"

        # Формируем путь
        class_dir = os.path.join(base_dataset_dir, split_type, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        file_name = f"sample_{index:06d}"
        save_path = os.path.join(class_dir, file_name)

        save_melspectrogram(audio_data, sample_rate, save_path)

    except Exception as e:
        print(f"Ошибка на индексе {index}: {e}. Пропускаем файл.")
        continue

    # Вывод прогресса
    if index % 100 == 0:
        print(f"Обработано {index} файлов...")

print("Распаковка, конвертация и разделение датасета завершены успешно!")