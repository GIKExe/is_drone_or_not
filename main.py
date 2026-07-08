import os
from PIL import Image

from spectrogram import compute_mel_db, extract_band, iter_chunks, chunk_to_image
from data_sources import iter_all_audio

# Корневая директория для готового датасета
base_dataset_dir = "out"

# Перенесли LABEL_MAP сюда, так как в data_sources.py он больше не используется,
# а тут он нужен для превращения label_id (0/1) в имя папки (no_drone/drone)
LABEL_MAP = {0: "no_drone", 1: "drone"}


def save_melspectrogram(audio_array, sample_rate, save_path):
    mel_db = compute_mel_db(audio_array, sample_rate)
    band = extract_band(mel_db)

    for i, chunk in iter_chunks(band):
        img_array = chunk_to_image(chunk)
        Image.fromarray(img_array).save(save_path + f'_chunk_{i:04d}.png')


# Обработка локальных wav-файлов.
for index, (audio_data, sample_rate, label_id, unique_name, split_type) in enumerate(iter_all_audio()):
    try:
        # Получаем имя класса из ID (1 -> 'drone', 0 -> 'no_drone')
        class_name = LABEL_MAP.get(label_id, str(label_id))
        # Сплит и класс уже готовы (например: split_type='train', class_name='drone')
        class_dir = os.path.join(base_dataset_dir, split_type, class_name)
        os.makedirs(class_dir, exist_ok=True)

        save_path = os.path.join(class_dir, unique_name)
        save_melspectrogram(audio_data, sample_rate, save_path)

    except Exception as e:
        print(f"Ошибка на {unique_name}: {e}. Пропускаем файл.")
        continue

    if index % 100 == 0:
        print(f"Обработано {index} файлов...")

print("Конвертация датасета завершена успешно!")

