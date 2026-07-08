import os
from PIL import Image

from spectrogram import compute_mel_db, extract_band, iter_chunks, chunk_to_image
from data_sources import iter_all_audio, LABEL_MAP

# Корневая директория для готового датасета
base_dataset_dir = "out"


def save_melspectrogram(audio_array, sample_rate, save_path):
    mel_db = compute_mel_db(audio_array, sample_rate)
    band = extract_band(mel_db)

    for i, chunk in iter_chunks(band):
        # Фиксированная нормализация (VMIN_DB/VMAX_DB из spectrogram.py) —
        # одна и та же что для parquet, что для wav-файлов.
        img_array = chunk_to_image(chunk)
        Image.fromarray(img_array).save(save_path + f'_chunk_{i:04d}.png')


# Обработка данных из ОБОИХ источников: parquet с HuggingFace + wav-файлы
# из wav_data/drone и wav_data/no_drone (см. data_sources.py)
for index, (audio_data, sample_rate, label_id, unique_name) in enumerate(iter_all_audio()):
    try:
        class_name = LABEL_MAP.get(label_id, str(label_id))

        # Разделение выборки: 20% в val, 80% в train.
        # index сквозной по ОБОИМ источникам — значит и wav-файлы, и
        # parquet-сэмплы равномерно распределяются между train/val,
        # а не оседают только в одной из папок.
        split_type = "val" if index % 5 == 0 else "train"

        class_dir = os.path.join(base_dataset_dir, split_type, class_name)
        os.makedirs(class_dir, exist_ok=True)

        save_path = os.path.join(class_dir, unique_name)
        save_melspectrogram(audio_data, sample_rate, save_path)

    except Exception as e:
        print(f"Ошибка на {unique_name}: {e}. Пропускаем файл.")
        continue

    if index % 100 == 0:
        print(f"Обработано {index} файлов...")

print("Распаковка, конвертация и разделение датасета (parquet + wav) завершены успешно!")

