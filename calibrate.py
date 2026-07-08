"""
Считает реальный диапазон дБ (VMIN_DB / VMAX_DB) по ВСЕМУ комбинированному
датасету — и parquet с HuggingFace, и wav-файлам из wav_data/drone и
wav_data/no_drone — вместо того чтобы гадать значения руками.

Запустить один раз перед генерацией датасета:
    python calibrate.py

Скрипт напечатает рекомендованные VMIN_DB / VMAX_DB — их нужно вставить
в spectrogram.py вместо заглушки (-80.0 / 0.0).

ВАЖНО: сканируем оба источника ОТДЕЛЬНО (не через общий iter_all_audio),
а потом объединяем — иначе если parquet большой, а wav-файлов мало,
общий лимит MAX_FILES_TO_SCAN съест их почти целиком на parquet, и wav
просто не попадёт в калибровку.
"""

import numpy as np

from spectrogram import compute_mel_db, extract_band, iter_chunks
from data_sources import iter_wav_audio

MAX_FILES_PER_SOURCE = 300  # лимит на КАЖДЫЙ источник отдельно


def scan_source(name, generator, limit):
    values = []
    count = 0
    for audio_data, sample_rate, label_id, unique_name, split_type in generator:
        if count >= limit:
            break
        try:
            mel_db = compute_mel_db(audio_data, sample_rate)
            band = extract_band(mel_db)
            for _, chunk in iter_chunks(band):
                values.append(chunk.flatten())
        except Exception as e:
            print(f"[{name}] пропуск {unique_name}: {e}")
            continue

        count += 1
        if count % 50 == 0:
            print(f"[{name}] обработано {count} файлов...")

    print(f"[{name}] всего обработано: {count}")
    return values


def main():

    wav_values = scan_source("wav", iter_wav_audio(), MAX_FILES_PER_SOURCE)

    all_chunks = wav_values
    if not all_chunks:
        print("Не найдено ни одного файла ни в одном источнике — проверьте пути.")
        return

    all_values = np.concatenate(all_chunks)

    vmin = float(np.percentile(all_values, 1))
    vmax = float(np.percentile(all_values, 99))

    print("\n=== Результат (по объединённому датасету) ===")
    print(f"Реальный диапазон дБ: [{all_values.min():.2f}, {all_values.max():.2f}]")
    print(f"Рекомендуемые (1-99 перцентиль) VMIN_DB = {vmin:.2f}, VMAX_DB = {vmax:.2f}")
    print("\nВставьте эти значения в spectrogram.py:")
    print(f"VMIN_DB = {vmin:.2f}")
    print(f"VMAX_DB = {vmax:.2f}")


if __name__ == "__main__":
    main()
