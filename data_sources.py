"""
Единая точка входа для аудио-данных: и parquet-датасет с HuggingFace,
и обычные .wav файлы, разложенные по папкам drone/no_drone.

И calibrate.py, и main.py используют iter_all_audio() — так что
калибровка (VMIN_DB/VMAX_DB) и генерация датасета всегда смотрят
на ОДИН и тот же комбинированный набор данных, без риска, что
что-то посчиталось по одному источнику, а применяется к другому.
"""

import os
import io
import glob
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio

LABEL_MAP = {0: "no_drone", 1: "drone"}

# ── Источник 1: parquet с HuggingFace ────────────────────────────────────
PARQUET_PATTERN = "data/*.parquet"

# ── Источник 2: обычные wav-файлы по папкам ──────────────────────────────
WAV_DRONE_DIR = "wav_data/drone"
WAV_NO_DRONE_DIR = "wav_data/no_drone"


def iter_parquet_audio():
    """Отдаёт (audio_array, sample_rate, label_id, unique_name) из parquet."""
    dataset = load_dataset("parquet", data_files={"train": PARQUET_PATTERN})
    dataset = dataset.cast_column("audio", Audio(decode=False))

    for index, item in enumerate(dataset["train"]):
        try:
            audio_bytes = item["audio"]["bytes"]  # type: ignore
            with io.BytesIO(audio_bytes) as byte_io:
                audio_data, sample_rate = sf.read(byte_io)
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            label_id = item["label"]  # type: ignore
            yield audio_data, sample_rate, label_id, f"parquet_{index:06d}"
        except Exception as e:
            print(f"[parquet] пропуск индекса {index}: {e}")
            continue


def iter_wav_audio():
    """Отдаёт (audio_array, sample_rate, label_id, unique_name) из wav-папок."""
    sources = [
        ("drone", 1, WAV_DRONE_DIR),
        ("no_drone", 0, WAV_NO_DRONE_DIR),
    ]

    for label_name, label_id, folder in sources:
        if not os.path.isdir(folder):
            print(f"[wav] папка {folder} не найдена, пропускаю")
            continue

        wav_paths = sorted(glob.glob(os.path.join(folder, "*.wav")))
        for path in wav_paths:
            try:
                audio_data, sample_rate = sf.read(path)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)

                base_name = os.path.splitext(os.path.basename(path))[0]
                yield audio_data, sample_rate, label_id, f"wav_{label_name}_{base_name}"
            except Exception as e:
                print(f"[wav] пропуск {path}: {e}")
                continue


def iter_all_audio():
    """Комбинированный источник: сначала parquet, потом wav-файлы."""
    yield from iter_parquet_audio()
    yield from iter_wav_audio()
