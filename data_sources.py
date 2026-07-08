"""
Источник аудио-данных для генерации датасета.

Сейчас используются только wav-файлы, уже разложенные ВАМИ по папкам:
    wav_data/train/drone/*.wav
    wav_data/train/no_drone/*.wav
    wav_data/val/drone/*.wav
    wav_data/val/no_drone/*.wav

Split (train/val) берётся напрямую из папки — никакого случайного
index % 5 разбиения тут не нужно, вы уже сами всё разложили.

Parquet с HuggingFace ВРЕМЕННО ОТКЛЮЧЁН (закомментирован ниже) — раз вы
им сейчас не пользуетесь. У него не было готового train/val деления по
папкам (это просто один блоб данных), так что если понадобится вернуть
parquet — там придётся снова городить искусственный сплит (например,
старый index % 5 == 0), а не просто читать из папки.

Ресемплирование: sf.read() отдаёт аудио на РОДНОЙ частоте дискретизации
файла — если она отличается от spectrogram.SAMPLE_RATE, принудительно
ресемплируем через librosa.resample, иначе частотный срез FREQ_LOW:FREQ_HIGH
будет означать разные реальные частоты для разных файлов.
"""

import os
import glob
import numpy as np
import librosa
import soundfile as sf

from spectrogram import SAMPLE_RATE

LABEL_MAP = {0: "no_drone", 1: "drone"}

WAV_ROOT = "wav_data"
SPLITS = ["train", "val"]
CLASSES = [("no_drone", 0), ("drone", 1)]


def _ensure_sample_rate(audio_data: np.ndarray, native_sr: int, target_sr: int = SAMPLE_RATE):
    """Приводит аудио к единой частоте дискретизации, если это ещё не так."""
    if native_sr == target_sr:
        return audio_data, target_sr
    resampled = librosa.resample(audio_data.astype(np.float32), orig_sr=native_sr, target_sr=target_sr)
    return resampled, target_sr


def iter_wav_audio():
    """Отдаёт (audio_array, sample_rate, label_id, unique_name, split_type).

    split_type ('train'/'val') берётся из структуры папок напрямую —
    файлы из wav_data/train/... всегда идут в train, из wav_data/val/...
    всегда в val, никакого перемешивания.
    """
    for split_type in SPLITS:
        for class_name, label_id in CLASSES:
            folder = os.path.join(WAV_ROOT, split_type, class_name)
            if not os.path.isdir(folder):
                print(f"[wav] папка {folder} не найдена, пропускаю")
                continue

            wav_paths = sorted(glob.glob(os.path.join(folder, "*.wav")))
            for path in wav_paths:
                try:
                    audio_data, native_sr = sf.read(path)
                    if len(audio_data.shape) > 1:
                        audio_data = np.mean(audio_data, axis=1)

                    audio_data, sample_rate = _ensure_sample_rate(audio_data, native_sr)

                    base_name = os.path.splitext(os.path.basename(path))[0]
                    unique_name = f"wav_{split_type}_{class_name}_{base_name}"
                    yield audio_data, sample_rate, label_id, unique_name, split_type
                except Exception as e:
                    print(f"[wav] пропуск {path}: {e}")
                    continue


def iter_all_audio():
    """Пока только wav (см. iter_wav_audio). Parquet временно отключен —
    см. закомментированный блок ниже, если понадобится вернуть."""
    yield from iter_wav_audio()

    # ── Parquet с HuggingFace (временно отключен) ────────────────────────────
    # import io
    # from datasets import load_dataset, Audio
    #
    # PARQUET_PATTERN = "data/*.parquet"
    #
    # def iter_parquet_audio():
    #     """Отдаёт (audio_array, sample_rate, label_id, unique_name, split_type).
    #     У parquet нет готового train/val деления по папкам — тут пришлось бы
    #     возвращаться к искусственному сплиту (например index % 5 == 0)."""
    #     dataset = load_dataset("parquet", data_files={"train": PARQUET_PATTERN})
    #     dataset = dataset.cast_column("audio", Audio(decode=False))
    #
    #     for index, item in enumerate(dataset["train"]):
    #         try:
    #             audio_bytes = item["audio"]["bytes"]
    #             with io.BytesIO(audio_bytes) as byte_io:
    #                 audio_data, native_sr = sf.read(byte_io)
    #             if len(audio_data.shape) > 1:
    #                 audio_data = np.mean(audio_data, axis=1)
    #
    #             audio_data, sample_rate = _ensure_sample_rate(audio_data, native_sr)
    #
    #             label_id = item["label"]
    #             split_type = "val" if index % 5 == 0 else "train"
    #             yield audio_data, sample_rate, label_id, f"parquet_{index:06d}", split_type
    #         except Exception as e:
    #             print(f"[parquet] пропуск индекса {index}: {e}")
    #             continue
    #
    # Чтобы вернуть parquet: раскомментируйте блок выше и в iter_all_audio()
    # добавьте "yield from iter_parquet_audio()".

