"""
Общий модуль генерации мел-спектрограмм для проекта is_drone_or_not.

ГЛАВНАЯ ИДЕЯ ИСПРАВЛЕНИЯ:
Раньше main.py (подготовка датасета) и test.py (инференс с микрофона)
нормализовали спектрограмму КАЖДЫЙ ПО-СВОЕМУ:
  - main.py нормализовал каждый чанк 32x16 по его собственному min/max
    (через автоматическую нормализацию plt.imsave).
  - test.py нормализовал весь массив 512xN по его min/max, потом обрезал
    строки, а потом ЕЩЁ РАЗ прогонял обрезанный кусок через plt.imsave,
    который снова считал min/max — уже другие.

В итоге сеть обучалась на одной статистике яркости, а в бою видела
совсем другую — отсюда куча ложных срабатываний.

Исправление: ОДИН фиксированный диапазон (VMIN_DB, VMAX_DB), вычисленный
один раз по датасету (см. calibrate.py) и захардкоженный здесь. Он
используется абсолютно одинаково и при генерации датасета, и в проде.
Никакой "адаптивной" нормализации по месту быть не должно.
"""

import numpy as np
import librosa
from PIL import Image

# ── Параметры аудио ──────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
N_MELS = 512

# Диапазон mel-бинов после flipud, в котором сосредоточен сигнал дрона.
# Массив идёт после flipud, поэтому направления инвертированы относительно
# исходного mel-спектра: увеличение FREQ_HIGH расширяет диапазон вниз по
# частоте (к низким частотам), уменьшение FREQ_LOW расширяет вверх по частоте (к высоким).
FREQ_LOW = 204
FREQ_HIGH = 460

# ── Параметры нарезки ────────────────────────────────────────────────────
CHUNK_WIDTH = 16     # ширина чанка в фреймах (совпадает с ~0.5с при hop=512)
CHUNK_STEP = 12      # шаг окна при нарезке датасета (main.py).
                     # Было 4 при CHUNK_WIDTH=16 — 75% перекрытие, раздувало
                     # датасет до 994к картинок. 12 = 25% перекрытие: почти
                     # втрое меньше данных (и время на эпоху), но чанки всё
                     # ещё немного перекрываются, так что сигнал дрона на
                     # границе между чанками не режется пополам целиком,
                     # как могло бы быть при полностью безперекрывочном step=16.

# ── Фиксированный диапазон дБ для нормализации (ЗАПОЛНИТЬ после calibrate.py) ──
# Это заглушка! Обязательно замените на реальные значения, посчитанные
# calibrate.py по вашему датасету — иначе диапазон снова будет "придуман",
# а не проверен.
VMIN_DB = -80.0
VMAX_DB = 0.0

# Финальный размер изображения на входе в сеть (квадрат, под YOLO-cls).
# ВАЖНО: было 256 при FREQ-высоте 256 и CHUNK_WIDTH=16 — это означало
# 16-кратное растяжение по ширине (16 честных временных фреймов
# растягивались в 256 пикселей путём интерполяции, то есть почти вся
# "детализация" по времени была фейковой). 64 — компромисс: высота
# сжимается в 4 раза (256->64), ширина растягивается в 4 раза (16->64) —
# искажение симметричное и заметно мягче, чем было.
IMG_SIZE = 64


def compute_mel_db(audio_array: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Аудио -> мел-спектрограмма в дБ (абсолютная шкала, ref=1.0),
    перевёрнутая по частоте (низкие частоты внизу после flipud)."""
    mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel_spec, ref=1.0)
    mel_db = np.flipud(mel_db)
    return mel_db


def extract_band(mel_db: np.ndarray) -> np.ndarray:
    """Обрезка до нужного диапазона mel-бинов."""
    return mel_db[FREQ_LOW:FREQ_HIGH, :]


def chunk_to_image(chunk_db: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Превращает кусок дБ-спектрограммы (n_mels x chunk_width) в готовое
    изображение uint8 (img_size x img_size x 3), используя ФИКСИРОВАННЫЙ
    диапазон VMIN_DB..VMAX_DB — одинаковый для датасета и для инференса.

    Никакой нормализации "по месту" (по min/max самого чанка) — это и был
    главный баг.
    """
    clipped = np.clip(chunk_db, VMIN_DB, VMAX_DB)
    normalized = (clipped - VMIN_DB) / (VMAX_DB - VMIN_DB)  # 0..1
    gray = (normalized * 255).astype(np.uint8)

    # Приводим прямоугольный чанк к квадрату img_size x img_size ОДНИМ и тем
    # же способом что при генерации датасета, что при инференсе, чтобы
    # train.py (imgsz=IMG_SIZE) и реальный вход модели точно совпадали.
    img = Image.fromarray(gray, mode="L").resize((img_size, img_size), Image.BILINEAR)
    rgb = np.stack([np.array(img)] * 3, axis=-1)
    return rgb


def iter_chunks(mel_band: np.ndarray, width: int = CHUNK_WIDTH, step: int = CHUNK_STEP):
    """Скользящее окно по временной оси. Возвращает (index, chunk)."""
    n_mels, total_frames = mel_band.shape
    i = 0
    while True:
        start = i * step
        end = start + width
        if end > total_frames:
            break
        yield i, mel_band[:, start:end]
        i += 1


def audio_to_images(audio_array: np.ndarray, sr: int = SAMPLE_RATE,
                     width: int = CHUNK_WIDTH, step: int = CHUNK_STEP):
    """Полный путь: аудио -> список готовых RGB-изображений (uint8)."""
    mel_db = compute_mel_db(audio_array, sr)
    band = extract_band(mel_db)
    images = []
    for _, chunk in iter_chunks(band, width, step):
        images.append(chunk_to_image(chunk))
    return images


def single_chunk_to_image(audio_array: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Для живого инференса с микрофона: один буфер аудио (например 0.5с) ->
    одно изображение, БЕЗ повторной адаптивной нормализации.
    Если буфер даёт больше/меньше CHUNK_WIDTH фреймов, берём первый чанк
    нужной ширины (либо паддим, если фреймов меньше width).
    """
    mel_db = compute_mel_db(audio_array, sr)
    band = extract_band(mel_db)
    n_mels, total_frames = band.shape

    if total_frames >= CHUNK_WIDTH:
        chunk = band[:, :CHUNK_WIDTH]
    else:
        pad_width = CHUNK_WIDTH - total_frames
        chunk = np.pad(band, ((0, 0), (0, pad_width)), mode="edge")

    return chunk_to_image(chunk)
