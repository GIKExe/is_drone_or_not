import os
import re
import soundfile as sf
import numpy as np
import librosa

# --- НАСТРОЙКИ РЕЖИМА РАБОТЫ ---
# mode = 1: обработка папки с аудиофайлами
# mode = 0: обработка одного аудиофайла
mode = 0

# !!! СЮДА ПУТЬ, ГДЕ БУДУТ СОЗДАВАТЬСЯ НАРЕЗАННЫЕ ФАЙЛЫ !!!
output_base_path = r"E:\Praktika\!_New_Dataset\Done\Vavavavavav\no_drone_cut" 
os.makedirs(output_base_path, exist_ok=True)

# ⭐ ЦЕЛЕВОЙ ФОРМАТ (все файлы будут приведены к этому)
TARGET_SAMPLE_RATE = 16000   # 16 kHz
TARGET_SUBTYPE = 'PCM_16'    # 16-bit PCM
TARGET_CHANNELS = 1       # None = сохранять оригинальное кол-во каналов, 1 = моно, 2 = стерео

# ⭐ ДЛИТЕЛЬНОСТЬ ФРАГМЕНТА В СЕКУНДАХ
FRAGMENT_DURATION = 0.5

# ⭐ ПЕРЕКРЫТИЕ ФРАГМЕНТОВ (от 0.0 до 0.99)
OVERLAP = 0.5

# ⭐ ЧТО ДЕЛАТЬ С ПОСЛЕДНИМ ФРАГМЕНТОМ, ЕСЛИ ОН КОРОЧЕ?
DROP_SHORT_TAIL = True

# --- ФУНКЦИЯ ЧТЕНИЯ И НОРМАЛИЗАЦИИ АУДИО ---

def load_and_normalize(file_path):
    """
    Читает аудиофайл любого формата и приводит к целевому формату:
    WAV, 16kHz, PCM_16
    
    Args:
        file_path: путь к файлу
    
    Returns:
        tuple: (audio_data, sample_rate, original_info_dict)
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    # Получаем информацию об оригинале (если возможно)
    original_info = {
        'sample_rate': None,
        'channels': None,
        'subtype': None,
        'format': ext
    }
    
    try:
        # Для WAV пробуем получить точную информацию через soundfile
        if ext == '.wav':
            info = sf.info(file_path)
            original_info['sample_rate'] = info.samplerate
            original_info['channels'] = info.channels
            original_info['subtype'] = info.subtype
    except Exception:
        pass
    
    # Читаем через librosa (поддерживает WAV, MP3, FLAC, OGG и т.д.)
    # sr=None — сохраняем оригинальную частоту, потом ресэмплим вручную
    # mono=False — сохраняем оригинальное количество каналов
    audio_data, original_sr = librosa.load(file_path, sr=None, mono=False)
    
    # librosa возвращает shape (channels, samples) для многоканального аудио
    # или (samples,) для моно
    if len(audio_data.shape) > 1:
        original_info['channels'] = audio_data.shape[0]
        # Транспонируем в (samples, channels) для удобства
        audio_data = audio_data.T
    else:
        original_info['channels'] = 1
    
    if original_info['sample_rate'] is None:
        original_info['sample_rate'] = original_sr
    
    # ⭐ РЕСЭМПЛИНГ до целевой частоты (если нужно)
    if original_sr != TARGET_SAMPLE_RATE:
        if len(audio_data.shape) > 1:
            # Многоканальное: ресэмплим каждый канал отдельно
            resampled_channels = []
            for ch in range(audio_data.shape[1]):
                resampled_ch = librosa.resample(
                    audio_data[:, ch], 
                    orig_sr=original_sr, 
                    target_sr=TARGET_SAMPLE_RATE
                )
                resampled_channels.append(resampled_ch)
            audio_data = np.column_stack(resampled_channels)
        else:
            # Моно
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=original_sr, 
                target_sr=TARGET_SAMPLE_RATE
            )
    
    # ⭐ КОНВЕРТАЦИЯ КАНАЛОВ (если задано TARGET_CHANNELS)
    if TARGET_CHANNELS is not None:
        if TARGET_CHANNELS == 1 and len(audio_data.shape) > 1:
            # Стерео → моно (усредняем каналы)
            audio_data = np.mean(audio_data, axis=1)
        elif TARGET_CHANNELS == 2 and len(audio_data.shape) == 1:
            # Моно → стерео (дублируем канал)
            audio_data = np.column_stack([audio_data, audio_data])
    
    # ⭐ НОРМАЛИЗАЦИЯ в диапазон [-1, 1] для PCM_16
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        audio_data = audio_data / max_val
    
    return audio_data, TARGET_SAMPLE_RATE, original_info

# --- ФУНКЦИЯ НАРЕЗКИ АУДИО ---

def slice_audio(audio_data, sample_rate, fragment_duration=0.5, overlap=0.5, drop_short_tail=True):
    """
    Нарезает аудио на фрагменты с перекрытием
    """
    fragment_length = int(fragment_duration * sample_rate)
    stride = int(fragment_length * (1 - overlap))
    
    if stride < 1:
        stride = 1
    
    total_length = len(audio_data)
    
    fragments = []
    start = 0
    
    while start + fragment_length <= total_length:
        fragments.append(audio_data[start:start + fragment_length])
        start += stride
    
    tail_length = total_length - start
    if tail_length > 0 and not drop_short_tail:
        fragments.append(audio_data[start:])
    
    return fragments

# --- ОСНОВНОЙ ЦИКЛ ПРОГРАММЫ ---

# Расширения, которые поддерживаем
SUPPORTED_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma')

try:
    while True:
        # Проверка корректности OVERLAP
        if not (0.0 <= OVERLAP < 1.0):
            print(f"\n✗ Неверное значение OVERLAP: {OVERLAP}. Должно быть от 0.0 до 0.99")
            input("\nНажмите Enter для выхода...")
            exit()

        if mode == 1:
            # === РЕЖИМ 1: ОБРАБОТКА ПАПКИ ===
            folder_path = input("Введите путь к папке с аудиофайлами: ").strip().strip('"').strip("'")

            if not os.path.isdir(folder_path):
                print(f"\n✗ Папка не найдена: {folder_path}")
                input("\nНажмите Enter для выхода...")
                exit()

            # Находим все поддерживаемые аудиофайлы
            audio_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(SUPPORTED_EXTENSIONS)]
            print(f"\n📁 Найдено {len(audio_files)} аудиофайлов")

            if len(audio_files) == 0:
                print("Нет файлов для обработки.")
                input("\nНажмите Enter для выхода...")
                exit()

            # Сортируем файлы по числу в имени
            def extract_number(filename):
                match = re.search(r'(\d+)', filename)
                if match:
                    return int(match.group(1))
                return 0

            audio_files.sort(key=extract_number)

            print(f"\n📂 Нарезанные файлы будут сохранены в: {output_base_path}")
            print(f"🎯 Целевой формат: WAV, {TARGET_SAMPLE_RATE} Hz, {TARGET_SUBTYPE}")
            print(f"⏱️  Длительность фрагмента: {FRAGMENT_DURATION} сек")
            print(f"🔀 Перекрытие (overlap): {OVERLAP*100:.0f}%")
            print(f"🔪 Короткий хвост: {'отбрасывается' if DROP_SHORT_TAIL else 'сохраняется'}\n")

            total_fragments = 0

            # Обрабатываем каждый файл
            for file_idx, filename in enumerate(audio_files, 1):
                file_path = os.path.join(folder_path, filename)
                
                try:
                    # ⭐ ЧИТАЕМ И ПРИВОДИМ К ЦЕЛЕВОМУ ФОРМАТУ
                    audio_data, sample_rate, original_info = load_and_normalize(file_path)
                    
                    base_name = os.path.splitext(filename)[0]
                    original_duration = len(audio_data) / sample_rate
                    
                    # Показываем, была ли конвертация
                    conversion_info = []
                    if original_info['sample_rate'] != TARGET_SAMPLE_RATE:
                        conversion_info.append(f"{original_info['sample_rate']}→{TARGET_SAMPLE_RATE}Hz")
                    if original_info['subtype'] and original_info['subtype'] != TARGET_SUBTYPE:
                        conversion_info.append(f"{original_info['subtype']}→{TARGET_SUBTYPE}")
                    
                    conversion_str = f" [конвертация: {', '.join(conversion_info)}]" if conversion_info else " [формат OK]"
                    
                    # Нарезаем аудио
                    fragments = slice_audio(
                        audio_data, 
                        sample_rate, 
                        fragment_duration=FRAGMENT_DURATION,
                        overlap=OVERLAP,
                        drop_short_tail=DROP_SHORT_TAIL
                    )
                    
                    print(f"🔹 [{file_idx}/{len(audio_files)}] {filename} ({original_duration:.2f} сек){conversion_str} → {len(fragments)} фрагментов")

                    # Сохраняем каждый фрагмент
                    for frag_idx, fragment in enumerate(fragments):
                        output_filename = f"{base_name}_{frag_idx}.wav"
                        output_path = os.path.join(output_base_path, output_filename)
                        
                        # ⭐ СОХРАНЯЕМ ВСЕГДА В ЦЕЛЕВОМ ФОРМАТЕ!
                        sf.write(output_path, fragment, TARGET_SAMPLE_RATE, subtype=TARGET_SUBTYPE)
                        total_fragments += 1
                    
                    print(f"  ✓ Создано файлов: {len(fragments)} (с _0 по _{len(fragments)-1})")
                
                except Exception as e:
                    print(f"  ✗ Ошибка при обработке {filename}: {e}")

            print("\n" + "="*60)
            print(f"✅ Готово! Из {len(audio_files)} файлов создано {total_fragments} фрагментов")
            print(f"📂 Результат в папке: {output_base_path}")
            print("="*60)

        elif mode == 0:
            # === РЕЖИМ 0: ОБРАБОТКА ОДНОГО ФАЙЛА ===
            file_path = input("Введите путь к аудиофайлу: ").strip().strip('"').strip("'")

            if not os.path.isfile(file_path):
                print(f"\n✗ Файл не найден: {file_path}")
                input("\nНажмите Enter для выхода...")
                exit()

            if not file_path.lower().endswith(SUPPORTED_EXTENSIONS):
                print(f"\n✗ Неподдерживаемый формат файла: {file_path}")
                input("\nНажмите Enter для выхода...")
                exit()

            try:
                # ⭐ ЧИТАЕМ И ПРИВОДИМ К ЦЕЛЕВОМУ ФОРМАТУ
                audio_data, sample_rate, original_info = load_and_normalize(file_path)
                
                filename = os.path.basename(file_path)
                base_name = os.path.splitext(filename)[0]
                original_duration = len(audio_data) / sample_rate
                
                # Показываем, была ли конвертация
                conversion_info = []
                if original_info['sample_rate'] != TARGET_SAMPLE_RATE:
                    conversion_info.append(f"SR: {original_info['sample_rate']}→{TARGET_SAMPLE_RATE}Hz")
                if original_info['subtype'] and original_info['subtype'] != TARGET_SUBTYPE:
                    conversion_info.append(f"Type: {original_info['subtype']}→{TARGET_SUBTYPE}")
                
                print(f"\n🎵 Файл: {filename}")
                print(f"📊 Оригинал: {original_info['format'].upper()}, {original_info['sample_rate']} Hz, {original_info['channels']} ch")
                print(f"🎯 Целевой формат: WAV, {TARGET_SAMPLE_RATE} Hz, {TARGET_SUBTYPE}")
                if conversion_info:
                    print(f"🔄 Конвертация: {', '.join(conversion_info)}")
                else:
                    print(f"✓ Формат уже соответствует целевому")
                print(f"⏱️  Длительность: {original_duration:.2f} сек")
                print(f"⏱️  Длительность фрагмента: {FRAGMENT_DURATION} сек")
                print(f"🔀 Перекрытие (overlap): {OVERLAP*100:.0f}%")
                print(f"🔪 Короткий хвост: {'отбрасывается' if DROP_SHORT_TAIL else 'сохраняется'}\n")

                # Нарезаем аудио
                fragments = slice_audio(
                    audio_data, 
                    sample_rate, 
                    fragment_duration=FRAGMENT_DURATION,
                    overlap=OVERLAP,
                    drop_short_tail=DROP_SHORT_TAIL
                )
                
                print(f"📊 Будет создано {len(fragments)} фрагментов\n")

                # Сохраняем каждый фрагмент
                for frag_idx, fragment in enumerate(fragments):
                    output_filename = f"{base_name}_{frag_idx}.wav"
                    output_path = os.path.join(output_base_path, output_filename)
                    
                    # ⭐ СОХРАНЯЕМ ВСЕГДА В ЦЕЛЕВОМ ФОРМАТЕ!
                    sf.write(output_path, fragment, TARGET_SAMPLE_RATE, subtype=TARGET_SUBTYPE)
                    
                    frag_duration = len(fragment) / sample_rate
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"  ✓ [{frag_idx}] {output_filename} ({frag_duration:.2f} сек, {size_mb:.2f} МБ)")

                print("\n" + "="*60)
                print(f"✅ Готово! Создано {len(fragments)} фрагментов")
                print(f"📂 Результат в папке: {output_base_path}")
                print("="*60)
            
            except Exception as e:
                print(f"\n✗ Ошибка: {e}")
                input("\nНажмите Enter для выхода...")
                continue

        else:
            print(f"\n✗ Неверный режим: {mode}. Установите mode = 0 или mode = 1")
            input("\nНажмите Enter для выхода...")
            exit()

        print("\nДля повторной обработки нажмите Enter или Ctrl+C для выхода.\n")
        
except KeyboardInterrupt:
    print("\n\n🛑 Программа остановлена пользователем. До свидания!")
    input("Нажмите Enter для закрытия окна...")