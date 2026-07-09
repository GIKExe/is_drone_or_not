import os
import re
import soundfile as sf
import numpy as np

# --- НАСТРОЙКИ РЕЖИМА РАБОТЫ ---
# mode = 1: обработка папки с WAV файлами
# mode = 0: обработка одного WAV файла
mode = 1

# !!! СЮДА ПУТЬ К ПАПКЕ С ФОНАМИ/ШУМАМИ !!!
background_folder = r"E:\Praktika\!_Parquet_To_Wav\Background_Noises"

# !!! СЮДА ПУТЬ, ГДЕ БУДУТ СОЗДАВАТЬСЯ ФАЙЛЫ С НАЛОЖЕННЫМ ФОНАМ !!!
output_base_path = r"E:\Praktika\!_Parquet_To_Wav\Noisy_Audio"   
os.makedirs(output_base_path, exist_ok=True)

# Настраиваем громкость накладываемого аудио (0.1 — тихий фон, 0.5 — громкий)
BACKGROUND_AMPLITUDE = 0.5

# --- ФУНКЦИЯ ДЛЯ НАЛОЖЕНИЯ АУДИО ---

def mix_audio_with_background(original_audio, background_audio, amplitude=0.3):
    """
    Накладывает фоновое аудио на оригинал
    
    Args:
        original_audio: оригинальное аудио (numpy array)
        background_audio: фоновое аудио (numpy array)
        amplitude: громкость фона (0.0 - 1.0)
    
    Returns:
        Смешанное аудио
    """
    original_length = len(original_audio)
    
    # Определяем количество каналов
    if len(original_audio.shape) > 1:
        original_channels = original_audio.shape[1]
    else:
        original_channels = 1
        original_audio = original_audio.reshape(-1, 1)
    
    if len(background_audio.shape) > 1:
        bg_channels = background_audio.shape[1]
    else:
        bg_channels = 1
        background_audio = background_audio.reshape(-1, 1)
    
    # Если каналы не совпадают, конвертируем фон
    if bg_channels != original_channels:
        if original_channels == 1:
            # Конвертируем стерео в моно
            background_audio = np.mean(background_audio, axis=1, keepdims=True)
        else:
            # Конвертируем моно в стерео (дублируем канал)
            background_audio = np.column_stack([background_audio] * original_channels)
    
    background_length = len(background_audio)
    
    # Если фон короче оригинала, зацикливаем его
    if background_length < original_length:
        repeats = (original_length // background_length) + 1
        background_audio = np.tile(background_audio, (repeats, 1))
        background_length = len(background_audio)
    
    # Выбираем случайную начальную позицию для вырезания фрагмента
    max_start = background_length - original_length
    start_pos = np.random.randint(0, max_start)
    
    # Вырезаем фрагмент нужной длины
    background_fragment = background_audio[start_pos:start_pos + original_length]
    
    # Применяем амплитуду к фону
    background_fragment = background_fragment * amplitude
    
    # Смешиваем и ограничиваем значения
    mixed = np.clip(original_audio + background_fragment, -1.0, 1.0)
    
    return mixed

# --- ОСНОВНОЙ ЦИКЛ ПРОГРАММЫ ---

try:
    while True:
        # Проверяем существование папки с фонами
        if not os.path.isdir(background_folder):
            print(f"\n✗ Папка с фонами не найдена: {background_folder}")
            input("\nНажмите Enter для выхода...")
            exit()

        # Находим все фоновые .wav файлы
        bg_files = [f for f in os.listdir(background_folder) if f.lower().endswith('.wav')]
        
        if len(bg_files) == 0:
            print(f"\n✗ В папке {background_folder} нет WAV файлов для наложения.")
            input("\nНажмите Enter для выхода...")
            exit()

        print(f"\n🎵 Найдено {len(bg_files)} фоновых файлов в: {background_folder}")

        if mode == 1:
            # === РЕЖИМ 1: ОБРАБОТКА ПАПКИ ===
            folder_path = input("Введите путь к папке с оригинальными WAV файлами: ").strip().strip('"').strip("'")

            if not os.path.isdir(folder_path):
                print(f"\n✗ Папка с оригиналами не найдена: {folder_path}")
                input("\nНажмите Enter для выхода...")
                exit()

            # Находим все .wav файлы
            wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
            
            print(f"📁 Найдено {len(wav_files)} оригинальных WAV файлов")

            if len(wav_files) == 0:
                print("Нет оригинальных файлов для обработки.")
                input("\nНажмите Enter для выхода...")
                exit()

            # Сортируем файлы по числу в имени
            def extract_number(filename):
                match = re.search(r'(\d+)', filename)
                if match:
                    return int(match.group(1))
                return 0

            wav_files.sort(key=extract_number)

            print(f"\n📂 Файлы с наложенным фоном будут сохранены в: {output_base_path}")
            print(f"🔊 Громкость фона (amplitude): {BACKGROUND_AMPLITUDE}")
            print(f"📊 Будет создано {len(wav_files) * len(bg_files)} файлов ({len(wav_files)} оригиналов × {len(bg_files)} фонов)\n")

            # Обрабатываем каждый файл
            total_files = len(wav_files) * len(bg_files)
            current_file = 0
            
            for file_idx, filename in enumerate(wav_files, 1):
                file_path = os.path.join(folder_path, filename)
                
                # ⭐ ПОЛУЧАЕМ СВОЙСТВА ОРИГИНАЛЬНОГО ФАЙЛА
                file_info = sf.info(file_path)
                original_subtype = file_info.subtype
                original_channels = file_info.channels
                sample_rate = file_info.samplerate
                
                # Читаем оригинал
                original_audio, sr = sf.read(file_path)
                
                base_name = os.path.splitext(filename)[0]
                
                print(f"🔹 [{file_idx}/{len(wav_files)}] Оригинальный файл: {filename}")

                # ⭐ ПЕРЕБИРАЕМ ВСЕ ФОНОВЫЕ ФАЙЛЫ
                for bg_filename in bg_files:
                    current_file += 1
                    bg_path = os.path.join(background_folder, bg_filename)
                    
                    # Читаем фон
                    background_audio, bg_sr = sf.read(bg_path)
                    
                    # Проверяем частоту дискретизации
                    if bg_sr != sample_rate:
                        print(f"  ⚠ Внимание: разная частота дискретизации! ({sample_rate} vs {bg_sr})")
                    
                    bg_base_name = os.path.splitext(bg_filename)[0]

                    # Накладываем фон
                    mixed_audio = mix_audio_with_background(
                        original_audio, 
                        background_audio, 
                        amplitude=BACKGROUND_AMPLITUDE
                    )
                    
                    # Формируем имя файла: оригинал_фон.wav
                    output_filename = f"{base_name}_{bg_base_name}.wav"
                    output_path = os.path.join(output_base_path, output_filename)
                    
                    # СОХРАНЯЕМ С ОРИГИНАЛЬНЫМ ФОРМАТОМ!
                    sf.write(output_path, mixed_audio, sample_rate, subtype=original_subtype)
                    
                    duration = len(mixed_audio) / sample_rate
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"  ✓ [{current_file}/{total_files}] {output_filename} ({duration:.2f} сек, {size_mb:.2f} МБ)")
                
                print()

            print("\n" + "="*60)
            print(f"✅ Готово! Создано {total_files} файлов")
            print(f"📂 Результат в папке: {output_base_path}")
            print("="*60)

        elif mode == 0:
            # === РЕЖИМ 0: ОБРАБОТКА ОДНОГО ФАЙЛА ===
            file_path = input("Введите путь к оригинальному WAV файлу: ").strip().strip('"').strip("'")

            if not os.path.isfile(file_path):
                print(f"\n✗ Оригинальный файл не найден: {file_path}")
                input("\nНажмите Enter для выхода...")
                exit()

            # Получаем свойства оригинала
            file_info = sf.info(file_path)
            original_subtype = file_info.subtype
            original_channels = file_info.channels
            sample_rate = file_info.samplerate
            
            # Читаем оригинал
            original_audio, sr = sf.read(file_path)
            
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0]
            
            print(f"\n🎵 Оригинальный файл: {filename}")
            print(f"📊 Характеристики: {original_subtype}, {original_channels} ch, {sample_rate} Hz")
            print(f"🔊 Громкость фона (amplitude): {BACKGROUND_AMPLITUDE}")
            print(f"📊 Будет создано {len(bg_files)} файлов (1 оригинал × {len(bg_files)} фонов)\n")

            # ⭐ ПЕРЕБИРАЕМ ВСЕ ФОНОВЫЕ ФАЙЛЫ
            for bg_idx, bg_filename in enumerate(bg_files, 1):
                bg_path = os.path.join(background_folder, bg_filename)
                
                # Читаем фон
                background_audio, bg_sr = sf.read(bg_path)
                
                if bg_sr != sample_rate:
                    print(f"⚠ Внимание: разная частота дискретизации! ({sample_rate} vs {bg_sr})")
                
                bg_base_name = os.path.splitext(bg_filename)[0]
                
                print(f"🔹 [{bg_idx}/{len(bg_files)}] Фон: {bg_filename}")

                # Накладываем фон
                mixed_audio = mix_audio_with_background(
                    original_audio, 
                    background_audio, 
                    amplitude=BACKGROUND_AMPLITUDE
                )
                
                # Формируем имя файла: оригинал_фон.wav
                output_filename = f"{base_name}_{bg_base_name}.wav"
                output_path = os.path.join(output_base_path, output_filename)
                
                sf.write(output_path, mixed_audio, sample_rate, subtype=original_subtype)
                
                duration = len(mixed_audio) / sample_rate
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  ✓ Сохранено: {output_filename} ({duration:.2f} сек, {size_mb:.2f} МБ)")
                print()

            print("\n" + "="*60)
            print(f"✅ Готово! Создано {len(bg_files)} файлов")
            print(f"📂 Результат в папке: {output_base_path}")
            print("="*60)

        else:
            print(f"\n✗ Неверный режим: {mode}. Установите mode = 0 или mode = 1")
            input("\nНажмите Enter для выхода...")
            exit()

        print("\nДля повторной обработки нажмите Enter или Ctrl+C для выхода.\n")
        
except KeyboardInterrupt:
    print("\n\n🛑 Программа остановлена пользователем. До свидания!")
    input("Нажмите Enter для закрытия окна...")