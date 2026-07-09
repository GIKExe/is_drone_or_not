import os
import re
import soundfile as sf
import numpy as np
import librosa
import random

# --- НАСТРОЙКИ РЕЖИМА РАБОТЫ ---
# mode = 1: обработка папки с WAV файлами
# mode = 0: обработка одного WAV файла
mode = 1

# !!! СЮДА ПУТЬ, ГДЕ БУДУТ СОЗДАВАТЬСЯ ФАЙЛЫ С АУГМЕНТАЦИЯМИ !!!
output_base_path = r"E:\Praktika\!_New_Dataset\Done\Vavavavavav\Drone_Cut" 
os.makedirs(output_base_path, exist_ok=True)

# ⭐ НАСТРОЙКИ СИЛЫ ШУМА (0.0 - 1.0)
WHITE_NOISE_AMPLITUDE = 0.03      # Белый шум
PINK_NOISE_AMPLITUDE = 0.02       # Розовый шум
GAUSSIAN_NOISE_AMPLITUDE = 0.01   # Гауссовый шум

# ⭐ НАСТРОЙКИ АУГМЕНТАЦИЙ
PITCH_SHIFT_RANGE = (-2, 2)       # Диапазон сдвига высоты тона (полутоны)
TIME_STRETCH_RANGE = (0.9, 1.1)   # Диапазон изменения скорости
VOLUME_JITTER_DB = (-3, 3)        # Диапазон изменения громкости (дБ)

# ⭐ КОЛИЧЕСТВО ФАЙЛОВ НА ОДИН ИСХОДНЫЙ
NUM_OUTPUTS_PER_FILE = 3          # Сколько аугментированных версий создать для каждого файла

# --- ФУНКЦИИ ГЕНЕРАЦИИ ШУМА ---

def generate_white_noise(shape, amplitude=0.01):
    """Генерация белого шума (равномерное распределение)"""
    return np.random.uniform(-amplitude, amplitude, shape)

def generate_gaussian_noise(shape, amplitude=0.01):
    """Генерация гауссового шума (нормальное распределение)"""
    return np.random.normal(0, amplitude, shape)

def generate_pink_noise(shape, amplitude=0.01):
    """Генерация розового шума (1/f спектр) через FFT"""
    length = shape[0]
    channels = shape[1] if len(shape) > 1 else 1
    
    pink_channels = []
    for _ in range(channels):
        white = np.random.uniform(-1, 1, length)
        freqs = np.fft.rfftfreq(length, d=1.0)
        freqs[0] = 1.0
        
        spectrum = np.fft.rfft(white)
        pink_spectrum = spectrum / np.sqrt(freqs) 
        pink_1d = np.fft.irfft(pink_spectrum, n=length)
        pink_channels.append(pink_1d)
    
    if channels > 1:
        pink = np.column_stack(pink_channels)
    else:
        pink = pink_channels[0]
    
    pink = pink / np.max(np.abs(pink))
    return pink * amplitude

# --- ФУНКЦИИ АУГМЕНТАЦИИ ---

def apply_pitch_shift(audio_data, sr, n_steps):
    """Изменение высоты тона (pitch shift)"""
    if len(audio_data.shape) == 1:
        return librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=n_steps)
    else:
        channels = []
        for ch in range(audio_data.shape[1]):
            shifted = librosa.effects.pitch_shift(audio_data[:, ch], sr=sr, n_steps=n_steps)
            channels.append(shifted)
        return np.column_stack(channels)

def apply_time_stretch(audio_data, rate):
    """Изменение скорости воспроизведения (time stretch)"""
    if len(audio_data.shape) == 1:
        return librosa.effects.time_stretch(audio_data, rate=rate)
    else:
        channels = []
        for ch in range(audio_data.shape[1]):
            stretched = librosa.effects.time_stretch(audio_data[:, ch], rate=rate)
            channels.append(stretched)
        return np.column_stack(channels)

def apply_volume_jitter(audio_data, gain_db):
    """Изменение громкости (volume jitter)"""
    gain_linear = 10 ** (gain_db / 20)
    return audio_data * gain_linear

# --- ФУНКЦИЯ ПРИМЕНЕНИЯ ОДНОЙ СЛУЧАЙНОЙ АУГМЕНТАЦИИ ---

def apply_single_augmentation(audio_data, sr, aug_name):
    """
    Применяет ОДНУ указанную аугментацию
    Возвращает: (аугментированные данные, строка с параметрами)
    """
    result_data = audio_data.copy()
    param_str = ""
    
    if aug_name == "WhiteNoise":
        noise = generate_white_noise(result_data.shape, WHITE_NOISE_AMPLITUDE)
        result_data = np.clip(result_data + noise, -1.0, 1.0)
        param_str = f"W{WHITE_NOISE_AMPLITUDE:.2f}"
        
    elif aug_name == "PinkNoise":
        noise = generate_pink_noise(result_data.shape, PINK_NOISE_AMPLITUDE)
        result_data = np.clip(result_data + noise, -1.0, 1.0)
        param_str = f"P{PINK_NOISE_AMPLITUDE:.2f}"
        
    elif aug_name == "GaussianNoise":
        noise = generate_gaussian_noise(result_data.shape, GAUSSIAN_NOISE_AMPLITUDE)
        result_data = np.clip(result_data + noise, -1.0, 1.0)
        param_str = f"G{GAUSSIAN_NOISE_AMPLITUDE:.2f}"
        
    elif aug_name == "PitchShift":
        n_steps = random.uniform(PITCH_SHIFT_RANGE[0], PITCH_SHIFT_RANGE[1])
        result_data = apply_pitch_shift(result_data, sr, n_steps)
        result_data = np.clip(result_data, -1.0, 1.0)
        param_str = f"PS{n_steps:+.1f}"
        
    elif aug_name == "TimeStretch":
        rate = random.uniform(TIME_STRETCH_RANGE[0], TIME_STRETCH_RANGE[1])
        result_data = apply_time_stretch(result_data, rate)
        result_data = np.clip(result_data, -1.0, 1.0)
        param_str = f"TS{rate:.2f}"
        
    elif aug_name == "VolumeJitter":
        gain_db = random.uniform(VOLUME_JITTER_DB[0], VOLUME_JITTER_DB[1])
        result_data = apply_volume_jitter(result_data, gain_db)
        result_data = np.clip(result_data, -1.0, 1.0)
        param_str = f"V{gain_db:+.1f}"
    
    return result_data, param_str

# --- ФУНКЦИЯ ГЕНЕРАЦИИ НЕСКОЛЬКИХ РАЗНЫХ АУГМЕНТАЦИЙ ---

def generate_multiple_augmentations(audio_data, sr, num_outputs):
    """
    Генерирует num_outputs разных аугментаций для одного файла
    Возвращает: список кортежей (аугментированные_данные, параметры)
    """
    # Все возможные аугментации
    all_augmentations = [
        "WhiteNoise", "PinkNoise", "GaussianNoise",
        "PitchShift", "TimeStretch", "VolumeJitter"
    ]
    
    # Выбираем num_outputs разных аугментаций без повторений
    selected_augs = random.sample(all_augmentations, min(num_outputs, len(all_augmentations)))
    
    results = []
    for aug_name in selected_augs:
        augmented_data, param_str = apply_single_augmentation(audio_data, sr, aug_name)
        results.append((augmented_data, param_str))
    
    return results

# --- ОСНОВНОЙ ЦИКЛ ПРОГРАММЫ ---

try:
    while True:
        if mode == 1:
            # === РЕЖИМ 1: ОБРАБОТКА ПАПКИ ===
            folder_path = input("Введите путь к папке с WAV файлами: ").strip().strip('"').strip("'")

            if not os.path.isdir(folder_path):
                print(f"\n✗ Папка не найдена: {folder_path}")
                input("\nНажмите Enter для выхода...")
                exit()

            wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
            print(f"\n📁 Найдено {len(wav_files)} WAV файлов")

            if len(wav_files) == 0:
                print("Нет файлов для обработки.")
                input("\nНажмите Enter для выхода...")
                exit()

            def extract_number(filename):
                match = re.search(r'(\d+)', filename)
                if match:
                    return int(match.group(1))
                return 0

            wav_files.sort(key=extract_number)

            print(f"\n📂 Файлы с аугментациями будут сохранены в: {output_base_path}\n")
            print(f"🎲 Логика: для каждого файла создается {NUM_OUTPUTS_PER_FILE} версии с РАЗНЫМИ аугментациями")
            print(f"🔊 Параметры аугментаций:")
            print(f"   • Белый шум:    {WHITE_NOISE_AMPLITUDE}")
            print(f"   • Розовый шум:  {PINK_NOISE_AMPLITUDE}")
            print(f"   • Гауссов шум:  {GAUSSIAN_NOISE_AMPLITUDE}")
            print(f"   • Pitch Shift:  {PITCH_SHIFT_RANGE[0]:+.1f} до {PITCH_SHIFT_RANGE[1]:+.1f} полутонов")
            print(f"   • Time Stretch: {TIME_STRETCH_RANGE[0]:.2f}x до {TIME_STRETCH_RANGE[1]:.2f}x")
            print(f"   • Volume Jitter: {VOLUME_JITTER_DB[0]:+.1f} до {VOLUME_JITTER_DB[1]:+.1f} дБ\n")

            total_created = 0
            for file_idx, filename in enumerate(wav_files, 1):
                file_path = os.path.join(folder_path, filename)
                
                file_info = sf.info(file_path)
                original_subtype = file_info.subtype
                original_channels = file_info.channels
                sample_rate = file_info.samplerate
                
                audio_data, sr = sf.read(file_path)
                
                base_name = os.path.splitext(filename)[0]
                print(f"🔹 [{file_idx}/{len(wav_files)}] Обработка: {filename} ({original_subtype}, {original_channels} ch, {sample_rate} Hz)")

                # Генерируем несколько разных аугментаций
                augmentations = generate_multiple_augmentations(audio_data, sample_rate, NUM_OUTPUTS_PER_FILE)
                
                # Сохраняем каждую аугментацию как отдельный файл
                for aug_idx, (augmented_data, param_str) in enumerate(augmentations, 1):
                    output_filename = f"{base_name}_AUG{aug_idx}_{param_str}.wav"
                    output_path = os.path.join(output_base_path, output_filename)
                    
                    sf.write(output_path, augmented_data, sample_rate, subtype=original_subtype)
                    
                    duration = len(augmented_data) / sample_rate
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    
                    print(f"  ✓ [{aug_idx}/{len(augmentations)}] {param_str:10s} -> {output_filename} ({duration:.2f} сек, {size_mb:.2f} МБ)")
                    total_created += 1
                
                print()

            print("\n" + "="*60)
            print(f"✅ Готово! Для {len(wav_files)} файлов создано {total_created} аугментированных версий")
            print(f"📂 Результат в папке: {output_base_path}")
            print("="*60)

        elif mode == 0:
            # === РЕЖИМ 0: ОБРАБОТКА ОДНОГО ФАЙЛА ===
            file_path = input("Введите путь к WAV файлу: ").strip().strip('"').strip("'")

            if not os.path.isfile(file_path):
                print(f"\n✗ Файл не найден: {file_path}")
                input("\nНажмите Enter для выхода...")
                exit()

            if not file_path.lower().endswith('.wav'):
                print(f"\n✗ Файл не является WAV: {file_path}")
                input("\nНажмите Enter для выхода...")
                exit()

            file_info = sf.info(file_path)
            original_subtype = file_info.subtype
            original_channels = file_info.channels
            sample_rate = file_info.samplerate
            
            audio_data, sr = sf.read(file_path)
            
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0]
            
            print(f"\n🎵 Обработка файла: {filename}")
            print(f"📊 Характеристики: {original_subtype}, {original_channels} ch, {sample_rate} Hz")
            print(f"🎲 Логика: создание {NUM_OUTPUTS_PER_FILE} версий с РАЗНЫМИ аугментациями")
            print(f"🔊 Параметры аугментаций:")
            print(f"   • Белый шум:    {WHITE_NOISE_AMPLITUDE}")
            print(f"   • Розовый шум:  {PINK_NOISE_AMPLITUDE}")
            print(f"   • Гауссов шум:  {GAUSSIAN_NOISE_AMPLITUDE}")
            print(f"   • Pitch Shift:  {PITCH_SHIFT_RANGE[0]:+.1f} до {PITCH_SHIFT_RANGE[1]:+.1f} полутонов")
            print(f"   • Time Stretch: {TIME_STRETCH_RANGE[0]:.2f}x до {TIME_STRETCH_RANGE[1]:.2f}x")
            print(f"   • Volume Jitter: {VOLUME_JITTER_DB[0]:+.1f} до {VOLUME_JITTER_DB[1]:+.1f} дБ\n")

            # Генерируем несколько разных аугментаций
            augmentations = generate_multiple_augmentations(audio_data, sample_rate, NUM_OUTPUTS_PER_FILE)
            
            # Сохраняем каждую аугментацию как отдельный файл
            for aug_idx, (augmented_data, param_str) in enumerate(augmentations, 1):
                output_filename = f"{base_name}_AUG{aug_idx}_{param_str}.wav"
                output_path = os.path.join(output_base_path, output_filename)
                
                sf.write(output_path, augmented_data, sample_rate, subtype=original_subtype)
                
                duration = len(augmented_data) / sample_rate
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                
                print(f"✓ [{aug_idx}/{len(augmentations)}] {param_str:10s} -> {output_filename} ({duration:.2f} сек, {size_mb:.2f} МБ)")

            print("\n" + "="*60)
            print(f"✅ Готово! Создано {len(augmentations)} версий файла с разными аугментациями")
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