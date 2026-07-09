import os
import re
import soundfile as sf
import numpy as np

try:
    while True:
        # Запрашиваем путь к папке с WAV файлами
        folder_path = input("Введите путь к папке с WAV файлами: ").strip().strip('"').strip("'")

        if not os.path.isdir(folder_path):
            print(f"\n✗ Папка не найдена: {folder_path}")
            input("\nНажмите Enter для выхода...")
            exit()

        output_base_path = r"E:\Praktika\!_Parquet_To_Wav\Merged_Audio"   # !!!   СЮДА ПУТЬ, ГДЕ БУДУТ СОЗДАВАТЬСЯ НОВЫЕ ПАПКИ   !!!
        os.makedirs(output_base_path, exist_ok=True)

        # Извлекаем имя папки и формируем имя выходной
        folder_name = os.path.basename(os.path.normpath(folder_path))
        match = re.search(r'(train-\d{5})', folder_name)
        if match:
            output_folder_name = f"Merged_{match.group(1)}"
        else:
            print(f"⚠ Не удалось извлечь 'train-XXXXX' из имени папки '{folder_name}'")
            output_folder_name = f"Merged_{folder_name}"

        output_dir = os.path.join(output_base_path, output_folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # Находим все .wav файлы
        wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
        print(f"\n📁 Найдено {len(wav_files)} WAV файлов")

        if len(wav_files) == 0:
            print("Нет файлов для обработки.")
            input("\nНажмите Enter для выхода...")
            exit()

        # Извлекаем числа из имён файлов
        def extract_number(filename):
            match = re.search(r'(\d+)', filename)
            if match:
                return int(match.group(1))
            return None

        files_with_numbers = []
        for f in wav_files:
            num = extract_number(f)
            if num is not None:
                files_with_numbers.append((num, f))

        files_with_numbers.sort(key=lambda x: x[0])

        # Группируем файлы
        groups = []
        current_group = [files_with_numbers[0]]

        for i in range(1, len(files_with_numbers)):
            prev_num, _ = current_group[-1]
            curr_num, _ = files_with_numbers[i]
            
            if curr_num == prev_num + 1:
                current_group.append(files_with_numbers[i])
            else:
                groups.append(current_group)
                current_group = [files_with_numbers[i]]

        groups.append(current_group)

        print(f"\n📊 Найдено {len(groups)} групп")
        print(f"📂 Склеенные файлы будут сохранены в: {output_dir}\n")

        # Обрабатываем каждую группу
        for group_idx, group in enumerate(groups, 1):
            first_num = group[0][0]
            last_num = group[-1][0]
            
            output_filename = f"merged_{first_num}-{last_num}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"🔹 Группа {group_idx}/{len(groups)}: {first_num} → {last_num} ({len(group)} файлов)")
            
            # Получаем свойства первого файла
            first_file_path = os.path.join(folder_path, group[0][1])
            file_info = sf.info(first_file_path)
            original_subtype = file_info.subtype      # Например: 'PCM_16'
            original_channels = file_info.channels
            sample_rate = file_info.samplerate
            
            # Читаем и склеиваем все файлы
            all_audio = []
            
            for num, filename in group:
                file_path = os.path.join(folder_path, filename)
                audio_data, sr = sf.read(file_path)
                
                if sr != sample_rate:
                    print(f"  ⚠ Внимание: разная частота дискретизации!")
                
                all_audio.append(audio_data)
            
            merged_audio = np.concatenate(all_audio)
            
            # ⭐ СОХРАНЯЕМ С ОРИГИНАЛЬНЫМ ФОРМАТОМ!
            sf.write(output_path, merged_audio, sample_rate, subtype=original_subtype)
            
            duration = len(merged_audio) / sample_rate
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  ✓ Сохранено: {output_filename} ({duration:.2f} сек, {original_subtype}, {size_mb:.2f} МБ)")

        print("\n" + "="*60)
        print(f"✅ Готово! Все {len(groups)} групп склеены")
        print(f"📂 Результат в папке: {output_dir}")
        print("="*60)
        
except KeyboardInterrupt:
    print("\nПрограмма остановлена пользователем")
