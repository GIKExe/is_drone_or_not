import os
import pandas as pd

Read = 0
Extarct = 1
output_base_path = r"E:\Praktika\!_Parquet_To_Wav\Extracted_Audio"   # !!!   СЮДА ПУТЬ, ГДЕ БУДУТ СОЗДАВАТЬСЯ НОВЫЕ ПАПКИ   !!!

try:
    while True:
        file_path = input("Введите путь к .parquet файлу: ").strip()
        file_path = file_path.strip('"').strip("'")

        if os.path.exists(file_path):
            print(f"✓ Файл найден: {file_path}")
            print(f"  Размер: {os.path.getsize(file_path) / (1024*1024):.2f} МБ")

            #----------------------------------------------------- Проверка Содержимого Начало
            if Read == 1:
                print("Читаю файл...")
                df = pd.read_parquet(file_path)

                print(f"\n✓ Файл успешно открыт!")
                print(f"  Всего строк: {len(df)}")
                print(f"  Всего столбцов: {len(df.columns)}")

                print("\n" + "="*60)
                print("НАЗВАНИЯ СТОЛБЦОВ:")
                print("="*60)
                for i, col in enumerate(df.columns, 1):
                    print(f"{i}. {col}")

                print("\n" + "="*60)
                print("СОДЕРЖИМОЕ ПЕРВОЙ СТРОКИ:")
                print("="*60)
                for col in df.columns:
                    value = df[col].iloc[0]
                    print(f"\nСтолбец '{col}':")
                    print(f"  Тип данных: {type(value).__name__}")
                    
                    # Если это словарь (часто бывает с аудио)
                    if isinstance(value, dict):
                        print(f"  Ключи: {list(value.keys())}")
                        for key, val in value.items():
                            if isinstance(val, bytes):
                                print(f"    - {key}: {len(val)} байт (бинарные данные)")
                            elif isinstance(val, (list, tuple)) and len(val) > 10:
                                print(f"    - {key}: массив из {len(val)} элементов")
                            else:
                                print(f"    - {key}: {val}")
                    # Если это строка
                    elif isinstance(value, str):
                        if len(value) > 100:
                            print(f"  Значение: {value[:100]}... (обрезано)")
                        else:
                            print(f"  Значение: {value}")
                    # Если это числа или массивы
                    else:
                        print(f"  Значение: {value}")
            else:
                print(f"\nДля проверки столбиков поменяй значение в переменной Read на 1")
            #----------------------------------------------------- Проверка Содержимого Конец

            #----------------------------------------------------- Извлечение Содержимого Начало
            if Extarct == 1:
                # Имя файла
                file_name = os.path.splitext(os.path.basename(file_path))[0] 

                # Создаём папку РЯДОМ СО СКРИПТОМ (а не рядом с parquet файлом)
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                output_dir = os.path.join(output_base_path, file_name)
                os.makedirs(output_dir, exist_ok=True)

                print("Загружаю файл...")
                df = pd.read_parquet(file_path)
                print(f"✓ Загружено {len(df)} записей")

                print(f"\nИзвлекаю аудиофайлы в папку: {output_dir}\n")

                # Проходим по всем строкам и сохраняем аудио
                for index, row in df.iterrows():
                    audio_data = row['audio']
                    
                    # Извлекаем бинарные данные и путь
                    audio_bytes = audio_data['bytes']
                    original_filename = audio_data['path']
                    
                    # Полный путь для сохранения
                    output_path = os.path.join(output_dir, original_filename)
                    
                    # Сохраняем файл
                    with open(output_path, 'wb') as f:
                        f.write(audio_bytes)
                    
                    # Показываем прогресс
                    if (index + 1) % 100 == 0 or index == len(df) - 1:
                        print(f"  [{index + 1}/{len(df)}] Сохранен: {original_filename}")

                print(f"\n✓ Готово! Все {len(df)} аудиофайлов сохранены в папку:")
                print(f"  {output_dir}")
            else:
                print(f"\nДля извлечения поменяй значение в переменной Extarct на 1")
            #----------------------------------------------------- Извлечение Содержимого Конец

        else:
            print(f"✗ Файл НЕ найден: {file_path}")
            print(f"  Текущая папка: {os.getcwd()}")
            print("  Файлы в текущей папке:")
            for f in os.listdir('.'):
                if f.endswith('.parquet'):
                    print(f"    - {f}")
except KeyboardInterrupt:
    print("\nПрограмма остановлена пользователем")