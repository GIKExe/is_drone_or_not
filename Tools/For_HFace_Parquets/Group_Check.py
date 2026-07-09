import os
import re

# Запрашиваем путь к папке
folder_path = input("Введите путь к папке с WAV файлами: ").strip().strip('"').strip("'")

# Проверяем, что папка существует
if not os.path.isdir(folder_path):
    print(f"\n✗ Папка не найдена: {folder_path}")
    input("\nНажмите Enter для выхода...")
    exit()

# Находим все .wav файлы
wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
print(f"\n📁 Найдено {len(wav_files)} WAV файлов")

if len(wav_files) == 0:
    print("Нет файлов для обработки.")
    input("\nНажмите Enter для выхода...")
    exit()

# Извлекаем числа из имён файлов и сортируем
def extract_number(filename):
    """Извлекает число из имени файла вида drone-26.wav"""
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

# Создаём список (число, имя_файла)
files_with_numbers = []
for f in wav_files:
    num = extract_number(f)
    if num is not None:
        files_with_numbers.append((num, f))
    else:
        print(f"⚠ Не удалось извлечь число из: {f}")

# Сортируем по числу
files_with_numbers.sort(key=lambda x: x[0])

# Группируем файлы по последовательностям
groups = []
current_group = [files_with_numbers[0]]

for i in range(1, len(files_with_numbers)):
    prev_num, prev_file = current_group[-1]
    curr_num, curr_file = files_with_numbers[i]
    
    # Если число идёт подряд (+1) — добавляем в текущую группу
    if curr_num == prev_num + 1:
        current_group.append(files_with_numbers[i])
    else:
        # Иначе — сохраняем текущую группу и начинаем новую
        groups.append(current_group)
        current_group = [files_with_numbers[i]]

# Не забываем добавить последнюю группу
groups.append(current_group)

# Выводим результат (без имён файлов)
print("\n" + "="*60)
print(f"📊 НАЙДЕНО ГРУПП: {len(groups)}")
print("="*60)

total_files = 0
for i, group in enumerate(groups, 1):
    first_num = group[0][0]
    last_num = group[-1][0]
    count = len(group)
    total_files += count
    print(f"🔹 Группа {i:>3}: числа {first_num:>5} → {last_num:>5}  ({count} файлов)")

print("="*60)
print(f"📈 Итого: {len(groups)} групп, {total_files} файлов")
print("="*60)

input("\nНажмите Enter для выхода...")