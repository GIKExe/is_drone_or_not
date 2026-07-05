
import shutil
import random
from pathlib import Path

def split_files_80_20(source_dir: str, dir_80: str, dir_20: str, ratio: float = 0.8, seed: int | None = None):
	"""
	Случайным образом делит файлы из source_dir на две папки в соотношении ratio/(1-ratio).
	По умолчанию: 80% / 20%.
	"""
	source_path = Path(source_dir)
	if not source_path.is_dir():
		raise FileNotFoundError(f"❌ Исходная папка '{source_dir}' не найдена.")

	# Собираем только файлы (игнорируем подпапки и скрытые системные файлы)
	files = [f for f in source_path.iterdir() if f.is_file()]

	if not files:
		print("⚠️ В папке нет файлов для разделения.")
		return

	# Фиксируем генератор случайных чисел для воспроизводимости результатов
	if seed is not None:
		random.seed(seed)

	random.shuffle(files)

	# Вычисляем границу разделения
	split_idx = int(len(files) * ratio)
	part_80 = files[:split_idx]
	part_20 = files[split_idx:]

	# Создаём целевые директории
	Path(dir_80).mkdir(parents=True, exist_ok=True)
	Path(dir_20).mkdir(parents=True, exist_ok=True)

	# Копируем файлы (shutil.copy сохраняет содержимое, shutil.copy2 сохраняет метаданные)
	print(f"📦 Копирование {len(part_80)} файлов в '{dir_80}'...")
	for f in part_80:
		shutil.copy(f, Path(dir_80) / f.name)

	print(f"📦 Копирование {len(part_20)} файлов в '{dir_20}'...")
	for f in part_20:
		shutil.copy(f, Path(dir_20) / f.name)

	print(f"✅ Готово! {len(part_80)} файлов → '{dir_80}', {len(part_20)} файлов → '{dir_20}'")


if __name__ == "__main__":
	# 🔧 НАСТРОЙКИ
	SOURCE_FOLDER = "micro"      # Папка с исходными файлами
	FOLDER_80     = "micro80"     # Куда сохранить 80%
	FOLDER_20     = "micro20"     # Куда сохранить 20%
	RANDOM_SEED   = 42            # Для воспроизводимости. None = полностью случайно

	split_files_80_20(SOURCE_FOLDER, FOLDER_80, FOLDER_20, ratio=0.8, seed=RANDOM_SEED)