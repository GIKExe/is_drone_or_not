# import torch
from ultralytics import YOLO

# ГЛАВНОЕ ИСПРАВЛЕНИЕ УТЕЧКИ ВО ВРЕМЯ ЭПОХИ ДЛЯ WINDOWS:
# Меняем стратегию передачи данных между процессами с file_descriptor на file_system.
# Это предотвращает накопление неосвобожденной памяти в Shared Memory.
# torch.multiprocessing.set_sharing_strategy('file_system')


def main():
	# 1. Инициализируем самую легкую модель YOLOv8 для классификации
	# Буква 'n' означает Nano (весит ~5 МБ, идеальна для портативок)
	# model = YOLO('yolov8n-cls.pt')
	model = YOLO('yolov8n-cls.pt')

	# 2. Запускаем обучение
	model.train(
		data='out',      # Путь к корневой папке с твоим датасетом
		epochs=20,       # Количество эпох (для начала 30-50 вполне хватит)
		imgsz=32,        # Размер картинок (должен совпадать с n_mels из прошлого скрипта)
		batch=512,      # Количество картинок, обрабатываемых за один шаг
		workers=4,       # Количество потоков процессора для загрузки данных
		device=0,        # 0 если есть видеокарта NVIDIA (CUDA), или 'cpu' если её нет

		mask_ratio=1,
		degrees=0.0,
		angle=0.0,
		mosaic=0.0,  # склеивает изображения
		erasing=0.0, # чёрные прямоугольники накладывает
		mixup=0.0,   # накладывает друг на друга с прозрачностью
		fliplr=0.0,
		flipud=0.0,
		translate=0.0,
		scale=0.0,
		rect=True,
		auto_augment=None,
	)

	print("Обучение успешно завершено!")


if __name__ == '__main__':
	main() 
