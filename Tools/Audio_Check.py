import soundfile as sf
import os

# Запрашиваем путь к файлу
file_path = input("Введите путь к WAV файлу для проверки: ").strip().strip('"').strip("'")

# Проверяем, что файл существует
if not os.path.isfile(file_path):
    print(f"\n✗ Файл не найден: {file_path}")
    input("\nНажмите Enter для выхода...")
    exit()

# Проверяем, что это WAV
if not file_path.lower().endswith('.wav'):
    print(f"\n⚠ Файл не имеет расширения .wav, но попробуем прочитать...")


# Читаем информацию о файле
info = sf.info(file_path)

print("\n" + "="*60)
print(f"📄 Файл: {os.path.basename(file_path)}")
print("="*60)
print(f"🎵 Частота дискретизации: {info.samplerate} Hz")
print(f"🎚  Битность (subtype):   {info.subtype}")
print(f"🔊 Каналы:                {info.channels}")
print(f"⏱  Длительность:          {info.duration:.2f} сек")
print(f"📊 Всего сэмплов:         {info.frames}")
print(f"💾 Формат:                {info.format}")
print("="*60)