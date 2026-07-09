# is_drone_or_not
https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples

Микроинструкция:
1. Прописываем в data_sources папки с датасетом формата dataset/(train and val)/(drone and no_drone) 
2. Запускаем calibrate
3. Результаты из calibrate вписываем в spectrogram
4. Дальше все также генерация спектрограмм в main и обученеи модели в train
