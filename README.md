# is_drone_or_not
Используемые датасеты: <br>
https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples <br>
https://www.kaggle.com/datasets/amineipad/drone-sound-audio-detection/data?select=Binary_Drone_Audio

Микроинструкция:
1. Прописываем в data_sources папки с датасетом формата dataset/(train and val)/(drone and no_drone) 
2. Запускаем calibrate
3. Результаты из calibrate вписываем в spectrogram
4. Дальше все также генерация спектрограмм в main и обученеи модели в train
