# is_drone_or_not
Используемые датасеты: <br>
https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples <br>
https://www.kaggle.com/datasets/amineipad/drone-sound-audio-detection/data?select=Binary_Drone_Audio <br>
https://drive.google.com/file/d/1uJn_Fa4fCDpjmu8_EyEpO46-HIGa9bag/view <br>
Микроинструкция:
1. Прописываем в data_sources папки с датасетом формата dataset/(train and val)/(drone and no_drone) 
2. Запускаем calibrate
3. Результаты из calibrate вписываем в spectrogram
4. Дальше все также генерация спектрограмм в main и обученеи модели в train

-----------------------------------------------------------------------
09/07 была показана работа модели v0.2b.pt <br>
Для корректной работы модели необходимо изменить значения переменных VMIN_DB и VMAX_DB (51-52 стр.) в файле spectrogram.py на: <br>
VMIN_DB = -84.05 <br>
VMAX_DB = 14.66 <br>
