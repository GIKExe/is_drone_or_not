import tkinter as tk
from tkinter import ttk
import threading
import queue
from collections import deque
import os
import shutil
from time import time

from ultralytics import YOLO
import sounddevice as sd
import numpy as np
import librosa
import matplotlib.pyplot as plt


def create_spectrogram(audio_array, sample_rate):
    mel_spec = librosa.feature.melspectrogram(
        y=audio_array, sr=sample_rate, n_mels=128)
    mel_spec = librosa.power_to_db(mel_spec, ref=1.0)
    plt.imsave('micro0.png', np.flipud(mel_spec), cmap='gray')

    vmin = mel_spec.min()
    vmax = mel_spec.max()
    
    if vmax - vmin < 1e-6:
        vmax = vmin + 1
    
    mel_spec = ((mel_spec - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    mel_spec = np.flipud(mel_spec)

    plt.imsave('micro.png', mel_spec, cmap='gray')
    return np.stack((mel_spec,) * 3, axis=-1)


class DroneDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Детектор дронов")
        self.root.configure(bg='#1e1e1e')
        
        # Настройка адаптивности
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=0)
        
        # Переменные
        self.current_prob = 0.0
        self.avg_prob = 0.0
        self.drone_detected = False
        self.running = False
        self.data_queue = queue.Queue()
        self.detection_thread = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Стили
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background='#1e1e1e')
        style.configure('Dark.TLabel', background='#1e1e1e', foreground='#ffffff', font=('Arial', 10))
        style.configure('Title.TLabel', background='#1e1e1e', foreground='#ffffff', font=('Arial', 12, 'bold'))
        style.configure('Dark.TCombobox', fieldbackground='#2d2d2d', background='#2d2d2d', foreground='#ffffff')
        
        # Верхняя панель - выбор модели
        top_frame = ttk.Frame(self.root, style='Dark.TFrame', padding=10)
        top_frame.grid(row=0, column=0, sticky='ew')
        top_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(top_frame, text="Модель:", style='Title.TLabel').grid(row=0, column=0, padx=(0, 10), sticky='w')
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(top_frame, textvariable=self.model_var, state='readonly', style='Dark.TCombobox')
        self.model_combo.grid(row=0, column=1, sticky='ew', padx=(0, 10))
        
        # Привязка события изменения модели
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        self.start_btn = tk.Button(top_frame, text="Старт", command=self.toggle_detection,
                                   bg='#0078d4', fg='white', font=('Arial', 10, 'bold'),
                                   relief='flat', padx=20, pady=5)
        self.start_btn.grid(row=0, column=2, sticky='e')
        
        # Загрузка списка моделей
        self.load_models()
        
        # Центральная панель - индикатор
        center_frame = ttk.Frame(self.root, style='Dark.TFrame')
        center_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)
        center_frame.grid_columnconfigure(0, weight=1)
        center_frame.grid_rowconfigure(0, weight=1)
        
        self.indicator_canvas = tk.Canvas(center_frame, bg='#2d2d2d', highlightthickness=0)
        self.indicator_canvas.grid(row=0, column=0, sticky='nsew')
        
        # Привязка изменения размера
        self.indicator_canvas.bind('<Configure>', self.resize_indicator)
        
        # Индикатор (круг)
        self.indicator = self.indicator_canvas.create_oval(50, 50, 150, 150, fill='#3d3d3d', outline='#4d4d4d', width=3)
        self.indicator_text = self.indicator_canvas.create_text(100, 100, text="ОЖИДАНИЕ", fill='#888888', font=('Arial', 14, 'bold'))
        
        # Нижняя панель - карточки с вероятностями
        bottom_frame = ttk.Frame(self.root, style='Dark.TFrame', padding=10)
        bottom_frame.grid(row=2, column=0, sticky='ew')
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=1)
        
        # Карточка 1 - текущая вероятность
        card1 = tk.Frame(bottom_frame, bg='#2d2d2d', padx=20, pady=15)
        card1.grid(row=0, column=0, sticky='ew', padx=(0, 5))
        card1.grid_columnconfigure(0, weight=1)
        
        tk.Label(card1, text="Текущая вероятность", bg='#2d2d2d', fg='#888888',
                 font=('Arial', 9)).grid(row=0, column=0, sticky='w')
        self.current_prob_label = tk.Label(card1, text="0.00%", bg='#2d2d2d', fg='#00ff88',
                                           font=('Arial', 24, 'bold'))
        self.current_prob_label.grid(row=1, column=0, sticky='w')
        
        # Карточка 2 - средняя вероятность
        card2 = tk.Frame(bottom_frame, bg='#2d2d2d', padx=20, pady=15)
        card2.grid(row=0, column=1, sticky='ew', padx=(5, 0))
        card2.grid_columnconfigure(0, weight=1)
        
        tk.Label(card2, text="Средняя (2.5 сек)", bg='#2d2d2d', fg='#888888',
                 font=('Arial', 9)).grid(row=0, column=0, sticky='w')
        self.avg_prob_label = tk.Label(card2, text="0.00%", bg='#2d2d2d', fg='#00aaff',
                                       font=('Arial', 24, 'bold'))
        self.avg_prob_label.grid(row=1, column=0, sticky='w')
        
    def load_models(self):
        models = []
        for filename in os.listdir():
            if filename.startswith('v') and filename.endswith('.pt'):
                models.append(filename)
        
        if models:
            self.model_combo['values'] = models
            self.model_combo.current(0)
        else:
            self.model_combo['values'] = ['Нет моделей .pt']
            self.model_combo.current(0)
    
    def resize_indicator(self, event):
        width = event.width
        height = event.height
        size = min(width, height) * 0.6
        x1 = (width - size) / 2
        y1 = (height - size) / 2
        x2 = x1 + size
        y2 = y1 + size
        
        self.indicator_canvas.coords(self.indicator, x1, y1, x2, y2)
        self.indicator_canvas.coords(self.indicator_text, width/2, height/2)
    
    def on_model_change(self, event):
        """Обработчик изменения модели"""
        if self.running:
            # Останавливаем текущий процесс
            self.stop_detection()
            
            # Перезапускаем с новой моделью через небольшую задержку
            self.root.after(100, self.start_detection)
    
    def toggle_detection(self):
        if not self.running:
            if self.model_combo.get() == 'Нет моделей .pt':
                return
            self.start_detection()
        else:
            self.stop_detection()
    
    def start_detection(self):
        self.running = True
        self.start_btn.config(text="Стоп", bg='#d4380d')
        
        # Очистка очереди данных
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
        
        # Запуск потока обработки
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
        # Запуск обновления UI
        self.update_ui()
    
    def stop_detection(self):
        self.running = False
        self.start_btn.config(text="Старт", bg='#0078d4')
        
        # Ждем завершения потока (с таймаутом)
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        
        # Сброс индикатора
        self.indicator_canvas.itemconfig(self.indicator, fill='#3d3d3d', outline='#4d4d4d')
        self.indicator_canvas.itemconfig(self.indicator_text, text="ОЖИДАНИЕ", fill='#888888')
        self.current_prob_label.config(text="0.00%", fg='#00ff88')
        self.avg_prob_label.config(text="0.00%", fg='#00aaff')
    
    def detection_loop(self):
        os.makedirs('micro', exist_ok=True)
        
        model_name = self.model_combo.get()
        model = YOLO(model_name)
        
        sample_rate = 16000
        duration = 0.5
        frames_to_read = int(duration * sample_rate)
        
        list_size = 5
        median_list = deque(maxlen=list_size)
        
        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
                while self.running:
                    audio, overflowed = stream.read(frames_to_read)
                    audio = audio.flatten()
                    
                    img_array = create_spectrogram(audio, sample_rate)
                    results = model(img_array, save=False, verbose=False, rect=True)[0]
                    
                    q = float(results.probs.data[0])
                    
                    # if q > 0.1:
                    #     shutil.copy('micro.png', f'micro/micro_{int(time()*1000)}.png')
                    
                    median_list.append(q)
                    aq = sum(median_list) / list_size
                    mq = min(median_list)
                    drone_detected = (aq > 0.6) and (mq > 0.2)
                    
                    # Отправка данных в UI
                    self.data_queue.put((q, aq, drone_detected))
        except Exception as e:
            print(f"Ошибка в потоке детекции: {e}")
            if self.running:
                self.data_queue.put((0.0, 0.0, False))
    
    def update_ui(self):
        if not self.running:
            return
        
        try:
            while not self.data_queue.empty():
                q, aq, drone_detected = self.data_queue.get_nowait()
                
                self.current_prob = q
                self.avg_prob = aq
                self.drone_detected = drone_detected
                
                # Обновление индикатора
                if drone_detected:
                    self.indicator_canvas.itemconfig(self.indicator, fill='#ff0000', outline='#ff4444')
                    self.indicator_canvas.itemconfig(self.indicator_text, text="ДРОН", fill='#ffffff')
                else:
                    self.indicator_canvas.itemconfig(self.indicator, fill='#3d3d3d', outline='#4d4d4d')
                    self.indicator_canvas.itemconfig(self.indicator_text, text="ОЖИДАНИЕ", fill='#888888')
                
                # Обновление карточек
                self.current_prob_label.config(text=f"{q*100:.2f}%")
                self.avg_prob_label.config(text=f"{aq*100:.2f}%")
                
                # Цветовая индикация вероятностей
                if q > 0.6:
                    self.current_prob_label.config(fg='#ff4444')
                elif q > 0.3:
                    self.current_prob_label.config(fg='#ffaa00')
                else:
                    self.current_prob_label.config(fg='#00ff88')
                
                if aq > 0.6:
                    self.avg_prob_label.config(fg='#ff4444')
                elif aq > 0.3:
                    self.avg_prob_label.config(fg='#ffaa00')
                else:
                    self.avg_prob_label.config(fg='#00aaff')
        except queue.Empty:
            pass
        
        self.root.after(100, self.update_ui)


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("600x500")
    root.minsize(400, 400)
    
    app = DroneDetectorGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
