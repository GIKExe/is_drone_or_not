from collections import deque
from time import time
import os
import threading
import tkinter as tk
from tkinter import messagebox

import numpy as np
import sounddevice as sd
import librosa
import matplotlib
matplotlib.use('Agg')          # без всплывающих окон matplotlib
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ====================================================================== #
#                           Спектрограмма                                #
# ====================================================================== #
def create_spectrogram(audio_array, sample_rate, save_micro=False):
	mel_spec = librosa.feature.melspectrogram(
		y=audio_array, sr=sample_rate, n_mels=128)
	mel_spec = librosa.power_to_db(mel_spec, ref=1.0)

	vmin = mel_spec.min()
	vmax = mel_spec.max()

	if vmax - vmin < 1e-6:
		vmax = vmin + 1

	mel_spec = ((mel_spec - vmin) / (vmax - vmin) * 255).astype(np.uint8)
	mel_spec = np.flipud(mel_spec)

	if save_micro:
		plt.imsave('micro.png', mel_spec, cmap='gray')

	img_rgb = np.stack((mel_spec,) * 3, axis=-1)
	return img_rgb, mel_spec


# ====================================================================== #
#                    Главное окно — выбор режима                         #
# ====================================================================== #
# class MainWindow:
# 	def __init__(self, root):
# 		self.root = root
# 		self.root.title("Детектор дронов — выбор режима")
# 		self.root.geometry("420x320")
# 		self.root.configure(bg="#1e1e1e")
# 		self.root.resizable(False, False)

# 		tk.Label(
# 			root, text="🎧 АКУСТИЧЕСКИЙ ДЕТЕКТОР ДРОНОВ",
# 			font=("Arial", 14, "bold"), bg="#1e1e1e", fg="#e0e0e0"
# 		).pack(pady=(20, 25))

# 		tk.Label(
# 			root, text="Выберите режим работы:",
# 			font=("Arial", 11), bg="#1e1e1e", fg="#bbbbbb"
# 		).pack(pady=(0, 15))

# 		modes = [
# 			("1) Тест модели (онлайн-детекция)", 1, "#2e7d32"),
# 			("2) Запись шума (сохранение фрагментов)", 2, "#1565c0"),
# 			("3) Полная запись (всё подряд)", 3, "#6a1b9a"),
# 		]
# 		for text, mode, color in modes:
# 			tk.Button(
# 				root, text=text, width=34, font=("Arial", 11),
# 				bg=color, fg="white", activebackground=color,
# 				command=lambda m=mode: self._open_mode(m)
# 			).pack(pady=6)

# 	def _open_mode(self, mode):
# 		self.root.withdraw()                    # скрываем главное окно
# 		if mode == 1:
# 			TestModelWindow(tk.Toplevel(self.root), self.root)
# 		else:
# 			RecordWindow(tk.Toplevel(self.root), self.root, mode)


# ====================================================================== #
#        Режим 1: Тест модели — выбор модели + индикатор дрона           #
# ====================================================================== #
class TestModelWindow:
	MODELS: list

	def __init__(self, root):

		self.MODELS = list()
		for filename in os.listdir():
			if not filename.startswith('v'):
				continue 
			if not filename.endswith('.pt'):
				continue
			self.MODELS.append(filename)

		self.root = root
		self.root.title("Тест модели — детекция дрона")
		self.root.geometry("480x560")
		self.root.configure(bg="#1e1e1e")
		self.root.resizable(False, False)
		self.root.protocol("WM_DELETE_WINDOW", self._on_close)

		tk.Label(
			root, text="Выберите модель:",
			font=("Arial", 12, "bold"), bg="#1e1e1e", fg="#e0e0e0"
		).pack(pady=(15, 8))

		# --- Список моделей ---
		self.model_var = tk.StringVar(value=self.MODELS[0])
		list_frame = tk.Frame(root, bg="#1e1e1e")
		list_frame.pack(pady=5)

		for m in self.MODELS:
			exists = os.path.exists(m)
			text = m if exists else f"{m}  (не найден)"
			rb = tk.Radiobutton(
				list_frame, text=text, variable=self.model_var, value=m,
				font=("Consolas", 11), bg="#1e1e1e", fg="#e0e0e0",
				selectcolor="#333333", activebackground="#1e1e1e",
				activeforeground="#e0e0e0",
				state=tk.NORMAL if exists else tk.DISABLED
			)
			rb.pack(anchor=tk.W, pady=2)

		# --- Кнопки ---
		btn_frame = tk.Frame(root, bg="#1e1e1e")
		btn_frame.pack(pady=12)

		self.start_btn = tk.Button(
			btn_frame, text="▶ Старт", width=12, font=("Arial", 11, "bold"),
			bg="#2e7d32", fg="white", command=self.start
		)
		self.start_btn.pack(side=tk.LEFT, padx=8)

		self.stop_btn = tk.Button(
			btn_frame, text="■ Стоп", width=12, font=("Arial", 11, "bold"),
			bg="#c62828", fg="white", command=self.stop, state=tk.DISABLED
		)
		self.stop_btn.pack(side=tk.LEFT, padx=8)

		# --- Индикатор ---
		self.canvas = tk.Canvas(root, width=220, height=220,
								bg="#1e1e1e", highlightthickness=0)
		self.canvas.pack(pady=5)
		self.canvas.create_oval(30, 30, 190, 190, fill="#2a2a2a", outline="#444", width=2)
		self.indicator = self.canvas.create_oval(
			50, 50, 170, 170, fill="#555555", outline="white", width=3
		)
		self.indicator_text = self.canvas.create_text(
			110, 110, text="ОЖИДАНИЕ",
			font=("Arial", 14, "bold"), fill="white"
		)

		# --- Статус и метрики ---
		self.status_label = tk.Label(
			root, text="Нажмите «Старт»", font=("Arial", 11),
			bg="#1e1e1e", fg="#aaaaaa"
		)
		self.status_label.pack(pady=3)

		self.metrics_label = tk.Label(
			root, text="Вер-ть: —     Среднее: —",
			font=("Consolas", 11), bg="#1e1e1e", fg="#888888"
		)
		self.metrics_label.pack(pady=2)

		# --- Состояние ---
		self.running = False
		self.thread = None
		self.drone_state = False

	# ---------- Запуск / остановка ----------
	def start(self):
		model_name = self.model_var.get()
		if not os.path.exists(model_name):
			messagebox.showerror("Ошибка", f"Файл модели не найден:\n{model_name}")
			return

		try:
			self.model = YOLO(model_name)
		except Exception as e:
			messagebox.showerror("Ошибка загрузки модели", str(e))
			return

		self.running = True
		self.start_btn.config(state=tk.DISABLED)
		self.stop_btn.config(state=tk.NORMAL)
		self.status_label.config(text=f"🟢 Работа на модели: {model_name}", fg="#66bb6a")

		self.thread = threading.Thread(target=self._audio_loop, daemon=True)
		self.thread.start()

	def stop(self):
		self.running = False
		self.start_btn.config(state=tk.NORMAL)
		self.stop_btn.config(state=tk.DISABLED)
		self._set_indicator(False)
		self.status_label.config(text="Остановлено", fg="#aaaaaa")
		self.metrics_label.config(text="Вер-ть: —     Среднее: —")

	def _on_close(self):
		self.stop()
		self.root.destroy()

	# ---------- Аудио-цикл ----------
	def _audio_loop(self):
		sample_rate = 16000
		duration = 0.5
		frames_to_read = int(duration * sample_rate)
		list_size = 5
		median_list = deque(maxlen=list_size)

		try:
			with sd.InputStream(samplerate=sample_rate, channels=1,
								dtype='float32') as stream:
				while self.running:
					audio, _ = stream.read(frames_to_read)
					audio = audio.flatten()

					img_array, _ = create_spectrogram(
						audio, sample_rate, save_micro=True
					)
					results = self.model(img_array, save=False, verbose=False)[0]
					q = float(results.probs.data[0])

					median_list.append(q)
					aq = sum(median_list) / list_size
					mq = min(median_list)

					drone = (aq > 0.6) and (mq > 0.2)

					# Обновляем GUI из потока через after()
					self.root.after(
						0, self._update_ui, q, aq, drone
					)
		except Exception as e:
			self.root.after(0, self.status_label.config,
							  {"text": f"Ошибка: {e}", "fg": "#ff5252"})
			self.root.after(0, self.stop)

	# ---------- Обновление GUI ----------
	def _update_ui(self, q, aq, drone):
		self.metrics_label.config(
			text=f"Вер-ть: {q*100:6.2f}%     Среднее: {aq*100:6.2f}%"
		)
		self._set_indicator(drone)

	def _set_indicator(self, drone_detected):
		if drone_detected == self.drone_state:
			return
		self.drone_state = drone_detected
		if drone_detected:
			self.canvas.itemconfig(self.indicator, fill="#d32f2f", outline="#ff5252")
			self.canvas.itemconfig(self.indicator_text, text="ДРОН!", fill="white")
			self.status_label.config(text="🚨 ОБНАРУЖЕН ДРОН 🚨", fg="#ff5252")
		else:
			self.canvas.itemconfig(self.indicator, fill="#2e7d32", outline="#66bb6a")
			self.canvas.itemconfig(self.indicator_text, text="ЧИСТО", fill="white")
			self.status_label.config(text="Сигнал чистый", fg="#66bb6a")


# ====================================================================== #
#       Режимы 2 и 3: Запись шума / Полная запись (с GUI-обёрткой)       #
# ====================================================================== #
# class RecordWindow:
# 	def __init__(self, window, main_root, mode):
# 		self.window = window
# 		self.main_root = main_root
# 		self.mode = mode
# 		mode_name = {2: "Запись шума", 3: "Полная запись"}[mode]

# 		self.window.title(f"Режим: {mode_name}")
# 		self.window.geometry("500x380")
# 		self.window.configure(bg="#1e1e1e")
# 		self.window.resizable(False, False)
# 		self.window.protocol("WM_DELETE_WINDOW", self._on_close)

# 		tk.Label(
# 			window, text=f"📼 {mode_name.upper()}",
# 			font=("Arial", 14, "bold"), bg="#1e1e1e", fg="#e0e0e0"
# 		).pack(pady=(15, 10))

# 		# --- Выбор модели ---
# 		tk.Label(
# 			window, text="Модель для анализа:",
# 			font=("Arial", 11), bg="#1e1e1e", fg="#bbbbbb"
# 		).pack(anchor=tk.W, padx=30)

# 		self.model_var = tk.StringVar(value=TestModelWindow.MODELS[0])
# 		model_frame = tk.Frame(window, bg="#1e1e1e")
# 		model_frame.pack(pady=5)
# 		for m in TestModelWindow.MODELS:
# 			exists = os.path.exists(m)
# 			tk.Radiobutton(
# 				model_frame, text=m if exists else f"{m} (нет)",
# 				variable=self.model_var, value=m,
# 				font=("Consolas", 10), bg="#1e1e1e", fg="#e0e0e0",
# 				selectcolor="#333333", activebackground="#1e1e1e",
# 				state=tk.NORMAL if exists else tk.DISABLED
# 			).pack(anchor=tk.W)

# 		# --- Кнопки ---
# 		btn_frame = tk.Frame(window, bg="#1e1e1e")
# 		btn_frame.pack(pady=10)

# 		self.start_btn = tk.Button(
# 			btn_frame, text="▶ Начать запись", width=16,
# 			font=("Arial", 11, "bold"), bg="#1565c0", fg="white",
# 			command=self.start
# 		)
# 		self.start_btn.pack(side=tk.LEFT, padx=8)

# 		self.stop_btn = tk.Button(
# 			btn_frame, text="■ Стоп", width=16,
# 			font=("Arial", 11, "bold"), bg="#c62828", fg="white",
# 			command=self.stop, state=tk.DISABLED
# 		)
# 		self.stop_btn.pack(side=tk.LEFT, padx=8)

# 		# --- Лог ---
# 		self.log = tk.Text(
# 			window, height=8, width=58, font=("Consolas", 10),
# 			bg="#0f0f0f", fg="#66bb6a", insertbackground="white",
# 			state=tk.DISABLED
# 		)
# 		self.log.pack(pady=10)

# 		self.status_label = tk.Label(
# 			window, text="Готов к записи", font=("Arial", 11),
# 			bg="#1e1e1e", fg="#aaaaaa"
# 		)
# 		self.status_label.pack()

# 		self.running = False
# 		self.thread = None

# 	# ---------- Лог ----------
# 	def _append_log(self, text):
# 		self.log.config(state=tk.NORMAL)
# 		self.log.insert(tk.END, text + "\n")
# 		self.log.see(tk.END)
# 		self.log.config(state=tk.DISABLED)

# 	# ---------- Старт / стоп ----------
# 	def start(self):
# 		model_name = self.model_var.get()
# 		if not os.path.exists(model_name):
# 			messagebox.showerror("Ошибка", f"Модель не найдена:\n{model_name}")
# 			return
# 		try:
# 			self.model = YOLO(model_name)
# 		except Exception as e:
# 			messagebox.showerror("Ошибка", str(e))
# 			return

# 		os.makedirs('micro', exist_ok=True)

# 		self.running = True
# 		self.start_btn.config(state=tk.DISABLED)
# 		self.stop_btn.config(state=tk.NORMAL)
# 		self.status_label.config(
# 			text=f"🔴 Идёт запись (модель: {model_name})", fg="#ff5252"
# 		)
# 		self.window.after(0, self._append_log,
# 						  f"=== Старт: режим {self.mode}, модель {model_name} ===")

# 		self.thread = threading.Thread(target=self._record_loop, daemon=True)
# 		self.thread.start()

# 	def stop(self):
# 		self.running = False
# 		self.start_btn.config(state=tk.NORMAL)
# 		self.stop_btn.config(state=tk.DISABLED)
# 		self.status_label.config(text="Запись остановлена", fg="#aaaaaa")
# 		self.window.after(0, self._append_log, "=== Запись остановлена ===")

# 	def _on_close(self):
# 		self.stop()
# 		self.window.destroy()
# 		self.main_root.deiconify()

# 	# ---------- Цикл записи ----------
# 	def _record_loop(self):
# 		sample_rate = 16000
# 		duration = 0.5
# 		frames_to_read = int(duration * sample_rate)
# 		list_size = 5
# 		median_list = deque(maxlen=list_size)

# 		try:
# 			with sd.InputStream(samplerate=sample_rate, channels=1,
# 								dtype='float32') as stream:
# 				while self.running:
# 					audio, _ = stream.read(frames_to_read)
# 					audio = audio.flatten()

# 					img_array, mel_spec_normalized = create_spectrogram(
# 						audio, sample_rate, save_micro=False
# 					)
# 					results = self.model(img_array, save=False, verbose=False)[0]
# 					q = float(results.probs.data[0])

# 					# Сохранение по логике оригинального test.py
# 					if ((q > 0.8) and (self.mode == 2)) or (self.mode == 3):
# 						fname = f'micro/micro_{int(time()*1000)}.png'
# 						plt.imsave(fname, mel_spec_normalized, cmap='gray')
# 						saved_msg = f"💾 Сохранено: {fname}"
# 						self.window.after(0, self._append_log, saved_msg)

# 					median_list.append(q)
# 					aq = sum(median_list) / list_size
# 					mq = min(median_list)
# 					text = 'ДРОН' if (aq > 0.6) and (mq > 0.2) else '    '

# 					line = (f"{text}  Вероятность: {q*100:6.2f}%"
# 							f"  Среднее: {aq*100:6.2f}%")
# 					self.window.after(0, self._append_log, line)

# 		except Exception as e:
# 			self.window.after(0, self._append_log, f"❌ Ошибка: {e}")
# 			self.window.after(0, self.stop)


# ====================================================================== #
#                                 Запуск                                 #
# ====================================================================== #
if __name__ == '__main__':
	try:
		root = tk.Tk()
		app = TestModelWindow(root)
		root.mainloop()
	except KeyboardInterrupt:
		pass
	except:
		raise