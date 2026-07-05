import os
import threading
import queue
import tkinter as tk
from tkinter import ttk
from collections import deque
import numpy as np
import librosa
import sounddevice as sd
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')

BG_MAIN = "#1a1a1a"
BG_PANEL = "#252525"
BG_CARD = "#2d2d2d"
ACCENT_GREEN = "#00d26a"
ACCENT_RED = "#ff3b3b"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#8a8a8a"
BORDER_COLOR = "#3a3a3a"

def create_spectrogram(audio_array, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sample_rate, n_mels=128)
    mel_spec = librosa.power_to_db(mel_spec, ref=1.0)
    min_db = 80
    mel_spec = np.clip(mel_spec, a_min=-min_db, a_max=0.0)
    mel_spec = ((mel_spec + min_db) / min_db * 255).astype(np.uint8)
    mel_spec = np.flipud(mel_spec)
    return np.stack((mel_spec,) * 3, axis=-1)

class DroneDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Система обнаружения дронов")
        self.root.geometry("500x750")
        self.root.resizable(False, False)
        self.root.configure(bg=BG_MAIN)
        self.root.option_add("*TCombobox*Listbox*Background", "#2d2d2d")
        self.root.option_add("*TCombobox*Listbox*Foreground", "#ffffff")
        self.root.option_add("*TCombobox*Listbox*selectBackground", "#00d26a")
        self.root.option_add("*TCombobox*Listbox*selectForeground", "#ffffff")

        self.running = False
        self.thread = None
        self.msg_queue = queue.Queue()
        self.models = self.find_models()

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Dark.TCombobox",
                             fieldbackground=BG_CARD,
                             background=BG_CARD,
                             foreground=TEXT_PRIMARY,
                             bordercolor=BORDER_COLOR,
                             arrowcolor=TEXT_PRIMARY,
                             padding=8)
        self.style.map("Dark.TCombobox",
                       fieldbackground=[("readonly", BG_CARD)],
                       selectbackground=[("readonly", BG_CARD)],
                       selectforeground=[("readonly", TEXT_PRIMARY)])

        self.create_widgets()
        self.update_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def find_models(self):
        models = []
        for filename in os.listdir():
            if filename.startswith('v') and filename.endswith('.pt'):
                models.append(filename)
        return models

    def create_widgets(self):
        header = tk.Frame(self.root, bg=BG_MAIN, pady=15)
        header.pack(fill=tk.X)
        tk.Label(header, text="ОБНАРУЖЕНИЕ ДРОНОВ",
                 font=("Segoe UI", 18, "bold"),
                 fg=TEXT_PRIMARY, bg=BG_MAIN).pack()
        tk.Label(header, text="Система акустического мониторинга",
                 font=("Segoe UI", 10),
                 fg=TEXT_SECONDARY, bg=BG_MAIN).pack()

        model_panel = tk.Frame(self.root, bg=BG_PANEL, padx=20, pady=15)
        model_panel.pack(fill=tk.X, padx=20)

        tk.Label(model_panel, text="МОДЕЛЬ",
                 font=("Segoe UI", 9, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_PANEL).pack(anchor=tk.W)

        if self.models:
            self.model_var = tk.StringVar()
            self.model_combo = ttk.Combobox(model_panel,
                                            textvariable=self.model_var,
                                            state="readonly",
                                            width=35,
                                            font=("Segoe UI", 11),
                                            style="Dark.TCombobox")
            self.model_combo['values'] = self.models
            self.model_combo.current(0)
            self.model_combo.pack(fill=tk.X, pady=(5, 0))
            self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)
        else:
            tk.Label(model_panel, text="Модели .pt не найдены",
                     fg=ACCENT_RED, bg=BG_PANEL,
                     font=("Segoe UI", 11)).pack(pady=(5, 0))
            self.model_combo = None

        self.btn_canvas = tk.Canvas(self.root, width=460, height=70,
                                    bg=BG_MAIN, highlightthickness=0)
        self.btn_canvas.pack(pady=15)
        self.btn_bg = self.btn_canvas.create_rectangle(
            0, 0, 460, 70, fill=ACCENT_GREEN, outline="", width=0)
        self.btn_text = self.btn_canvas.create_text(
            230, 35, text="СТАРТ", fill="#ffffff",
            font=("Segoe UI", 16, "bold"))
        self.btn_canvas.bind("<Button-1>", lambda e: self.toggle_detection())
        self.btn_canvas.bind("<Enter>", lambda e: self.btn_canvas.config(cursor="hand2"))
        self.btn_canvas.bind("<Leave>", lambda e: self.btn_canvas.config(cursor=""))

        self.indicator_canvas = tk.Canvas(self.root, width=280, height=280,
                                          bg=BG_MAIN, highlightthickness=0)
        self.indicator_canvas.pack(pady=5)

        self.glow_colors_off = [BG_MAIN, BG_MAIN, BG_MAIN]
        self.glow_colors_on = ["#2a0a0a", "#4a1010", "#7a1a1a"]
        self.glow_layers = []
        for i, r in enumerate([130, 115, 100]):
            layer = self.indicator_canvas.create_oval(
                140 - r, 140 - r, 140 + r, 140 + r,
                fill=self.glow_colors_off[i], outline="")
            self.glow_layers.append(layer)

        self.indicator = self.indicator_canvas.create_oval(
            60, 60, 220, 220, fill="#3a3a3a", outline="#4a4a4a", width=2)
        self.indicator_inner = self.indicator_canvas.create_oval(
            80, 80, 200, 200, fill="#2a2a2a", outline="")

        self.status_label = tk.Label(self.root, text="ОЖИДАНИЕ",
                                     font=("Segoe UI", 16, "bold"),
                                     fg=TEXT_SECONDARY, bg=BG_MAIN)
        self.status_label.pack()

        stats_frame = tk.Frame(self.root, bg=BG_MAIN, pady=20, padx=20)
        stats_frame.pack(fill=tk.X)

        tk.Label(stats_frame, text="ТЕКУЩАЯ",
                 font=("Segoe UI", 9, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_MAIN).grid(row=0, column=0, pady=(0, 8))

        self.lbl_current = tk.Label(stats_frame, text="0.00%",
                                    font=("Segoe UI", 20, "bold"),
                                    fg=TEXT_PRIMARY, bg=BG_CARD,
                                    width=12, anchor=tk.W,
                                    highlightbackground=BORDER_COLOR,
                                    highlightthickness=1)
        self.lbl_current.grid(row=1, column=0, padx=(0, 5), sticky=tk.EW)

        tk.Label(stats_frame, text="СРЕДНЯЯ (5с)",
                 font=("Segoe UI", 9, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_MAIN).grid(row=0, column=1, pady=(0, 8))

        self.lbl_avg = tk.Label(stats_frame, text="0.00%",
                                font=("Segoe UI", 20, "bold"),
                                fg=TEXT_PRIMARY, bg=BG_CARD,
                                width=12, anchor=tk.W,
                                highlightbackground=BORDER_COLOR,
                                highlightthickness=1)
        self.lbl_avg.grid(row=1, column=1, padx=5, sticky=tk.EW)

        tk.Label(stats_frame, text="МИНИМАЛЬНАЯ",
                 font=("Segoe UI", 9, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_MAIN).grid(row=0, column=2, pady=(0, 8))

        self.lbl_min = tk.Label(stats_frame, text="0.00%",
                                font=("Segoe UI", 20, "bold"),
                                fg=TEXT_PRIMARY, bg=BG_CARD,
                                width=12, anchor=tk.W,
                                highlightbackground=BORDER_COLOR,
                                highlightthickness=1)
        self.lbl_min.grid(row=1, column=2, padx=(5, 0), sticky=tk.EW)

        stats_frame.grid_columnconfigure(0, weight=1)
        stats_frame.grid_columnconfigure(1, weight=1)
        stats_frame.grid_columnconfigure(2, weight=1)


    def on_model_change(self, event):
        if self.running:
            self.stop_detection()

    def toggle_detection(self):
        if not self.running:
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        if not self.models:
            return
        self.running = True
        self.btn_canvas.itemconfig(self.btn_bg, fill=ACCENT_RED)
        self.btn_canvas.itemconfig(self.btn_text, text="СТОП")
        selected_model = self.model_var.get()
        self.thread = threading.Thread(target=self.audio_worker,
                                       args=(selected_model,), daemon=True)
        self.thread.start()

    def stop_detection(self):
        self.running = False
        self.btn_canvas.itemconfig(self.btn_bg, fill=ACCENT_GREEN)
        self.btn_canvas.itemconfig(self.btn_text, text="СТАРТ")
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        self.set_indicator_state(False)
        self.status_label.config(text="ОЖИДАНИЕ", fg=TEXT_SECONDARY)

    def set_indicator_state(self, is_drone):
        if is_drone:
            self.indicator_canvas.itemconfig(self.indicator, fill=ACCENT_RED, outline="#ff6666")
            self.indicator_canvas.itemconfig(self.indicator_inner, fill="#cc2020")
            for i, layer in enumerate(self.glow_layers):
                self.indicator_canvas.itemconfig(layer, fill=self.glow_colors_on[i])
        else:
            self.indicator_canvas.itemconfig(self.indicator, fill="#3a3a3a", outline="#4a4a4a")
            self.indicator_canvas.itemconfig(self.indicator_inner, fill="#2a2a2a")
            for i, layer in enumerate(self.glow_layers):
                self.indicator_canvas.itemconfig(layer, fill=self.glow_colors_off[i])

    def audio_worker(self, model_path):
        try:
            model = YOLO(model_path)
            sample_rate = 16000
            duration = 0.5
            frames_to_read = int(duration * sample_rate)
            list_size = 5
            median_list = deque(maxlen=list_size)

            with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
                while self.running:
                    audio, overflowed = stream.read(frames_to_read)
                    audio = audio.flatten()
                    img_array = create_spectrogram(audio, sample_rate)
                    results = model(img_array, save=False, verbose=False, rect=True)[0]
                    q = float(results.probs.data[0])
                    median_list.append(q)
                    aq = sum(median_list) / list_size
                    mq = min(median_list)
                    is_drone = (aq > 0.6) and (mq > 0.2)
                    self.msg_queue.put((q, aq, mq, is_drone))
        except Exception as e:
            self.msg_queue.put(("error", str(e), None, None))

    def update_gui(self):
        try:
            while True:
                q, aq, mq, is_drone = self.msg_queue.get_nowait()
                if q == "error":
                    self.status_label.config(text=f"ОШИБКА: {aq}", fg=ACCENT_RED)
                    self.stop_detection()
                    break
                self.lbl_current.config(text=f"{q*100:.2f}%")
                self.lbl_avg.config(text=f"{aq*100:.2f}%")
                self.lbl_min.config(text=f"{mq*100:.2f}%")
                self.set_indicator_state(is_drone)
                if is_drone:
                    self.status_label.config(text="ВНИМАНИЕ: ДРОН!", fg=ACCENT_RED)
                else:
                    self.status_label.config(text="МОНИТОРИНГ", fg=ACCENT_GREEN)
        except queue.Empty:
            pass
        self.root.after(100, self.update_gui)

    def on_closing(self):
        self.stop_detection()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = DroneDetectorApp(root)
    root.mainloop()
