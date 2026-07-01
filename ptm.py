import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# ==========================================
# 1. Настройки и гиперпараметры
# ==========================================
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DATA_DIR = 'out'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Используется устройство: {DEVICE}")

# ==========================================
# 2. Подготовка данных
# ==========================================
# Так как картинки ЧБ, но сохранены в RGB, конвертируем их в 1 канал (Grayscale)
# Это ускорит обучение и снизит потребление памяти.
transform = transforms.Compose([
	transforms.Grayscale(num_output_channels=1), 
	transforms.ToTensor(),
	# Нормализация (значения могут понадобиться позже, пока оставим 0.5)
	transforms.Normalize(mean=[0.5], std=[0.5]) 
])

# ImageFolder автоматически создаст классы из названий папок (drone, no_drone)
train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'val'), transform=transform)

# Для Windows лучше использовать num_workers=0, чтобы избежать ошибок с многопроцессностью
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Классы: {train_dataset.classes}")
print(f"Размер train: {len(train_dataset)}, val: {len(val_dataset)}")

# ==========================================
# 3. Архитектура модели (Custom Lightweight CNN)
# ==========================================
class DroneSpectrogramCNN(nn.Module):
	def __init__(self):
		super(DroneSpectrogramCNN, self).__init__()
		
		# Вход: [Batch, 1, 128, 16] (C, H, W)
		
		# Блок 1
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
		self.relu1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 128->64, 16->8
		
		# Блок 2
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
		self.relu2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 64->32, 8->4
		
		# Блок 3
		self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
		self.relu3 = nn.ReLU()
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 32->16, 4->2
		
		# Блок 4
		self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
		self.relu4 = nn.ReLU()
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 16->8, 2->1
		
		# Адаптивный пулинг, чтобы гарантированно получить 1x1 на выходе 
		# (защита от ошибок, если размер картинки вдруг изменится)
		self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
		
		# Полносвязные слои
		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(128 * 1 * 1, 64)
		self.dropout = nn.Dropout(0.5) # Защита от переобучения
		self.fc2 = nn.Linear(64, 1)    # 1 нейрон для бинарной классификации

	def forward(self, x):
		x = self.pool1(self.relu1(self.conv1(x)))
		x = self.pool2(self.relu2(self.conv2(x)))
		x = self.pool3(self.relu3(self.conv3(x)))
		x = self.pool4(self.relu4(self.conv4(x)))
		
		x = self.adaptive_pool(x)
		x = self.flatten(x)
		
		x = self.dropout(torch.relu(self.fc1(x)))
		x = self.fc2(x)
		return x

model = DroneSpectrogramCNN().to(DEVICE)
print(model)

# ==========================================
# 4. Loss и Optimizer
# ==========================================
# BCEWithLogitsLoss объединяет Sigmoid и BCELoss, что численно стабильнее
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Опционально: снижение learning rate, если модель застрянет
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ==========================================
# 5. Цикл обучения
# ==========================================
def train_model():
	best_val_acc = 0.0

	for epoch in range(EPOCHS):
		model.train()
		running_loss = 0.0
		correct_train = 0
		total_train = 0

		# --- TRAINING ---
		for inputs, labels in train_loader:
			inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
			
			# Приводим метки к float для BCEWithLogitsLoss
			labels = labels.unsqueeze(1).float() 
			
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item() * inputs.size(0)
			
			# Считаем точность
			preds = (torch.sigmoid(outputs) >= 0.5).float()
			correct_train += (preds == labels).sum().item()
			total_train += labels.size(0)

		epoch_loss = running_loss / total_train
		train_acc = correct_train / total_train

		# --- VALIDATION ---
		model.eval()
		correct_val = 0
		total_val = 0
		val_loss = 0.0

		with torch.no_grad():
			for inputs, labels in val_loader:
				inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
				labels = labels.unsqueeze(1).float()
				
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				val_loss += loss.item() * inputs.size(0)
				
				preds = (torch.sigmoid(outputs) >= 0.5).float()
				correct_val += (preds == labels).sum().item()
				total_val += labels.size(0)

		val_epoch_loss = val_loss / total_val
		val_acc = correct_val / total_val
		
		# Обновляем learning rate
		scheduler.step(val_epoch_loss)

		print(f"Epoch [{epoch+1}/{EPOCHS}] | "
			  f"Train Loss: {epoch_loss:.4f} Acc: {train_acc:.4f} | "
			  f"Val Loss: {val_epoch_loss:.4f} Acc: {val_acc:.4f}")

		# Сохраняем лучшую модель
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			torch.save(model.state_dict(), 'best_drone_model.pth')
			print(f"-> Сохранена лучшая модель с Val Acc: {best_val_acc:.4f}")

	print("Обучение завершено!")

# ==========================================
# 6. Точка входа (Обязательно для Windows!)
# ==========================================
if __name__ == '__main__':
	train_model()