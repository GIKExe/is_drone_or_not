import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os

# ==========================================
# 1. Настройки и гиперпараметры
# ==========================================
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DATA_DIR = 'out'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Используется устройство: {DEVICE}")

# ==========================================
# 2. Подготовка данных
# ==========================================
transform = transforms.Compose([
	transforms.Grayscale(num_output_channels=1), 
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.5], std=[0.5]) 
])

train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'val'), transform=transform)

# Для Windows num_workers=0 безопаснее
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Классы: {train_dataset.classes}")
print(f"Размер train: {len(train_dataset)}, val: {len(val_dataset)}")

# ==========================================
# 3. Архитектура модели
# ==========================================
class DroneSpectrogramCNN(nn.Module):
	def __init__(self):
		super(DroneSpectrogramCNN, self).__init__()
		
		self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
		self.pool1 = nn.MaxPool2d(2, 2) 
		
		self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
		self.pool2 = nn.MaxPool2d(2, 2) 
		
		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
		self.pool3 = nn.MaxPool2d(2, 2) 
		
		self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
		self.pool4 = nn.MaxPool2d(2, 2) 
		
		self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
		
		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(128, 64)
		self.dropout = nn.Dropout(0.5)
		self.fc2 = nn.Linear(64, 1)

	def forward(self, x):
		x = self.pool1(torch.relu(self.conv1(x)))
		x = self.pool2(torch.relu(self.conv2(x)))
		x = self.pool3(torch.relu(self.conv3(x)))
		x = self.pool4(torch.relu(self.conv4(x)))
		
		x = self.adaptive_pool(x)
		x = self.flatten(x)
		
		x = self.dropout(torch.relu(self.fc1(x)))
		x = self.fc2(x)
		return x

model = DroneSpectrogramCNN().to(DEVICE)

# ==========================================
# 4. Loss и Optimizer
# ==========================================
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ==========================================
# 5. Цикл обучения с прогресс-барами
# ==========================================
def train_model():
	best_val_acc = 0.0

	for epoch in range(EPOCHS):
		model.train()
		running_loss = 0.0
		correct_train = 0
		total_train = 0

		# --- TRAINING ---
		# Оборачиваем train_loader в tqdm
		train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
		
		for inputs, labels in train_loop:
			inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
			labels = labels.unsqueeze(1).float() 
			
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# Считаем метрики
			running_loss += loss.item() * inputs.size(0)
			preds = (torch.sigmoid(outputs) >= 0.5).float()
			correct_train += (preds == labels).sum().item()
			total_train += labels.size(0)

			# <--- Обновляем статус в прогресс-баре в реальном времени --->
			train_loop.set_postfix({
				'loss': f"{running_loss / total_train:.4f}",
				'acc': f"{correct_train / total_train:.4f}"
			})

		epoch_loss = running_loss / total_train
		train_acc = correct_train / total_train

		# --- VALIDATION ---
		model.eval()
		correct_val = 0
		total_val = 0
		val_loss = 0.0

		# Оборачиваем val_loader в tqdm
		val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  ", leave=False)
		
		with torch.no_grad():
			for inputs, labels in val_loop:
				inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
				labels = labels.unsqueeze(1).float()
				
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				val_loss += loss.item() * inputs.size(0)
				
				preds = (torch.sigmoid(outputs) >= 0.5).float()
				correct_val += (preds == labels).sum().item()
				total_val += labels.size(0)

				# <--- Обновляем статус валидации --->
				val_loop.set_postfix({
					'loss': f"{val_loss / total_val:.4f}",
					'acc': f"{correct_val / total_val:.4f}"
				})

		val_epoch_loss = val_loss / total_val
		val_acc = correct_val / total_val
		
		scheduler.step(val_epoch_loss)

		# Выводим итоговые метрики эпохи (leave=True, чтобы осталась в истории консоли)
		print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
			  f"Train Loss: {epoch_loss:.4f} Acc: {train_acc:.4f} | "
			  f"Val Loss: {val_epoch_loss:.4f} Acc: {val_acc:.4f}", end="")

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			torch.save(model.state_dict(), 'best_drone_model.pth')
			print(" | 💾 Saved best model!")
		else:
			print() # Перенос строки, если модель не сохранилась

	print("\nОбучение завершено!")

# ==========================================
# 6. Точка входа (Обязательно для Windows!)
# ==========================================
if __name__ == '__main__':
	try:
		train_model()
	except KeyboardInterrupt:
		pass
	except:
		raise