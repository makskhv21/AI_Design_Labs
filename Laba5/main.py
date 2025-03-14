import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import requests
import os
import json
import shutil
import random
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pexels API ключ
PEXELS_API_KEY = "your_KEY_API"
headers = {"Authorization": PEXELS_API_KEY}

dataset_path = "./pexels_dogs_dataset"
specific_breed_folder = "./specific_breed_images"

# Створення власного датасету з Pexels
class PexelsDataset(Dataset):
    def __init__(self, queries, num_images_per_class, save_dir, transform=None):
        self.transform = transform
        self.save_dir = save_dir
        self.images = []
        self.labels = []
        self.class_names = queries
        
        os.makedirs(save_dir, exist_ok=True)
        self._download_images(queries, num_images_per_class)

    def _download_images(self, queries, num_images):
        url = "https://api.pexels.com/v1/search"
        per_page = min(num_images, 100)
        global_image_counter = 1

        for q in queries:
            downloaded = 0
            page = 1
            while downloaded < num_images:
                params = {"query": f"{q} dog", "per_page": per_page, "page": page}
                response = requests.get(url, headers=headers, params=params)
                data = response.json()

                if "photos" not in data or not data["photos"]:
                    break

                for photo in data["photos"]:
                    img_url = photo["src"]["medium"]
                    img_data = requests.get(img_url).content
                    img_name = f"image{global_image_counter}.jpg"
                    img_path = os.path.join(self.save_dir, img_name)
                    
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    
                    self.images.append(img_path)
                    self.labels.append(self.class_names.index(q))
                    downloaded += 1
                    global_image_counter += 1
                    if downloaded >= num_images:
                        break
                page += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Трансформації з аугментацією для покращення узагальнення
transform = transforms.Compose([
    transforms.RandomResizedCrop((299, 299), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Тестові трансформації (без аугментації)
test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Список порід
queries = ["labrador", "german shepherd", "bulldog", "poodle", "beagle"]
num_images_per_class = 100  # Збільшено для кращого навчання

# Завантаження датасету
dataset = PexelsDataset(queries, num_images_per_class, dataset_path, transform=transform)
print(f"Кількість зображень у датасеті: {len(dataset)}")

# Розділення на тренувальну та валідаційну вибірки
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = test_transform  # Використовуємо тест трансформацію для валідації

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Класи
class_names = dataset.class_names
num_classes = len(class_names)
print(f"Класи: {class_names}")

# Ініціалізація моделі
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
# Заморожуємо всі шари, крім останнього
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Оптимізатор лише для останнього шару
optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Навчання з валідацією
num_epochs = 50
for epoch in range(num_epochs):
    # Тренування
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # Валідація
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    print(f"Епоха {epoch + 1}: Тренувальна втрата: {train_loss:.4f}, Точність: {train_acc:.4f} | "
          f"Валідаційна втрата: {val_loss:.4f}, Точність: {val_acc:.4f}")

# Збереження моделі
model_path = "./inception_v3_dogs.pth"
torch.save(model.state_dict(), model_path)
with open("./class_names.json", "w") as f:
    json.dump(class_names, f)

# Аналіз, збереження та відображення топ-5 зображень
def analyze_and_save_breed(breed_name):
    if breed_name not in class_names:
        print(f"Порода {breed_name} не знайдена в датасеті.")
        return
    
    model.eval()
    breed_count = 0
    breed_images = []
    os.makedirs(specific_breed_folder, exist_ok=True)
    
    with torch.no_grad():
        for img_path in dataset.images:
            image = Image.open(img_path).convert("RGB")
            image_tensor = test_transform(image).unsqueeze(0).to(device)
            outputs = model(image_tensor)
            predicted = torch.argmax(outputs, 1).item()
            
            if class_names[predicted] == breed_name:
                breed_count += 1
                breed_images.append(img_path)
                new_name = f"image_{breed_count}.jpg"
                shutil.copy(img_path, os.path.join(specific_breed_folder, new_name))
    
    print(f"Знайдено {breed_count} собак породи {breed_name}")
    print(f"Зображення збережено в {specific_breed_folder}")

    # Відображення топ-5 випадкових зображень
    if breed_count > 0:
        num_to_show = min(5, breed_count)
        random_images = random.sample(breed_images, num_to_show)
        
        plt.figure(figsize=(15, 5))
        for i, img_path in enumerate(random_images):
            image = Image.open(img_path).convert("RGB")
            plt.subplot(1, num_to_show, i + 1)
            plt.imshow(image)
            plt.title(f"{breed_name} #{i+1}")
            plt.axis("off")
        plt.show()
    else:
        print(f"Немає зображень породи {breed_name} для відображення.")

# Інтерактивний вибір породи
breed_to_find = input("Введіть породу собаки для пошуку (наприклад, 'labrador'): ").lower()
analyze_and_save_breed(breed_to_find)
