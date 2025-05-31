
from torchvision import transforms

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

start_transform = transforms.Resize((224, 224))
end_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
base_transform = transforms.Compose([
    start_transform,
    end_transform
])

augment_transform = transforms.Compose([
    start_transform,
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    end_transform
])