from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.CenterCrop((195, 195)),  # Remove the blue border
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def fetch_data():
    path = './data/asl_big_set/ASL_Alphabet_Dataset'
    dset = datasets.ImageFolder(root=path, transform=transform)
    return dset
