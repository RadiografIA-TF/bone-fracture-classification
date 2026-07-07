from torchvision import transforms

def get_train_transform(out_size = (384, 384), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                        rand_h_flip=0.5, rand_v_flip=0.5, rot_deg=30, scale=(0.8, 1.0)):
    return transforms.Compose([
        transforms.RandomResizedCrop(out_size[0], scale=scale),
        transforms.RandomHorizontalFlip(p=rand_h_flip),
        transforms.RandomVerticalFlip(p=rand_v_flip),
        transforms.RandomRotation(degrees=rot_deg),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def get_val_transform(output_size=(384, 384), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Devuelve las transformaciones de imagen para el conjunto de datos.
    """
    return transforms.Compose([
        transforms.Resize(output_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])