from torchvision import transforms

#Emotion dictionary
emotion = {
    "Angry" : 0,
    "Disgust" : 1,
    "Fear" : 2,
    "Happy" : 3,
    "Neutral" : 4,
    "Sad" : 6,
    "Surprise" : 7
}

#Image -> tensor
image2tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, ], [0.5, ])
])

d_model = 9
n_classes = 7
img_size = (48, 48)
patch_size = (16, 16)
n_channels = 1
n_heads = 3
n_layers = 3
batch_size = 128
epochs = 5
alpha = 0.005
