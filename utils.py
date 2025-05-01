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