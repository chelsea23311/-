import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import numpy as np
from torchvision import transforms
import torch.nn as nn
import os
import cv2
import glob
import torch.optim as optim
from torch.utils.data._utils.collate import default_collate
import csv
train_image_paths=[]
train_label_paths=[]
val_image_paths=[]
val_label_paths=[]
test_image_paths=[]
class PointDetectionDataset(Dataset):
    def __init__(self,image_paths,label_paths,transform=None, device=torch.device('cpu')):
        self.image_paths=image_paths
        self.label_paths=label_paths
        self.transform = transform
        self.device = device

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if image is None:
            raise ValueError(f"无法从路径 {image_path} 读取图像，检查文件是否存在及完整性")

        label=[]
        f = open(self.label_paths[index],"r",encoding="UTF-8")
        for line in f.readlines():
            numbers = line.strip().split()
            for index, number in enumerate(numbers):
                if index!=0:
                    label.append(float(number))
        image = transforms.ToTensor()(image).to(self.device)
        if self.transform:
            image = self.transform(image)
        label=torch.tensor(label).to(self.device)
        return image,label

    def __len__(self):
        length = len(self.image_paths)
        return length

class PointDetectionDataset_test(Dataset):
    def __init__(self,image_paths,transform=None, device=torch.device('cpu')):
        self.image_paths=image_paths
        self.transform = transform
        self.device = device
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if image is None:
            raise ValueError(f"无法从路径 {image_path} 读取图像，检查文件是否存在及完整性")

        image = transforms.ToTensor()(image).to(self.device)
        image.cuda()
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)
for jpgfile in glob.glob((r'C:\zeqi\three\train' + '/*.jpg')):
    image_name=os.path.join(r"C:\zeqi\three\train",os.path.basename(jpgfile))
    train_image_paths.append(image_name)
for txtfile in glob.glob((r'C:\zeqi\three\train' + '/*.txt')):
    txt_name=os.path.join(r"C:\zeqi\three\train", os.path.basename(txtfile))
    train_label_paths.append(txt_name)
for jpgfile in glob.glob((r'C:\zeqi\three\val' + '/*.jpg')):
    image_name=os.path.join(r"C:\zeqi\three\val",os.path.basename(jpgfile))
    val_image_paths.append(image_name)
for txtfile in glob.glob((r'C:\zeqi\three\val' + '/*.txt')):
    txt_name=os.path.join(r"C:\zeqi\three\val", os.path.basename(txtfile))
    val_label_paths.append(txt_name)
for jpgfile in glob.glob((r'C:\zeqi\three\test' + '/*.jpg')):
    image_name = os.path.join(r"C:\zeqi\three\test", os.path.basename(jpgfile))
    test_image_paths.append(image_name)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset=PointDetectionDataset(train_image_paths,train_label_paths,device=device)
train_dataloader=DataLoader(train_dataset,batch_size = 50,shuffle=False,drop_last=False)
val_dataset=PointDetectionDataset(val_image_paths,val_label_paths,device=device)
val_dataloader=DataLoader(val_dataset,batch_size = 50,shuffle=False,drop_last=False)
test_dataset=PointDetectionDataset_test(test_image_paths,device=device)
test_dataloader=DataLoader(test_dataset,batch_size = 50,shuffle=False,drop_last=False)
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 125 * 125,256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 18)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model=AlexNet()
model.to(device)
def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)

model.apply(initialize_weights)
criterion=nn.SmoothL1Loss()
optimizer=optim.Adam(model.parameters(), lr=0.001)
num_epochs = 3
for epoch in range(num_epochs):
    running_loss = 0.0
    for index,data in enumerate(train_dataloader):
        images, labels=data

        if isinstance(images, tuple):
            images = images[0]

        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print("loss为", running_loss/len(train_dataloader))
    torch.cuda.empty_cache()
def compute_pck(labels_pred,labels,threshold):
    correct_count=0
    for i in range(len(labels_pred)):
        for j in range(18):
            dists=torch.sqrt(torch.sum((labels_pred[i][j]-labels[i][j])**2))
            correct_count = torch.sum(dists <= threshold)
    pck = correct_count / (len(labels_pred)* 18)
    return pck
for index,data in enumerate(val_dataloader):
        images_val, labels_val=data
        if isinstance(images_val,tuple):
            images_val=images[0]
        outputs_val=model(images_val)
        pck=compute_pck(outputs_val,labels_val,200000)
        print(pck)
with open(r"C:\zeqi\three\result\test.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for index, images_test in enumerate(test_dataloader):
        if isinstance(images_test, tuple):
            images_test = images_test[0]
        outputs_test = model(images_test)
        for i in range(outputs_test.shape[0]):
            row_data = outputs_test[i].tolist()
            row_data.insert(0,9)
            writer.writerow(row_data)