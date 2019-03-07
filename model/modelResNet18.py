
# coding: utf-8

# In[1]:


import makedatalist
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torchvision.models as models

#クラス数変更
num_classes = 5


model_ft = models.resnet18(pretrained=False)
# fc層を置き換える
num_features = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_features, num_classes)
model_ft.conv1 = nn.Conv2d(1, 64, 3, padding=1)
model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
print(model_ft)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.00001,weight_decay=5e-3)


# In[2]:


num_epochs = 40

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
dataXList,dataYList = makedatalist.makeDataList()
numx = np.array(dataXList)
numy = np.array(dataYList)
# 学習用(全体の8割)とテスト用(全体の2割)に分離する
x_train, x_test, y_train, y_test = train_test_split(
numx, numy, test_size = 0.2, train_size = 0.8, shuffle = True)

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
# DataLoaderを作る
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=512)

test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=512)

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0
    
    #train
    net.train()
    #0から
    for i, (images, labels) in enumerate(train_loader):
        images = images.float()
        # バッチ x チャネル x 高さ x 幅
        images = images.view(-1, 1, 28, 28)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        #print(images.size())
        outputs = net(images)
        #print(outputs)
        #print(labels)
        #criterion = nn.CrossEntropyLoss() 上で書いた
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        train_acc += (outputs.max(1)[1] == labels).sum().item()
        loss.backward()
    optimizer.step()
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc = train_acc / len(train_loader.dataset)
    
    #val
    net.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.float()
            # バッチ x チャネル x 高さ x 幅
            images = images.view(-1,1, 28, 28)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()
    avg_val_loss = val_loss / len(test_loader.dataset)
    avg_val_acc = val_acc / len(test_loader.dataset)
    
    print ('Epoch [{}/{}], Loss: {loss:.4f},train_acc: {train_acc:.10f},val_loss: {val_loss:.4f}, val_acc: {val_acc:.10f}' 
                   .format(epoch+1, num_epochs, i+1, loss=avg_train_loss,train_acc=avg_train_acc,val_loss=avg_val_loss, val_acc=avg_val_acc))
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)
torch.save(net.state_dict(), 'quickdrawResnet18.model')


# In[3]:



import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure()
plt.plot(range(num_epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
plt.plot(range(num_epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss')
plt.grid()

plt.figure()
plt.plot(range(num_epochs), train_acc_list, color='blue', linestyle='-', label='train_acc')
plt.plot(range(num_epochs), val_acc_list, color='green', linestyle='--', label='val_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('Training and validation accuracy')
plt.grid()

