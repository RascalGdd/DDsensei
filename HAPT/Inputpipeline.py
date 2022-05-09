import os.path
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


root_path = "D:\HAPTDataSet\RawData/"

datadict = {}


file = pd.read_csv("D:\HAPTDataSet\RawData\labels.txt",header=None)

idx = 1
for row in file.values:
    sep = row[0].split()
    exp = sep[0]
    user = sep[1]
    category = int(sep[2])
    start = int(sep[3])
    end = int(sep[4])
    datadict.update({idx:[exp,user,start,end,category]})
    idx += 1

length = len(file.values)


matrix2 = []
matrix3 = []
kontroll = 0
for i in range(length):
    kontroll += 1


    expstr = datadict[i+1][0]
    userstr = datadict[i+1][1]
    cate = datadict[i+1][4]
    starttime = datadict[i+1][2] - 1
    endtime = datadict[i+1][3] - 1
    if int(expstr) < 10:
        expstr = "0" + expstr
    if int(userstr) < 10:
        userstr = "0" + userstr

    file2 = pd.read_csv(os.path.join(root_path,"acc_exp"+expstr+"_user"+userstr+".txt"), header=None)
    file3 = pd.read_csv(os.path.join(root_path,"gyro_exp"+expstr+"_user"+userstr+".txt"), header=None)
    length2 = int((endtime - starttime)/125) -1
    # print(file2.values[0][0].split())
    for j in range(length2):
        matrix1 = []
        for m in range(250):
            sep2 = file2.values[starttime+j*125+m][0].split()
            sep3 = file3.values[starttime+j*125+m][0].split()
            for k in range(3):
                sep2[k] = float(sep2[k])
                sep3[k] = float(sep3[k])
            newsep = sep2+sep3
            matrix1.append(newsep)
        matrix2.append(matrix1)
        matrix3.append(cate)


    if (kontroll ==10):
        break

# matrix2 = np.asarray(matrix2)
# matrix3 = np.asarray(matrix3)

setdata = torch.Tensor(matrix2)
setlabel = torch.Tensor(matrix3)
# print(setlabel.shape)

class MyData(Dataset):

    # def __init__(self,root_path):
    #     self.root_path = os.path.join(root_path)
    #     self.datadict = {}
    #     self.file = pd.read_csv(os.path.join(self.root_path,"labels.txt"), header=None)

    def __getitem__(self, idx):
        data = setdata[idx]
        label = setlabel[idx]
        return data, label

    def __len__(self):
        return len(matrix3)

dataset = MyData()
print(len(dataset))


train_set, test_set = torch.utils.data.random_split(dataset,[int(0.7*len(dataset)),len(dataset)-int(0.7*len(dataset))])




train_loader = DataLoader(train_set,batch_size=3,shuffle=True)
test_loader = DataLoader(test_set,batch_size=3,shuffle=True)








