import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler 
import torch,re,os,time
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import scipy.io as scio
import torch.nn as nn
import gc,math
import datetime
from scipy import stats
torch.__version__
import sys
import time


torch.manual_seed(128)
np.random.seed(1024)

def BG_scale(data):
    data_len = len(data)
    for i in range(data_len):
        if data[i] > 400:
            data[i] = 400
        elif data[i] < 40:
            data[i] = 40
    return data


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def RMSE_MARD(pred_values,ref_values):    
    data_length = len(ref_values)
    total = 0
    for i in range (data_length):
        temp = (ref_values[i] - pred_values[i]) * (ref_values[i] - pred_values[i])
        total = total + temp

    smse_value = math.sqrt(total / data_length)
    print('RMSE:{:.4f}'.format(smse_value))
    
    total = 0
    for i in range(data_length):
        temp = abs((ref_values[i] - pred_values[i]) / ref_values[i])
        total = total + temp
    mard_value = total / data_length
    print('MARD:{:.4f}'.format(mard_value))
        
    return smse_value,mard_value

def CONCOR(pred_values,ref_values):    
    data_length = len(ref_values)
    M1 = 0
    M2 = 0
    M3 = 0
    for i in range (data_length):
        v1 = abs(ref_values[i] - pred_values[i])
        v2 = v1 / ref_values[i]
        if v1 <= 15 or v2 <= 0.15:
           M1 = M1 + 1
        if v1 <= 20 or v2 <= 0.2:
            M2 = M2 + 1
        if v1 <= 30 or v2 <= 0.3:
            M3 = M3 + 1
    per_M1 = M1 / data_length
    per_M2 = M2 / data_length
    per_M3 = M3 / data_length
    print('per_M1:{:.4f}, per_M2:{:.4f}, per_M3:{:.4f}'.format(per_M1, per_M2, per_M3)) 
    per_M = np.matrix([per_M1, per_M2, per_M3]) 
    return per_M


def clarke_error_grid(pred_values,ref_values):

    assert (len(ref_values) == len(pred_values)), "Unequal number of values (reference : {}) (prediction : {}).".format(len(ref_values), len(pred_values))
    if max(ref_values) > 400 or max(pred_values) > 400:
        print ("Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal physiological range of glucose (<400 mg/dl).".format(max(ref_values), max(pred_values)))
    if min(ref_values) < 0 or min(pred_values) < 0:
        print ("Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.".format(min(ref_values),  min(pred_values)))

#    plt.figure()
#
#    plt.scatter(ref_values, pred_values, marker='o', color='black', s=8)
#    plt.xlabel("Reference Concentration (mg/dl)")
#    plt.ylabel("Prediction Concentration (mg/dl)")
#    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
#    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
#    plt.gca().set_facecolor('white')
#
#    #Set axes lengths
#    plt.gca().set_xlim([0, 400])
#    plt.gca().set_ylim([0, 400])
#    plt.gca().set_aspect((400)/(400))
#
#    #Plot zone lines
#    plt.plot([0,400], [0,400], ':', c='black')                      #Theoretical 45 regression line
#    plt.plot([0, 175/3], [70, 70], '-', c='black')
#    #plt.plot([175/3, 320], [70, 400], '-', c='black')
#    plt.plot([175/3, 400/1.2], [70, 400], '-', c='black')           #Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
#    plt.plot([70, 70], [84, 400],'-', c='black')
#    plt.plot([0, 70], [180, 180], '-', c='black')
#    plt.plot([70, 290],[180, 400],'-', c='black')
#    # plt.plot([70, 70], [0, 175/3], '-', c='black')
#    plt.plot([70, 70], [0, 56], '-', c='black')                     #Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
#    # plt.plot([70, 400],[175/3, 320],'-', c='black')
#    plt.plot([70, 400], [56, 320],'-', c='black')
#    plt.plot([180, 180], [0, 70], '-', c='black')
#    plt.plot([180, 400], [70, 70], '-', c='black')
#    plt.plot([240, 240], [70, 180],'-', c='black')
#    plt.plot([240, 400], [180, 180], '-', c='black')
#    plt.plot([130, 180], [0, 70], '-', c='black')
#
#    #Add zone titles
#    plt.text(30, 15, "A", fontsize=15)
#    plt.text(370, 260, "B", fontsize=15)
#    plt.text(280, 370, "B", fontsize=15)
#    plt.text(160, 370, "C", fontsize=15)
#    plt.text(160, 15, "C", fontsize=15)
#    plt.text(30, 140, "D", fontsize=15)
#    plt.text(370, 120, "D", fontsize=15)
#    plt.text(30, 370, "E", fontsize=15)
#    plt.text(370, 15, "E", fontsize=15)

    #Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        if (ref_values[i] <= 70 and pred_values[i] <= 70) or (pred_values[i] <= 1.2*ref_values[i] and pred_values[i] >= 0.8*ref_values[i]):
            zone[0] += 1    #Zone A

        elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (ref_values[i] <= 70 and pred_values[i] >= 180):
            zone[4] += 1    #Zone E

        elif ((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110) or ((ref_values[i] >= 130 and ref_values[i] <= 180) and (pred_values[i] <= (7/5)*ref_values[i] - 182)):
            zone[2] += 1    #Zone C
        elif (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180)) or (ref_values[i] <= 175/3 and pred_values[i] <= 180 and pred_values[i] >= 70) or ((ref_values[i] >= 175/3 and ref_values[i] <= 70) and pred_values[i] >= (6/5)*ref_values[i]):
            zone[3] += 1    #Zone D
        else:
            zone[1] += 1    #Zone B

    A_score = zone[0]/(zone[0]+zone[1]+zone[2]+zone[3]+zone[4])
    B_score = zone[1]/(zone[0]+zone[1]+zone[2]+zone[3]+zone[4])
    C_score = zone[2]/(zone[0]+zone[1]+zone[2]+zone[3]+zone[4])
    D_score = zone[3]/(zone[0]+zone[1]+zone[2]+zone[3]+zone[4])
    E_score = zone[4]/(zone[0]+zone[1]+zone[2]+zone[3]+zone[4])
    score = np.matrix([A_score, B_score, C_score, D_score, E_score]) 
    print('Clarke:  A:{:.4f}, B:{:.4f}, C:{:.4f}, D:{:.4f}, E:{:.4f}'.format(A_score, B_score, C_score, D_score, E_score))
    return score


class ConvBlock(nn.Module):
    def __init__(self, in_channel, f, filters, p1, p2, p3, p4):
        super(ConvBlock,self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel,F1,kernel_size=(1,1),stride=(1,2), padding=p1, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1,F2,f,stride=1, padding=p2, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2,F3,kernel_size=(1,1),stride=1, padding=p3, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.shortcut_1 = nn.Conv2d(in_channel, F3, kernel_size=(1,1), stride=(1,2), padding=p4, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(True)
        
    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X     
    
class IndentityBlock(nn.Module):
    def __init__(self, in_channel, f, filters, p1, p2, p3):
        super(IndentityBlock,self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel,F1,kernel_size=(1,1),stride=1, padding=p1, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1,F2,f,stride=1, padding=p2, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2,F3,kernel_size=(1,1),stride=1, padding=p3, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)
        
    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X
    
class ResModel(nn.Module):
    def __init__(self):
        super(ResModel,self).__init__()
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels =1,out_channels =4,kernel_size=(1,6),stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.MaxPool2d((1,2),padding=0),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(4, f=(1,6), filters=[4, 4, 16],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(16, (1,5), [4, 4, 16],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(16, (1,5), [4, 4, 16],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(16, f=(1,6), filters=[8, 8, 32],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(32, (1,5), [8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(32, (1,5), [8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(32, (1,5), [8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage4 = nn.Sequential(
            ConvBlock(32, f=(1,6), filters=[16, 16, 64],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage5 = nn.Sequential(
            ConvBlock(64, f=(1,6), filters=[32, 32, 128],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(128, (1,5), [32, 32, 128],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(128, (1,5), [32, 32, 128],p1=(0,1),p2=(0,1),p3=0),
        )
        
        self.stage6 = nn.Sequential(
            nn.Conv2d(in_channels =1,out_channels =4,kernel_size=(1,6),stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.MaxPool2d((1,2),padding=0),
        )
        self.stage7 = nn.Sequential(
            ConvBlock(4, f=(1,6), filters=[4, 4, 16],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(16, (1,5), [4, 4, 16],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(16, (1,5), [4, 4, 16],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage8 = nn.Sequential(
            ConvBlock(16, f=(1,6), filters=[8, 8, 32],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(32, (1,5), [8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(32, (1,5), [8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(32, (1,5), [8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage9 = nn.Sequential(
            ConvBlock(32, f=(1,6), filters=[16, 16, 64],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage10 = nn.Sequential(
            ConvBlock(64, f=(1,6), filters=[32, 32, 128],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(128, (1,5), [32, 32, 128],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(128, (1,5), [32, 32, 128],p1=(0,1),p2=(0,1),p3=0),
        )   

        self.stage11 = nn.Sequential(
            nn.Conv2d(in_channels =1,out_channels =4,kernel_size=(1,6),stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.MaxPool2d((1,2),padding=0),
        )
        self.stage12 = nn.Sequential(
            ConvBlock(4, f=(1,6), filters=[4, 4, 16],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(16, (1,5), [4, 4, 16],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(16, (1,5), [4, 4, 16],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage13 = nn.Sequential(
            ConvBlock(16, f=(1,6), filters=[8, 8, 32],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(32, (1,5), [8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(32, (1,5), [8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(32, (1,5), [8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage14 = nn.Sequential(
            ConvBlock(32, f=(1,6), filters=[16, 16, 64],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(64, (1,5), [16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage15 = nn.Sequential(
            ConvBlock(64, f=(1,6), filters=[32, 32, 128],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(128, (1,5), [32, 32, 128],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(128, (1,5), [32, 32, 128],p1=(0,1),p2=(0,1),p3=0),
        )   
        
        self.bn = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d((1,2),padding=0)
        
        self.fc1 = nn.Linear(30720,4000)
        self.fc2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(4000,400)
        self.fc4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(400,40)
        self.fc6 = nn.Linear(40,1)
        
       
        
    
    def forward(self, X, Y, Z):
        out1 = self.stage1(X)
        out1 = self.stage2(out1)
        out1 = self.stage3(out1)
        out1 = self.stage4(out1)
        out1 = self.stage5(out1)
        out1 = self.pool(out1)   
        out1 = self.bn(out1)
        out1 = out1.view(out1.size(0),-1)
        
        out2 = self.stage6(Y)
        out2 = self.stage7(out2)
        out2 = self.stage8(out2)
        out2 = self.stage9(out2)
        out2 = self.stage10(out2)
        out2 = self.pool(out2)   
        out2 = self.bn(out2)
        out2 = out2.view(out2.size(0),-1)
        
        out3 = self.stage11(Z)
        out3 = self.stage12(out3)
        out3 = self.stage13(out3)
        out3 = self.stage14(out3)
        out3 = self.stage15(out3)
        out3 = self.pool(out3)   
        out3 = self.bn(out3)
        out3 = out3.view(out3.size(0),-1)        
        
        out = torch.cat((out1,out2,out3),dim = 1)          
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        
        return out
    
# resmodel = ResModel()
# resmodel(torch.rand(10,1,20,200),torch.rand(10,1,20,200))


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts



path1 = '/data0/ljz/ECG_BG/ECG signal MAT'
path2 = '/data0/ljz/ECG_BG/ECG BG MAT'


data_dir_s = sorted(os.listdir(path1),key=numericalSort) 
Label_data = np.loadtxt('/data0/ljz/ECG_BG/new label.txt')
label_id = Label_data[:,0]
wave_id = Label_data[:,1]



samp_num = 20
train_id_list = []
test_id_list = []
test_num_list = []
train_num_list = []
train_glu_list = []
test_glu_list = []

time_start = 6
time_end = 7
test_n = 0
train_n = 0

for Path in data_dir_s:        
    tp = scio.loadmat(path2 + '/' + Path + '.mat')
    temp_glucose = tp['BG_value']  
    temp_glucose = temp_glucose.reshape(-1)
    temp_glucose = temp_glucose * 18    
    temp_BG = BG_scale(temp_glucose)          
    ECG_dir_list = sorted(os.listdir(path1 + '/' + Path),key=numericalSort) 
    ECG_len = len(ECG_dir_list)
    num = np.linspace(0,ECG_len-1,ECG_len)
    N1, N2 = train_test_split(num, test_size = 0.3, shuffle = True, random_state = 8)
    N1_sort = np.sort(N1)
    N2_sort = np.sort(N2)
    for i in range (len(N1_sort)):
        num_id = int(N1_sort[i])
        train_id_list.append(ECG_dir_list[num_id])
        train_num_list.append(train_n)
        train_n = train_n + 1
        train_glu_list.append(temp_BG[num_id])
            
    for i in range (len(N2_sort)):
        num_id = int(N2_sort[i])
        test_id_list.append(ECG_dir_list[num_id])
        test_num_list.append(test_n)
        test_n = test_n + 1
        test_glu_list.append(temp_BG[num_id])
    
train_num = np.array(train_num_list)
test_num = np.array(test_num_list) 
train_glu = np.array(train_glu_list)
test_glu = np.array(test_glu_list)

train_num = torch.from_numpy(train_num)
test_num = torch.from_numpy(test_num)
train_glu = torch.from_numpy(train_glu)
test_glu = torch.from_numpy(test_glu)  


BATCH_SIZE = 128
train_set = torch.utils.data.TensorDataset(train_num, train_glu)
test_set = torch.utils.data.TensorDataset(test_num, test_glu)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True) 
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False) 


fusion = ResModel()

DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(DEVICE)
fusion = fusion.to(DEVICE)
criterion = nn.MSELoss().to(DEVICE)
LEARNING_RATE = 0.0001
optimizer = torch.optim.Adam(fusion.parameters(), lr=LEARNING_RATE)
TOTAL_EPOCHS = 300
losses = []

EPOCH_list = []
train_coef_list =[]
train_p_list = []
test_coef_list =[]
test_p_list = []
RMSE1_list = []
MARD1_list = []
RMSE2_list = []
MARD2_list = []
train_ClarkeA_list = []
train_ClarkeAB_list = []
train_ClarkeC_list = []
train_ClarkeD_list = []
train_ClarkeE_list = []
train_concor15_list = []
train_concor20_list = []
train_concor30_list = []
test_ClarkeA_list = []
test_ClarkeAB_list = []
test_ClarkeC_list = []
test_ClarkeD_list = []
test_ClarkeE_list = []
test_concor15_list = []
test_concor20_list = []
test_concor30_list = []

for epoch in range(TOTAL_EPOCHS):
    train_pred_list = []
    train_true_list = []
    test_pred_list = []
    test_true_list = []
       
    for y, (train_images, tr_labels) in enumerate(train_loader):

        tr_labels = tr_labels.to(DEVICE).float()
        train_images = train_images.numpy()
        data_len = len(train_images)
        train_ECG1 = np.zeros((data_len*samp_num, 200))
        train_ECG2 = np.zeros((data_len*samp_num, 200))
        train_ECG3 = np.zeros((data_len*samp_num, 200))
        
        for n in range(data_len):
            train_label_temp = int(train_images[n])
            train_id = train_id_list[train_label_temp]
            file_name = train_id[0:7]
            # temp_name = int(file_name)
            # for s in range(len(label_id)):
            #     if int(label_id[s]) == temp_name:
            #         wave = int(wave_id[s])
            #         break
            ecg = scio.loadmat(path1 + '/' + file_name + '/' + train_id, variable_names=['ECG_data'])
            train_ECG = ecg['ECG_data'] 
            # if wave == 0:
            #     train_ECG = 1 - train_ECG                     
            Q1 = int(n * samp_num)
            Q2 = Q1 + samp_num            
            train_ECG1[Q1:Q2,:] = train_ECG[0:20,:]
            train_ECG2[Q1:Q2,:] = train_ECG[20:40,:]
            train_ECG3[Q1:Q2,:] = train_ECG[40:60,:]
           
        train_ECG1 = train_ECG1.reshape(-1,20,200)
        train_ECG1 = np.expand_dims(train_ECG1,axis=1)
        train_ECG1 = torch.from_numpy(train_ECG1)
        train_ECG1 = train_ECG1.float().to(DEVICE)
        
        train_ECG2 = train_ECG2.reshape(-1,20,200)
        train_ECG2 = np.expand_dims(train_ECG2,axis=1)        
        train_ECG2 = torch.from_numpy(train_ECG2)
        train_ECG2 = train_ECG2.float().to(DEVICE)
        
        train_ECG3 = train_ECG3.reshape(-1,20,200)
        train_ECG3 = np.expand_dims(train_ECG3,axis=1)        
        train_ECG3 = torch.from_numpy(train_ECG3)        
        train_ECG3 = train_ECG3.float().to(DEVICE)        

        optimizer.zero_grad()
        train_outputs = fusion(train_ECG1,train_ECG2,train_ECG3)
        train_outputs = train_outputs.view(-1)
        train_loss = criterion(train_outputs,tr_labels)
        train_loss.backward()
        optimizer.step()
        # losses.append(train_loss.cpu().data.item())
                
        Train_outputs = train_outputs.cpu()
        Train_outputs = Train_outputs.detach().numpy()
        Tr_labels = tr_labels.cpu()
        Tr_labels = Tr_labels.numpy() 
        Train_outputs = BG_scale(Train_outputs)        
        for t in range(data_len):
            train_pred_list.append(Train_outputs[t])
            train_true_list.append(Tr_labels[t])
        del ecg, train_ECG 
        gc.collect() 
        
    train_pred = np.array(train_pred_list)
    train_true = np.array(train_true_list)
    print('Epoch: %d' %(epoch + 1))
    train_coef, train_p = stats.pearsonr(train_pred, train_true)
    train_result = RMSE_MARD(train_pred, train_true)
    train_concor = CONCOR(train_pred, train_true)
    train_Clarke = clarke_error_grid(train_pred, train_true)   
    
    fusion.eval() 
    for y, (test_images, te_labels) in enumerate(test_loader):
        te_labels = te_labels.float()
        test_images = test_images.numpy()
        data_len = len(test_images)
        test_ECG1 = np.zeros((data_len*samp_num, 200))
        test_ECG2 = np.zeros((data_len*samp_num, 200))
        test_ECG3 = np.zeros((data_len*samp_num, 200))       
        
        for n in range(data_len):
            test_label_temp = int(test_images[n]) 
            test_id = test_id_list[test_label_temp]
            file_name = test_id[0:7]
            # temp_name = int(file_name)
            # for s in range(len(label_id)):
            #     if int(label_id[s]) == temp_name:
            #         wave = int(wave_id[s])
            #         break
            ecg = scio.loadmat(path1 + '/' + file_name + '/' + test_id, variable_names=['ECG_data'])
            test_ECG = ecg['ECG_data'] 
            # if wave == 0:
            #     test_ECG = 1 - test_ECG             
            Q1 = int(n * samp_num)
            Q2 = int((n + 1) * samp_num)            
            test_ECG1[Q1:Q2,:] = test_ECG[0:20,:]
            test_ECG2[Q1:Q2,:] = test_ECG[20:40,:]
            test_ECG3[Q1:Q2,:] = test_ECG[40:60,:]
            
        test_ECG1 = test_ECG1.reshape(-1,20,200)
        test_ECG1 = np.expand_dims(test_ECG1,axis=1)
        test_ECG1 = torch.from_numpy(test_ECG1)
        test_ECG1 = test_ECG1.float().to(DEVICE)
        
        test_ECG2 = test_ECG2.reshape(-1,20,200)
        test_ECG2 = np.expand_dims(test_ECG2,axis=1)        
        test_ECG2 = torch.from_numpy(test_ECG2)
        test_ECG2 = test_ECG2.float().to(DEVICE)
        
        test_ECG3 = test_ECG3.reshape(-1,20,200)
        test_ECG3 = np.expand_dims(test_ECG3,axis=1)        
        test_ECG3 = torch.from_numpy(test_ECG3)        
        test_ECG3 = test_ECG3.float().to(DEVICE)                    


        test_outputs = fusion(test_ECG1,test_ECG2,test_ECG3)
        test_outputs = test_outputs.view(-1)                
        test_outputs = test_outputs.cpu()
        test_outputs = test_outputs.detach().numpy()
        Te_labels = te_labels.numpy() 
        test_outputs = BG_scale(test_outputs)
        for t in range(data_len):                 
            test_pred_list.append(test_outputs[t])
            test_true_list.append(Te_labels[t])
    
    test_pred = np.array(test_pred_list)
    test_true = np.array(test_true_list)
    test_coef, test_p = stats.pearsonr(test_pred, test_true)
    test_result = RMSE_MARD(test_pred, test_true)
    test_concor = CONCOR(test_pred, test_true)
    test_Clarke = clarke_error_grid(test_pred, test_true)
    

    torch.save(fusion, '/data0/ljz/ECG_BG/save/random3' + "/" + "fusion_model_EPOCHS " + str(epoch + 1) + ".pth")
    
    EPOCH_list.append(epoch + 1)
    train_coef_list.append(train_coef)
    train_p_list.append(train_p)
    test_coef_list.append(test_coef)
    test_p_list.append(test_p)    
    RMSE1_list.append(train_result[0])  
    MARD1_list.append(train_result[1])  
    RMSE2_list.append(test_result[0])  
    MARD2_list.append(test_result[1]) 
    train_concor15_list.append(train_concor[0,0])
    train_concor20_list.append(train_concor[0,1])
    train_concor30_list.append(train_concor[0,2])
    test_concor15_list.append(test_concor[0,0])
    test_concor20_list.append(test_concor[0,1])
    test_concor30_list.append(test_concor[0,2])    
    train_ClarkeA_list.append(train_Clarke[0,0])
    train_ClarkeAB_list.append(train_Clarke[0,0] + train_Clarke[0,1])
    train_ClarkeC_list.append(train_Clarke[0,2])
    train_ClarkeD_list.append(train_Clarke[0,3])
    train_ClarkeE_list.append(train_Clarke[0,4])
    test_ClarkeA_list.append(test_Clarke[0,0])
    test_ClarkeAB_list.append(test_Clarke[0,0] + test_Clarke[0,1])
    test_ClarkeC_list.append(test_Clarke[0,2])
    test_ClarkeD_list.append(test_Clarke[0,3])
    test_ClarkeE_list.append(test_Clarke[0,4])    
    

    Train_coef = np.array(train_coef_list)
    Train_p = np.array(train_p_list)
    Test_coef = np.array(test_coef_list)
    Test_p = np.array(test_p_list)    
    EPOCH = np.array(EPOCH_list)    
    RMSE1 = np.array(RMSE1_list)
    MARD1 = np.array(MARD1_list)
    RMSE2 = np.array(RMSE2_list)
    MARD2 = np.array(MARD2_list)
    train_concor15 = np.array(train_concor15_list)
    train_concor20 = np.array(train_concor20_list)
    train_concor30 = np.array(train_concor30_list)
    test_concor15 = np.array(test_concor15_list)
    test_concor20 = np.array(test_concor20_list)
    test_concor30 = np.array(test_concor30_list)    
    train_ClarkeA = np.array(train_ClarkeA_list)
    train_ClarkeAB = np.array(train_ClarkeAB_list)
    train_ClarkeC = np.array(train_ClarkeC_list)
    train_ClarkeD = np.array(train_ClarkeD_list)
    train_ClarkeE = np.array(train_ClarkeE_list)
    test_ClarkeA = np.array(test_ClarkeA_list)
    test_ClarkeAB = np.array(test_ClarkeAB_list)
    test_ClarkeC = np.array(test_ClarkeC_list)
    test_ClarkeD = np.array(test_ClarkeD_list)
    test_ClarkeE = np.array(test_ClarkeE_list)    

    EPOCH = EPOCH.reshape(-1,1) 
    Train_coef = Train_coef.reshape(-1,1)
    Train_p = Train_p.reshape(-1,1)
    Test_coef = Test_coef.reshape(-1,1)
    Test_p = Test_p.reshape(-1,1)    
    RMSE1 = RMSE1.reshape(-1,1)
    MARD1 = MARD1.reshape(-1,1) 
    RMSE2 = RMSE2.reshape(-1,1)
    MARD2 = MARD2.reshape(-1,1)
    train_concor15 = train_concor15.reshape(-1,1)
    train_concor20 = train_concor20.reshape(-1,1)
    train_concor30 = train_concor30.reshape(-1,1)
    test_concor15 = test_concor15.reshape(-1,1)
    test_concor20 = test_concor20.reshape(-1,1)    
    test_concor30 = test_concor30.reshape(-1,1)    
    train_ClarkeA = train_ClarkeA.reshape(-1,1)
    train_ClarkeAB = train_ClarkeAB.reshape(-1,1)
    train_ClarkeC = train_ClarkeC.reshape(-1,1)
    train_ClarkeD = train_ClarkeD.reshape(-1,1)
    train_ClarkeE = train_ClarkeE.reshape(-1,1)
    test_ClarkeA = test_ClarkeA.reshape(-1,1)
    test_ClarkeAB = test_ClarkeAB.reshape(-1,1)
    test_ClarkeC = test_ClarkeC.reshape(-1,1)
    test_ClarkeD = test_ClarkeD.reshape(-1,1)
    test_ClarkeE = test_ClarkeE.reshape(-1,1)
    train_save = np.hstack((EPOCH, Train_coef, Train_p, RMSE1, MARD1, train_concor15, train_concor20, train_concor30,
                            train_ClarkeA, train_ClarkeAB, train_ClarkeC, train_ClarkeD, train_ClarkeE))
    test_save = np.hstack((EPOCH, Test_coef, Test_p, RMSE2, MARD2, test_concor15, test_concor20, test_concor30,
                            test_ClarkeA, test_ClarkeAB, test_ClarkeC, test_ClarkeD, test_ClarkeE))
    np.savetxt('/data0/ljz/ECG_BG/save/random3/train result.csv', train_save, delimiter=',', fmt='%1.5f')
    np.savetxt('/data0/ljz/ECG_BG/save/random3/test result.csv', test_save, delimiter=',', fmt='%1.5f')
