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


def clarke_error_zone_detailed(act, pred):
    if (act < 70 and pred < 70) or abs(act - pred) < 0.2 * act:
        return 0
    # Zone E - left upper
    if act <= 70 and pred >= 180:
        return 8
    # Zone E - right lower
    if act >= 180 and pred <= 70:
        return 7
    # Zone D - right
    if act >= 240 and 70 <= pred <= 180:
        return 6
    # Zone D - left
    if act <= 70 <= pred <= 180:
        return 5
    # Zone C - upper
    if 70 <= act <= 290 and pred >= act + 110:
        return 4
    # Zone C - lower
    if 130 <= act <= 180 and pred <= (7/5) * act - 182:
        return 3
    # Zone B - upper
    if act < pred:
        return 2
    # Zone B - lower
    return 1

def parkes_error_zone_detailed(act, pred, diabetes_type):
    def above_line(x_1, y_1, x_2, y_2, strict=False):
        if x_1 == x_2:
            return False

        y_line = ((y_1 - y_2) * act + y_2 * x_1 - y_1 * x_2) / (x_1 - x_2)
        return pred > y_line if strict else pred >= y_line

    def below_line(x_1, y_1, x_2, y_2, strict=False):
        return not above_line(x_1, y_1, x_2, y_2, not strict)

    def parkes_type_1(act, pred):
        # Zone E
        if above_line(0, 150, 35, 155) and above_line(35, 155, 50, 550):
            return 7
        # Zone D - left upper
        if (pred > 100 and above_line(25, 100, 50, 125) and
                above_line(50, 125, 80, 215) and above_line(80, 215, 125, 550)):
            return 6
        # Zone D - right lower
        if (act > 250 and below_line(250, 40, 550, 150)):
            return 5
        # Zone C - left upper
        if (pred > 60 and above_line(30, 60, 50, 80) and
                above_line(50, 80, 70, 110) and above_line(70, 110, 260, 550)):
            return 4
        # Zone C - right lower
        if (act > 120 and below_line(120, 30, 260, 130) and below_line(260, 130, 550, 250)):
            return 3
        # Zone B - left upper
        if (pred > 50 and above_line(30, 50, 140, 170) and
                above_line(140, 170, 280, 380) and (act < 280 or above_line(280, 380, 430, 550))):
            return 2
        # Zone B - right lower
        if (act > 50 and below_line(50, 30, 170, 145) and
                below_line(170, 145, 385, 300) and (act < 385 or below_line(385, 300, 550, 450))):
            return 1
        # Zone A
        return 0

    def parkes_type_2(act, pred):
        # Zone E
        if (pred > 200 and above_line(35, 200, 50, 550)):
            return 7
        # Zone D - left upper
        if (pred > 80 and above_line(25, 80, 35, 90) and above_line(35, 90, 125, 550)):
            return 6
        # Zone D - right lower
        if (act > 250 and below_line(250, 40, 410, 110) and below_line(410, 110, 550, 160)):
            return 5
        # Zone C - left upper
        if (pred > 60 and above_line(30, 60, 280, 550)):
            return 4
        # Zone C - right lower
        if (below_line(90, 0, 260, 130) and below_line(260, 130, 550, 250)):
            return 3
        # Zone B - left upper
        if (pred > 50 and above_line(30, 50, 230, 330) and
                (act < 230 or above_line(230, 330, 440, 550))):
            return 2
        # Zone B - right lower
        if (act > 50 and below_line(50, 30, 90, 80) and below_line(90, 80, 330, 230) and
                (act < 330 or below_line(330, 230, 550, 450))):
            return 1
        # Zone A
        return 0

    if diabetes_type == 1:
        return parkes_type_1(act, pred)

    if diabetes_type == 2:
        return parkes_type_2(act, pred)

    raise Exception('Unsupported diabetes type')

clarke_error_zone_detailed = np.vectorize(clarke_error_zone_detailed)
parkes_error_zone_detailed = np.vectorize(parkes_error_zone_detailed)

def zone_accuracy(act_arr, pred_arr, mode='clarke', detailed=False, diabetes_type=1):
    """
    Calculates the average percentage of each zone based on Clarke or Parkes
    Error Grid analysis for an array of predictions and an array of actual values
    """
    acc = np.zeros(9)
    if mode == 'clarke':
        res = clarke_error_zone_detailed(act_arr, pred_arr)
    elif mode == 'parkes':
        res = parkes_error_zone_detailed(act_arr, pred_arr, diabetes_type)
    else:
        raise Exception('Unsupported error grid mode')

    acc_bin = np.bincount(res)
    acc[:len(acc_bin)] = acc_bin

    if not detailed:
        acc[1] = acc[1] + acc[2]
        acc[2] = acc[3] + acc[4]
        acc[3] = acc[5] + acc[6]
        acc[4] = acc[7] + acc[8]
        acc = acc[:5]
    score = acc / sum(acc)
    print('CEG:  A:{:.4f}, B:{:.4f}, C:{:.4f}, D:{:.4f}, E:{:.4f}'.format(score[0], score[1], score[2], score[3],score[4]))
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


def sources_and_subsets_nodes(N):
    str1 = "{0:{fill}"+str(N)+"b}"
    a = []
    for i in range(1,2**N):
        a.append(str1.format(i, fill='0'))

    sourcesInNode = []
    sourcesNotInNode = []
    subset = []
    sourceList = list(range(N))
    def node_subset(node, sourcesInNodes):
        return [node - 2**(i) for i in sourcesInNodes]
    
    def string_to_integer_array(s, ch):
        N = len(s) 
        return [(N - i - 1) for i, ltr in enumerate(s) if ltr == ch]
    
    for j in range(len(a)):
        idxLR = string_to_integer_array(a[j],'1')
        sourcesInNode.append(idxLR)  
        sourcesNotInNode.append(list(set(sourceList) - set(idxLR)))
        subset.append(node_subset(j,idxLR))

    return sourcesInNode, subset


def subset_to_indices(indices):
    return [i for i in indices]

class Choquet_integral(torch.nn.Module):
    
    def __init__(self, N_in, N_out):
        super(Choquet_integral,self).__init__()
        self.N_in = N_in
        self.N_out = N_out
        self.nVars = 2**self.N_in - 2       
        dummy = (1./self.N_in) * torch.ones((self.nVars, self.N_out), requires_grad=True)
        dummy = dummy.to(DEVICE)
        self.vars = torch.nn.Parameter(dummy)        
        self.sourcesInNode, self.subset = sources_and_subsets_nodes(self.N_in)        
        self.sourcesInNode = [torch.tensor(x) for x in self.sourcesInNode]
        self.subset = [torch.tensor(x) for x in self.subset]
        
    def forward(self,inputs):    
        self.FM = self.chi_nn_vars(self.vars)
        sortInputs, sortInd = torch.sort(inputs,1, True)
        M, N = inputs.size()
        sortInputs = torch.cat((sortInputs, torch.zeros(M,1).to(DEVICE)), 1)
        sortInputs = sortInputs[:,:-1] -  sortInputs[:,1:]        
        out = torch.cumsum(torch.pow(2,sortInd),1) - torch.ones(1, dtype=torch.int64).to(DEVICE)        
        data = torch.zeros((M,self.nVars+1)).to(DEVICE)       
        for i in range(M):
            data[i,out[i,:]] = sortInputs[i,:]                 
        ChI = torch.matmul(data,self.FM)
            
        return ChI
    

    def chi_nn_vars(self, chi_vars):
        chi_vars = torch.abs(chi_vars)        
        FM = chi_vars[None, 0,:]
        FM = FM.to(DEVICE)
        for i in range(1,self.nVars):
            indices = subset_to_indices(self.subset[i])
            if (len(indices) == 1):
                FM = torch.cat((FM,chi_vars[None,i,:]),0)
            else:
                maxVal,_ = torch.max(FM[indices,:],0)
                temp = torch.add(maxVal,chi_vars[i,:])
                FM = torch.cat((FM,temp[None,:]),0)              
        FM = torch.cat([FM, torch.ones((1,self.N_out)).to(DEVICE)],0)
        FM = torch.min(FM, torch.ones(1).to(DEVICE))  
        
        return FM


    
# resmodel = ResModel()
# resmodel(torch.rand(10,1,20,200),torch.rand(10,1,20,200))


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(DEVICE)
net = torch.load('/data0/ljz/ECG_BG/save/random3/fusion_model_EPOCHS 80.pth')
net.eval()

path1 = '/data0/ljz/ECG_BG/ECG signal MAT'
path2 = '/data0/ljz/ECG_BG/ECG BG MAT'


data_dir_s = sorted(os.listdir(path1),key=numericalSort) 
Label_data = np.loadtxt('/data0/ljz/ECG_BG/new label.txt')
label_id = Label_data[:,0]
wave_id = Label_data[:,1]

samp_num = 20
time_start = 6
time_end = 7

CHI_train_data_list = []
CHI_train_label_list = []
CHI_test_data_list = []
CHI_test_label_list = []


for Path in data_dir_s:  
    train_pred_list = []
    train_true_list = []
    test_pred_list = []
    test_true_list = []    
    train_id_list = []
    test_id_list = []
    test_num_list = []
    train_num_list = []
    train_glu_list = []
    test_glu_list = [] 
    test_n = 0
    train_n = 0        
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

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=False) 
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)  
    for y, (train_images, tr_labels) in enumerate(train_loader):
        tr_labels = tr_labels.float()
        tr_labels = tr_labels.numpy() 
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

        train_outputs = net(train_ECG1,train_ECG2,train_ECG3)
        train_outputs = train_outputs.view(-1)                
        train_outputs = train_outputs.cpu()
        train_outputs = train_outputs.detach().numpy() 
        train_outputs = BG_scale(train_outputs)
        for t in range(data_len):                 
            train_pred_list.append(train_outputs[t])
            train_true_list.append(tr_labels[t])      
 
    for y, (test_images, te_labels) in enumerate(test_loader):
        te_labels = te_labels.float()
        te_labels = te_labels.numpy() 
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
        #         if int(label_id[s]) == temp_name:
        #             wave = int(wave_id[s])
        #             break            
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

        test_outputs = net(test_ECG1,test_ECG2,test_ECG3)
        test_outputs = test_outputs.view(-1)                
        test_outputs = test_outputs.cpu()
        test_outputs = test_outputs.detach().numpy()
        test_outputs = BG_scale(test_outputs)
        for t in range(data_len):                 
            test_pred_list.append(test_outputs[t])
            test_true_list.append(te_labels[t])
        
 
    train_pred = np.array(train_pred_list)
    train_true = np.array(train_true_list)      
    test_pred = np.array(test_pred_list)
    test_true = np.array(test_true_list)

    sp = 5
    all_len = len(train_pred) + 1
    for m in range(sp,all_len):
        st = train_pred[m-sp:m]
        ent = train_true[m-1]
        CHI_train_data_list.append(st.reshape(1,-1))
        CHI_train_label_list.append(ent)
        
    all_len = len(test_pred) + 1
    for m in range(sp,all_len):
        st = test_pred[m-sp:m]
        ent = test_true[m-1]
        CHI_test_data_list.append(st.reshape(1,-1))
        CHI_test_label_list.append(ent)    

CHI_train_data = np.array(CHI_train_data_list)
CHI_train_data = np.squeeze(CHI_train_data)
CHI_train_label = np.array(CHI_train_label_list)

CHI_test_data = np.array(CHI_test_data_list)
CHI_test_data = np.squeeze(CHI_test_data)
CHI_test_label = np.array(CHI_test_label_list)

CHI_train_data = torch.from_numpy(CHI_train_data)
CHI_train_label = torch.from_numpy(CHI_train_label)
CHI_test_data = torch.from_numpy(CHI_test_data)
CHI_test_label = torch.from_numpy(CHI_test_label)  

BATCH_SIZE2 = 512
train_set2 = torch.utils.data.TensorDataset(CHI_train_data, CHI_train_label)
test_set2 = torch.utils.data.TensorDataset(CHI_test_data, CHI_test_label)
train_loader2 = torch.utils.data.DataLoader(dataset=train_set2, batch_size=BATCH_SIZE2, shuffle=True) 
test_loader2 = torch.utils.data.DataLoader(dataset=test_set2, batch_size=BATCH_SIZE2, shuffle=False)  

CHI_net = Choquet_integral(N_in=sp,N_out=1).to(DEVICE) 
learning_rate = 0.001
criterion = torch.nn.MSELoss().to(DEVICE)
optimizer = torch.optim.Adam(CHI_net.parameters(), lr=learning_rate) 

TOTAL_EPOCHS = 10
losses = []

EPOCH_list = []
RMSE1_list = []
MARD1_list = []
RMSE2_list = []
MARD2_list = []
train_ClarkeA_list = []
train_ClarkeAB_list = []
train_CERA_list = []
train_CERAB_list = []
train_concor15_list = []
train_concor20_list = []
train_concor30_list = []
test_ClarkeA_list = []
test_ClarkeAB_list = []
test_CERA_list = []
test_CERAB_list = []
test_concor15_list = []
test_concor20_list = []
test_concor30_list = []

for epoch in range(TOTAL_EPOCHS):
    train_pred_list = []
    train_true_list = []
    test_pred_list = []
    test_true_list = []
    for x, (train_cgm, tr_labels) in enumerate(train_loader2):
        train_cgm = train_cgm.to(DEVICE)
        tr_labels = tr_labels.to(DEVICE)
        optimizer.zero_grad()
        train_outputs = CHI_net(train_cgm)
        train_outputs = train_outputs.view(-1)
        train_loss = criterion(train_outputs,tr_labels)
        train_loss.backward()
        optimizer.step() 
        Train_outputs = train_outputs.cpu()
        Train_outputs = Train_outputs.detach().numpy()
        Tr_labels = tr_labels.cpu()
        Tr_labels = Tr_labels.numpy() 
        Train_outputs = BG_scale(Train_outputs)        
        for t in range(len(Tr_labels)):
            train_pred_list.append(Train_outputs[t])
            train_true_list.append(Tr_labels[t])   
    
    CHI_net.eval()        
    for y, (test_cgm, test_labels) in enumerate(test_loader2):  
        test_cgm = test_cgm.to(DEVICE) 
        test_outputs = CHI_net(test_cgm)
        test_outputs = test_outputs.view(-1)
        test_outputs = test_outputs.cpu()
        test_outputs = test_outputs.detach().numpy()
        test_labels = test_labels.numpy()                            
        for t in range(len(test_labels)):
              test_pred_list.append(test_outputs[t])
              test_true_list.append(test_labels[t])           
 
 
    train_pred = np.array(train_pred_list)
    train_true = np.array(train_true_list)
    print('Epoch: %d' %(epoch + 1))
    train_result = RMSE_MARD(train_pred, train_true)
    train_concor = CONCOR(train_pred, train_true)
    train_Clarke = zone_accuracy(train_true, train_pred, mode='clarke', detailed=False)
    train_CER = zone_accuracy(train_true, train_pred, mode='parkes',diabetes_type=1, detailed=False)
        
    test_pred = np.array(test_pred_list)
    test_true = np.array(test_true_list)
    test_result = RMSE_MARD(test_pred, test_true)
    test_concor = CONCOR(test_pred, test_true)
    test_Clarke = zone_accuracy(test_true, test_pred, mode='clarke', detailed=False)
    test_CER = zone_accuracy(test_true, test_pred, mode='parkes',diabetes_type=1, detailed=False)
    

    torch.save(CHI_net, '/data0/ljz/ECG_BG/save/random3' + "/" + "CHI_model_EPOCHS " + str(epoch + 1) + ".pth")
    EPOCH_list.append(epoch + 1)  
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
    train_ClarkeA_list.append(train_Clarke[0])
    train_ClarkeAB_list.append(train_Clarke[0] + train_Clarke[1])
    test_ClarkeA_list.append(test_Clarke[0])
    test_ClarkeAB_list.append(test_Clarke[0] + test_Clarke[1])
    train_CERA_list.append(train_CER[0])
    train_CERAB_list.append(train_CER[0] + train_CER[1])
    test_CERA_list.append(test_CER[0])
    test_CERAB_list.append(test_CER[0] + test_CER[1])    
        
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
    test_ClarkeA = np.array(test_ClarkeA_list)
    test_ClarkeAB = np.array(test_ClarkeAB_list)
    train_CERA = np.array(train_CERA_list)
    train_CERAB = np.array(train_CERAB_list)
    test_CERA = np.array(test_CERA_list)
    test_CERAB = np.array(test_CERAB_list)    

    EPOCH = EPOCH.reshape(-1,1)   
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
    test_ClarkeA = test_ClarkeA.reshape(-1,1)
    test_ClarkeAB = test_ClarkeAB.reshape(-1,1)
    train_CERA = train_CERA.reshape(-1,1)
    train_CERAB = train_CERAB.reshape(-1,1)   
    test_CERA = test_CERA.reshape(-1,1)
    test_CERAB = test_CERAB.reshape(-1,1)
    train_save = np.hstack((EPOCH, RMSE1, MARD1, train_concor15, train_concor20, train_concor30,
                            train_ClarkeA, train_ClarkeAB,train_CERA, train_CERAB))
    test_save = np.hstack((EPOCH, RMSE2, MARD2, test_concor15, test_concor20, test_concor30,
                            test_ClarkeA, test_ClarkeAB, test_CERA, test_CERAB))
    np.savetxt('/data0/ljz/ECG_BG/save/random3/CHI train result.csv', train_save, delimiter=',', fmt='%1.5f')
    np.savetxt('/data0/ljz/ECG_BG/save/random3/CHI test result.csv', test_save, delimiter=',', fmt='%1.5f')

