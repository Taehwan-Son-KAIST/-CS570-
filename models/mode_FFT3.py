import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import numpy as np
from torchvision import transforms

def hardwire_layer(input, device, verbose=False):
    """
    Proprocess the given consecutive input (grayscaled) frames into 2 different styles. 
    input : array of shape (N, frames, height, width)
    ex) TRECVID dataset : (N, 7, 60, 40),  KTH dataset : (N, 9, 80, 60)
    
    ##################################### FFT frames #####################################
    # content: [[[FFT_Amplitude frames], [FFT_Phase frames]], ...]
    # shape:   [[[---- f ----], [----- f -----]], ...]
    #           => total: 2f frames
    ############################################################################################
    """

    """
    cut_param: Proportion of cutting size for each direction (x, y).

    """
    assert len(input.shape) == 4 
    if verbose: print("Before hardwired layer:\t", input.shape)
    N, f, h, w = input.shape
    
    cut_param = 0.5
    cut_size_h = round(h*cut_param+1e-8)
    cut_size_w = round(w*cut_param+1e-8)
    
    if cut_size_h % 2 == 0:
        cut_size_h = cut_size_h + 1
        
    if cut_size_w % 2 == 0:
        cut_size_w = cut_size_w + 1
    
    hardwired = torch.zeros((N, 2*f, cut_size_h, cut_size_w)).to(device)
    input = input.to(device)

    gray = input.clone()
    
    gray_cut = torch.zeros((N, f, cut_size_h, cut_size_w))
    fft_xy_abs = torch.zeros((N, f, cut_size_h, cut_size_w))
    fft_xy_phase = torch.zeros((N, f, cut_size_h, cut_size_w))
    for i in range(N):
        for j in range(f):
            img = gray[i][j]
            fft_tensor = torch.fft.fftshift(torch.fft.fft2(img, norm='ortho'))

            cut_temp_h = round((cut_size_h-1)/2+1e-8)
            cut_temp_w = round((cut_size_w-1)/2+1e-8)

            fft_xy_tensor_cut = fft_tensor[round(np.size(img,0)/2+1e-8)-cut_temp_h:round(np.size(img,0)/2+1e-8)+cut_temp_h+1, round(np.size(img,1)/2+1e-8)-cut_temp_w:round(np.size(img,1)/2+1e-8)+cut_temp_w+1]

            fft_xy_abs_temp = torch.abs(fft_xy_tensor_cut) # h*w
            fft_xy_phase_temp = torch.angle(fft_xy_tensor_cut) # h*w
            
            fft_xy_abs[i,j,:,:] = fft_xy_abs_temp
            fft_xy_phase[i,j,:,:] = fft_xy_phase_temp # (N*h)*f*w
            
            gray_cut_temp = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fft_xy_tensor_cut), norm='ortho'))
            
            gray_cut_temp = gray_cut_temp/torch.max(gray_cut_temp)*255
            
            gray_cut[i,j,:,:] = gray_cut_temp
    
    fft_xy_abs = fft_xy_abs.to(device)
    fft_xy_phase = fft_xy_phase.to(device)
    
    fft_tx_abs = torch.zeros((N, f, cut_size_h, cut_size_w))
    fft_tx_phase = torch.zeros((N, f, cut_size_h, cut_size_w))
    for i in range(N):
        for j in range(cut_size_h):
            tx = gray_cut[i][:, j, :]
            
            fft_tx_tensor = torch.fft.fftshift(torch.fft.fft2(tx, norm='ortho'))

            fft_tx_abs_temp = torch.abs(fft_tx_tensor) # f*w
            fft_tx_phase_temp = torch.angle(fft_tx_tensor) # f*w
            
            fft_tx_abs[i,:,j,:] = fft_tx_abs_temp
            fft_tx_phase[i,:,j,:] = fft_tx_phase_temp # (N*h)*f*w
    
    fft_tx_abs = fft_tx_abs.to(device)
    fft_tx_phase = fft_tx_phase.to(device)
    
    fft_ty_abs = torch.zeros((N, f, cut_size_h, cut_size_w))
    fft_ty_phase = torch.zeros((N, f, cut_size_h, cut_size_w))
    for i in range(N):
        for j in range(cut_size_w):
            ty = gray_cut[i][:, :, j]
            
            fft_ty_tensor = torch.fft.fftshift(torch.fft.fft2(ty, norm='ortho'))

            fft_ty_abs_temp = torch.abs(fft_ty_tensor) # f*w
            fft_ty_phase_temp = torch.angle(fft_ty_tensor) # f*w
            
            fft_ty_abs[i,:,:,j] = fft_ty_abs_temp
            fft_ty_phase[i,:,:,j] = fft_ty_phase_temp # (N*h)*f*w
    
    fft_ty_abs = fft_ty_abs.to(device)
    fft_ty_phase = fft_ty_phase.to(device)

    hardwired = torch.cat([fft_xy_abs, fft_xy_phase, fft_tx_abs, fft_tx_phase, fft_ty_abs, fft_ty_phase], dim=1)
    hardwired = hardwired.unsqueeze(dim=1)
    if verbose: print("After hardwired layer :\t", hardwired.shape)
    return hardwired


#* 3D-CNN Model
class Original_Model(nn.Module):
    """
    3D-CNN model designed by the '3D-CNN for HAR' paper. 
    Input Shape: (N, C_in=1, Dimension=2f, Height=h, Width=w)
    """
    def __init__(self, verbose=False, mode='KTH'):
        """You need to give the dataset type as 'mode', which is one of 'KTH' or 'TRECVID'."""
        super(Original_Model, self).__init__()
        self.verbose = verbose
        self.mode = mode
        if self.mode == 'KTH':
            self.f = 9 # num. of frames
        elif self.mode == 'TRECVID':
            self.f = 7
        else:
            print("This mode is not available. Choose one of KTH or TRECVID.")
            return 
        self.dim = self.f * 6
        self.dim1, self.dim2 = (self.dim-12)*2, (self.dim-24)*6

        if self.mode == 'KTH':
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(3,10,8), stride=1)
            self.conv2 = nn.Conv3d(in_channels=2, out_channels=6, kernel_size=(3,5,4), stride=1)
            self.pool1 = nn.MaxPool2d(2)
            self.pool2 = nn.MaxPool2d(3)
            self.conv3 = nn.Conv2d(in_channels=self.dim2, out_channels=128, kernel_size=(4,3), stride=1)
            self.fc1 = nn.Linear(128, 6, bias=False)

        elif self.mode == 'TRECVID':
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(3,7,7), stride=1)
            self.conv2 = nn.Conv3d(in_channels=2, out_channels=6, kernel_size=(3,7,6), stride=1)
            self.pool1 = nn.MaxPool2d(2)
            self.pool2 = nn.MaxPool2d(3)
            self.conv3 = nn.Conv2d(in_channels=self.dim2, out_channels=128, kernel_size=(7,4), stride=1)
            self.fc1 = nn.Linear(128, 3, bias=False)
        

    def forward(self, x):
        if self.verbose: print("연산 전:\t", x.size())
        assert x.size()[1] == 1

        (x1, x2, x3, x4, x5, x6) = torch.split(x, [self.f,self.f,self.f,self.f,self.f,self.f], dim=2)
        x1 = F.relu(self.conv1(x1))
        x2 = F.relu(self.conv1(x2))
        x3 = F.relu(self.conv1(x3))
        x4 = F.relu(self.conv1(x4))
        x5 = F.relu(self.conv1(x5))
        x6 = F.relu(self.conv1(x6))

        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=2)
        if self.verbose: print("conv1 연산 후:\t", x.shape)

        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        x = self.pool1(x)
        x = x.view(-1, 2, self.dim1//2, x.shape[2], x.shape[3])
        if self.verbose: print("pool1 연산 후:\t", x.shape)

        (x1, x2, x3, x4, x5, x6) = torch.split(x, [self.f-2,self.f-2,self.f-2,self.f-2,self.f-2,self.f-2], dim=2)
        x1 = F.relu(self.conv2(x1))
        x2 = F.relu(self.conv2(x2))
        x3 = F.relu(self.conv2(x3))
        x4 = F.relu(self.conv2(x4))
        x5 = F.relu(self.conv2(x5))
        x6 = F.relu(self.conv2(x6))

        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=2)
        if self.verbose: print("conv2 연산 후:\t",x.shape)

        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        x = self.pool2(x)
        x = x.view(-1, self.dim2, x.shape[2], x.shape[3])
        if self.verbose: print("pool2 연산 후:\t", x.shape)

        x = F.relu(self.conv3(x))
        if self.verbose: print("conv3 연산 후:\t", x.shape)

        x = x.view(-1, 128)
        x = self.fc1(x)
        if self.verbose: print("fc1 연산 후:\t", x.shape)

        return x
