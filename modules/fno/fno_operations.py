import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Flatten
import numpy as np
import os
from timeit import default_timer
import warnings
from torch.autograd import Variable


def data_loaders(data, batch_size, normalized, shuffle_train=False):
    if normalized:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data['u_train_n'], data['g_u_train_n']), batch_size=batch_size, shuffle=shuffle_train)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data['u_val_n'], data['g_u_val_n']), batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data['u_test_n'], data['g_u_test_n']), batch_size=batch_size, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data['u_train'], data['g_u_train']), batch_size=batch_size, shuffle=shuffle_train)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data['u_val'], data['g_u_val']), batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data['u_test'], data['g_u_test']), batch_size=batch_size, shuffle=False)
    
    warn_txt = "{dataname} length {length} must be a multiple of the batch size {batch_size}."
    if len(train_loader.dataset)%batch_size!=0:
        raise ValueError(warn_txt.format(dataname="Training data", length=len(train_loader.dataset), batch_size=batch_size))
    if len(val_loader.dataset)%batch_size!=0:
        raise ValueError(warn_txt.format(dataname="Validation data", length=len(val_loader.dataset), batch_size=batch_size))
    if len(test_loader.dataset)%batch_size!=0:
        raise ValueError(warn_txt.format(dataname="Testing data", length=len(test_loader.dataset), batch_size=batch_size))

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, n_epochs, optimizer, scheduler, checkpoints_folder="checkpoints",
                checkpoints_freq=1000, log_output=True, log_output_freq=100, log_temp_path="training_log.out", last_cp=0, cuda=True):
    
    train_loss, val_loss = list(), list()
    checkpoint_path = os.path.join(checkpoints_folder, "cp-{epoch:04d}.ckpt")
    log_temp = open(log_temp_path, 'w')

    if log_output:
        log_string = ""

    for i in range(last_cp, last_cp+n_epochs):
        
        t1 = default_timer()

        mse_train = train_step(model, data_loader=train_loader, optimizer=optimizer, scheduler=scheduler, cuda=cuda)
        train_loss.append(mse_train)

        mse_val, _, _ = evaluate_model(model, data_loader=val_loader, inverse_norm=False, g_u_normalizer=None, cuda=cuda)
        val_loss.append(mse_val)

        t2 = default_timer()
        
        # TODO: use function from log

        if i % log_output_freq == 0:

            log_epoch = "epoch: {}\t time: {}\t train_mse: {}\t val_mse: {}".format(
                i, np.round(t2-t1,5), np.round(mse_train,5), np.round(mse_val,5))       
            print(log_epoch)

            with open(log_temp_path, 'a') as log_temp:
                log_temp.write(f"{log_epoch}\n")
            log_temp.close()

            if log_output:
                log_string += log_epoch + "\n"

        if (i+1) % checkpoints_freq == 0:
            torch.save(model.state_dict(), checkpoint_path.format(epoch=(i+1)))  

    history = dict({'train_loss': train_loss, 'val_loss': val_loss})
    
    if log_output:
        return history, log_string

    else:
        return history

def train_step(model, data_loader, optimizer, scheduler, cuda=True):
    model.train()
    mse_total = 0
    #loss = nn.MSELoss()
    for x, y in data_loader:
        x, y = (x.cuda(), y.cuda()) if cuda else (x.float(), y.float())
        
        optimizer.zero_grad(set_to_none=True)
        
        #if len(y.shape)==3:
        y_pred = model(x).view(data_loader.batch_size, y.shape[1], y.shape[2])
        #else:
        #    y_pred = model(x).view(data_loader.batch_size, y.shape[1], y.shape[2], y.shape[3])
            
        mse = F.mse_loss(y_pred, y, reduction='mean')
        #mse = loss(y_pred, y)
        mse.backward()

        optimizer.step()
        mse_total += mse.item()

    mse_total /= len(data_loader)
    scheduler.step()

    return mse_total

def evaluate_model(model, data_loader, inverse_norm, g_u_normalizer=None, cuda=True):

    mse_total = 0.0
    y_pred_all = list()
    y_all = list()

    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x, y = (x.cuda(), y.cuda()) if cuda else (x.float(), y.float())
            if len(y.shape)==3:
                y_pred = model(x).view(data_loader.batch_size, y.shape[1], y.shape[2])
            else:
                y_pred = model(x).view(data_loader.batch_size, y.shape[1], y.shape[2], y.shape[3])
 
            if inverse_norm:  
                y = g_u_normalizer.decode(y)
                y_pred = g_u_normalizer.decode(y_pred)
            y_pred_all.append(y_pred)    
            y_all.append(y)

            mse = F.mse_loss(y_pred, y, reduction='mean')
            mse_total += mse.item()

    mse_total /= len(data_loader) 
    y_pred_all = torch.cat(y_pred_all)
    y_all = torch.cat(y_all)

    return mse_total, y_pred_all, y_all

################################################################
# 2d fourier layers
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

#class FNO2d_RNN(nn.Module):
#    """
#    This is the regular FNO for now
#    """
#    def __init__(self, modes1, modes2,  width):
#        super(FNO2d_RNN, self).__init__()
#
#        """
#        The overall network. It contains 4 layers of the Fourier layer.
#        1. Lift the input to the desire channel dimension by self.fc0 .
#        2. 4 layers of the integral operators u' = (W + K)(u).
#            W defined by self.w; K defined by self.conv .
#        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
#        
#        input: the solution of the coefficient function and locations (a(x, y), x, y)
#        input shape: (batchsize, x=s, y=s, c=3)
#        output: the solution 
#        output shape: (batchsize, x=s, y=s, c=1)
#        """
#
#        self.modes1 = modes1
#        self.modes2 = modes2
#        self.width = width
#        self.padding = 9 # pad the domain if input is non-periodic
#        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

#        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#        self.w0 = nn.Conv2d(self.width, self.width, 1)
#        self.w1 = nn.Conv2d(self.width, self.width, 1)
#        self.w2 = nn.Conv2d(self.width, self.width, 1)
#        self.w3 = nn.Conv2d(self.width, self.width, 1)
#
#        self.fc1 = nn.Linear(self.width, 128)
#        self.fc2 = nn.Linear(128, 1)
#
#
#    def forward(self, x):
#        grid = self.get_grid(x.shape, x.device)
#        x = torch.cat((x, grid), dim=-1)
#        x = self.fc0(x)
#        x = x.permute(0, 3, 1, 2)
#        x = F.pad(x, [0,self.padding, 0,self.padding])
#
#        x1 = self.conv0(x)
#        x2 = self.w0(x)
#        x = x1 + x2
#        x = F.gelu(x)
#
#        x1 = self.conv1(x)
#        x2 = self.w1(x)
#        x = x1 + x2
#        x = F.gelu(x)
#
#        x1 = self.conv2(x)
#        x2 = self.w2(x)
#        x = x1 + x2
#        x = F.gelu(x)
#
#        x1 = self.conv3(x)
#        x2 = self.w3(x)
#        x = x1 + x2
#
#        x = x[..., :-self.padding, :-self.padding]
#        x = x.permute(0, 2, 3, 1)
#
#        x = self.fc1(x)
#        x = F.gelu(x)
#        x = self.fc2(x)
#
#        return x
#    
#
#    def get_grid(self, shape, device):
#        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
#        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#        return torch.cat((gridx, gridy), dim=-1).to(device)


class FNO2d_RNN(nn.Module):
    """
    For already trained models this is named FNO2d_RNN2.
    """
    def __init__(self, modes1, modes2,  width, hidden_rnn, x_len):
        super(FNO2d_RNN, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.hidden_rnn = hidden_rnn
        
        #n_neurons = 200
        #x_len = 50
        self.x_len = x_len
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.rnn = nn.RNN(self.x_len, self.hidden_rnn, 1, batch_first=True, nonlinearity='tanh')

        # Readout layer
        self.fc_out = nn.Linear(self.hidden_rnn, self.x_len)

        #self.rnn = nn.RNN(input_size=x_len, hidden_size=n_neurons, num_layers=1)


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        x = x.reshape(x.size(0), x.size(1), x.size(2))

        # One time step
        x, hn = self.rnn(x)#, h0.to(x.device))
        x = self.fc_out(x) #To return only the last timestep: [:, -1, :]) 

        return x.to(x.device)
    

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class FNO2d_LSTM(nn.Module):
    """
    For already trained models this is named FNO2d_RNN2.
    """
    def __init__(self, modes1, modes2,  width, hidden_rnn, x_len):
        super(FNO2d_LSTM, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.hidden_rnn = hidden_rnn
        
        #n_neurons = 200
        #x_len = 50
        self.x_len = x_len
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.lstm = nn.LSTM(self.x_len, self.hidden_rnn, 1, batch_first=True)

        # Readout layer
        self.fc_out = nn.Linear(self.hidden_rnn, self.x_len)

        #self.rnn = nn.RNN(input_size=x_len, hidden_size=n_neurons, num_layers=1)


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        x = x.reshape(x.size(0), x.size(1), x.size(2))

        # One time step
        x, hn = self.lstm(x)#, h0.to(x.device))
        x = self.fc_out(x) #To return only the last timestep: [:, -1, :]) 

        return x.to(x.device)
    

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class FNO2d_GRU(nn.Module):
    """
    For already trained models this is named FNO2d_RNN2.
    """
    def __init__(self, modes1, modes2,  width, hidden_rnn, x_len):
        super(FNO2d_GRU, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.hidden_rnn = hidden_rnn
        
        #n_neurons = 200
        #x_len = 50
        self.x_len = x_len
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.gru = nn.GRU(self.x_len, self.hidden_rnn, 1, batch_first=True)

        # Readout layer
        self.fc_out = nn.Linear(self.hidden_rnn, self.x_len)

        #self.rnn = nn.RNN(input_size=x_len, hidden_size=n_neurons, num_layers=1)


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        x = x.reshape(x.size(0), x.size(1), x.size(2))

        # One time step
        x, hn = self.gru(x)#, h0.to(x.device))
        x = self.fc_out(x) #To return only the last timestep: [:, -1, :]) 

        return x.to(x.device)
    

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(4, self.width) # input channel is 4: (a(x, y), x, y)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

class FNO3d_RNN(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, hidden_rnn, x_len):
        super(FNO3d_RNN, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.x_len = x_len
        self.hidden_rnn = hidden_rnn
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(4, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.rnn = nn.RNN(self.x_len, self.hidden_rnn, 1, batch_first=True, nonlinearity='tanh')

        # Readout layer
        self.fc_out = nn.Linear(self.hidden_rnn, self.x_len)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2, 4)

        x = torch.flatten(x, 2, -1)

        # One time step
        x, hn = self.rnn(x)#, h0.to(x.device))
        x = self.fc_out(x)#[:, -1, :]) 

        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)