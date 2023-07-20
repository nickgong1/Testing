import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import numpy as np
import torchvision.models as model
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from resnet import res_for_cifar


device = "cuda"

class flatten(nn.Module):
    def forward(self,x):
      x = torch.squeeze(x, dim = 1)
      return torch.flatten(x,1)


class encoder(nn.Module):
  def __init__(self, in_channel, out_channel):
    super().__init__()
    resnet = res_for_cifar()
    resnet.linear = nn.Linear(512,256)
    self.encoder = resnet.to(device)

  def forward(self, x):
    output = self.encoder(x)
    return output
  

class autoregres(nn.Module):
  def __init__(self, in_dim, hidden_dim, num = 1):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.gru = nn.GRU(in_dim, hidden_dim, num, bidirectional = False, batch_first = True).to(device)
  
  def forward(self, x, batch = None):
    hidden = torch.zeros(1, batch, self.hidden_dim ).to(device)
    return self.gru(x, hidden)


  
class linear_layer(nn.Module):
  def __init__(self, in_dim, max_num) -> None:
    super().__init__()
    self.linear_list = nn.ModuleList()
    for i in range(max_num):
      self.linear_list.append(
        nn.Sequential(
        nn.Linear(in_dim,64),
        nn.Linear(64,256)
        ).to(device)
      )

  def get_layer(self):
    return self.linear_list
  


class CPClayer(nn.Module):
  def __init__(self, in_channel, encoder_out_channel, gru_out_channel, pred_len):
    super().__init__()
    self.encoder = encoder(in_channel, encoder_out_channel)
    self.autoreg = autoregres(encoder_out_channel, gru_out_channel)
    ll = linear_layer(gru_out_channel,pred_len)
    self.linear_list = ll.get_layer()
    self.pred_len = pred_len
    self.loss = nn.BCELoss(reduction = 'mean')

  def regist_param(self, encoder_lr = 0.001, reg_lr = 0.001,linear_lr = 0.001):
    param_list = []
    encoder_param = self.encoder.parameters()
    autoreg_param = self.autoreg.parameters()
    param_list.append({'params':encoder_param, 'lr':encoder_lr})
    param_list.append({'params':autoreg_param, 'lr':reg_lr})

    for i in range(self.pred_len):
      param_list.append({'params': self.linear_list[i].parameters(), 'lr':linear_lr})

    self.optim = SGD(param_list)
    return self.optim

    
  def get_encode(self,x):
    return self.encoder(x).detach()
  
  
  def forward(self, x, test = False):
    out = self.encoder(x)
    if test:
      if len(out.shape) == 1:
        out = torch.unsqueeze(out,dim = 0)
        out = torch.unsqueeze(out,dim=0)
      else:
        out = torch.unsqueeze(out,1)
      output, hidden = self.autoreg(out, out.shape[0])
    else:
      out = torch.unsqueeze(out,dim = 0)
      if len(out.shape) == 1:
        out = torch.unsqueeze(out,dim = 0)
      output, hidden = self.autoreg(out, out.shape[0])

    hidden = torch.squeeze(hidden,0)
    return hidden
  
  
  def train(self,train_batch,pred_batch,label, temperature = 0.5):
    param_before = [params.clone() for params in self.encoder.parameters()]
    train_batch = train_batch.to(device)
    pred_batch = pred_batch.to(device)
    label = torch.squeeze(label).to(device)
    output = torch.zeros([len(train_batch), self.pred_len])
    for i in range(len(train_batch)):
      out = self.forward(train_batch[i])
      for j in range(self.pred_len):
        clasific = F.normalize(self.linear_list[j](out).squeeze(),dim = 0)
        pred = self.get_encode(pred_batch[i][j].unsqueeze(0))
        sim = (clasific * pred).sum()/temperature
        output[i][j] = sim

    self.optim.zero_grad()

    for k in range(self.pred_len):
      idx_out = output[:,k]
      idx_out = torch.squeeze(idx_out).to(device)

      pos_all = torch.where(label == 1)
      neg_all = torch.where(label == 0)
      pos_all = torch.exp(idx_out[pos_all]).sum()
      neg_all = torch.exp(idx_out[neg_all]).sum()

      loss_ = -torch.log(pos_all/(pos_all+neg_all))

      # print(pos_all)
      # print(neg_all)
      # print("####################################################")
      # loss_ = self.loss(idx_out, label)
      if k == 1:
        print(loss_)
      if k == self.pred_len - 1:
        loss_.backward()
      else:
        loss_.backward(retain_graph = True)
    
    self.optim.step()
    
    # for before, after in zip(param_before,self.encoder.parameters()):
    #   # if torch.equal(before, after):
    #   #   print("No")
    #   # else:
    #   #   print("Yes")
    #   print(after.grad)
    # print("##############################")



class Classic(nn.Module):
  def __init__(self, in_dim) -> None:
    super().__init__()
    self.linear  = nn.Sequential(
      nn.Linear(in_dim, 256),
      nn.Linear(256, 512),
      nn.Linear(512,10)
    ).to(device)
    self.optim = SGD(self.linear.parameters(), 0.01)

  def forward (self,x):
    return self.linear(x)
    
  def train(self, train_batch, label):
    out = self.forward(train_batch)
    out = torch.squeeze(out).to(device)
    loss = nn.CrossEntropyLoss()
    gradient = loss(out, label)
    self.optim.zero_grad()
    gradient.backward()
    self.optim.step()
                                                                                                                          
    


    

  



