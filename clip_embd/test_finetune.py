import torch
import torch.nn as nn
from torch.optim import SGD
import clip
from tqdm import tqdm


class Finetune():
    def __init__(self, indx, epoch, model):
        self.model = model
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.optim = SGD(self.model.parameters(), 0.001)
        self.indx  = indx
        self.epoch = epoch

    def fine_tune(self, train_loader):
        params = []
        for name, param in self.model.named_parameters():
                if "visual." not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    params.append(param)
        self.params = params

        for i in tqdm(range(self.epoch)):
            for data, label in train_loader:
                data = data.cuda()
                img_emb = self.model.encode_image(data)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                txt_emb = torch.cat([clip.tokenize(f"the image of a {c}") for c in self.indx]).cuda()
                txt_emb = self.model.encode_text(txt_emb)
                txt_emb = txt_emb/ txt_emb.norm(dim=-1, keepdim=True)

                self.optim.zero_grad()
                l = self.get_loss(img_emb, label, txt_emb)
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.params, 1.0)
                self.optim.step()
        
    
    def get_loss(self,img, label, txt):
        sim = (img @ txt.T)
        ref = torch.zeros((img.size(0), txt.size(0)))
        for i in range(len(ref)):
            ref[i][label[i]] = 1
        ref = ref.cuda()
        loss = self.loss(sim, ref)
        return loss


