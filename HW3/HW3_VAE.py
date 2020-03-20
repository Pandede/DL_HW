import torch
import torch.nn as nn
from torch.nn import functional as F

from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda:0')

class DataLoader:
    def __init__(self, folder_path, img_size):
        self.folder_path = folder_path
        self.img_size = img_size
        
        self.path_list = glob(folder_path + '/*')
    
    def __imread(self, img_path):
        return np.array(Image.open(img_path).convert('RGB').resize(self.img_size[:-1]))
    
    def sampling_data(self, batch_size, shuffle=True, channel_first=False):
        img_path_list = self.path_list
        
        if shuffle:
            img_idx = np.arange(len(img_path_list))
            np.random.shuffle(img_idx)
            img_path_list = list(np.array(img_path_list)[img_idx])
        
        for batch_idx in range(0, len(img_path_list), batch_size):
            img_slice = slice(batch_idx, batch_idx + batch_size)
            path_set = img_path_list[img_slice]
            
            img_set = np.zeros((len(path_set),) + self.img_size)
            for img_idx, path in enumerate(path_set):
                img_set[img_idx] = self.__imread(path)
            img_set = img_set / 127.5 - 1
            if not channel_first:
                img_set = np.transpose(img_set, (0, 3, 1, 2))
            yield img_set
            
class VAE(nn.Module):
    def __init__(self, img_size, latent_dim=128):
        super(VAE, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(img_size, latent_dim)
        print(self.__count_params(self.encoder))
        self.decoder = Decoder(img_size, latent_dim)
        print(self.__count_params(self.decoder))
        self.dataloader = DataLoader('./cartoon/cartoon', self.img_size)
        
        self.mse_criterion = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.parameters())
        
    def forward(self, x):
        _, _, h = self.encoder(x)
        return self.decoder(h)
    
    def __kl_loss(self, z_mean, z_logstd):
        return -0.5 * torch.sum(1 + z_logstd - z_mean**2 - torch.exp(z_logstd))
    
    def __count_params(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def __sample_image(self, epoch):
        self.eval()
        with torch.no_grad():   
            r, c = 7, 4
            x = next(self.dataloader.sampling_data(r*c))
            reconstruct_x = self(torch.from_numpy(x).type('torch.FloatTensor').to(device))
            
            x = np.transpose(x, (0, 2, 3, 1)).reshape((-1, ) + self.img_size) * .5 + .5
            reconstruct_x = np.transpose(reconstruct_x.cpu().numpy(), (0, 2, 3, 1)).reshape((-1, ) + self.img_size) * .5 + .5
            
            fig = plt.figure(figsize=(14, 14))
            axs = fig.subplots(r, c)
            
            fig_cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(np.concatenate((x[fig_cnt], reconstruct_x[fig_cnt]), axis=1))
                    axs[i, j].set_axis_off()
                    fig_cnt += 1
            fig.savefig('./cc_images/%d.png' % epoch)
            plt.close()
        
    def fit(self, epochs, batch_size, sample_interval=20):
        self.history = []
        for e in range(epochs):
            batch_loss = 0
            for i, batch_img in enumerate(self.dataloader.sampling_data(batch_size)):
                self.train()
                self.optimizer.zero_grad()
                x = torch.from_numpy(batch_img).type('torch.FloatTensor').to(device)
                z_mean, z_logstd, z = self.encoder(x)
                reconstruct_x = self.decoder(z)
                loss = self.mse_criterion(x, reconstruct_x) + self.__kl_loss(z_mean, z_logstd) * 10
                loss.backward()
                self.optimizer.step()
                
                batch_loss += loss.item() 
            print('[Epoch %d] [Loss: %f]' % (e, loss.item()/len(batch_img)))
            
            torch.save(self.state_dict(), 'vae.pkl')
            self.__sample_image(e)
            batch_loss /= len(self.dataloader.path_list)
            self.history.append(batch_loss)
        return self.history

        
class Encoder(nn.Module):
    def __init__(self, img_size, latent_dim):
        super(Encoder, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        
        self.conv1 = nn.Sequential(
                nn.Conv2d(img_size[-1], 8, 3, stride=2, padding=1),
                nn.BatchNorm2d(8, momentum=.8),
                nn.ReLU())
        self.conv2 = nn.Sequential(
                nn.Conv2d(8, 12, 3, stride=2, padding=1),
                nn.BatchNorm2d(12, momentum=.8),
                nn.ReLU())
        self.conv3 = nn.Sequential(
                nn.Conv2d(12, 24, 3, stride=2, padding=1),
                nn.BatchNorm2d(24, momentum=.8),
                nn.ReLU())
        self.conv4 = nn.Sequential(
                nn.Conv2d(24, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32, momentum=.8),
                nn.ReLU())
        
        conv_shape = (32, 8, 8)
        self.z_mean = nn.Linear(np.prod(conv_shape), latent_dim)
        self.z_logstd = nn.Linear(np.prod(conv_shape), latent_dim)

    def __sampling_z(self, z_mean, z_logstd):
        z_std = torch.exp(z_logstd * .5)
        epsilon = torch.randn(self.latent_dim).to(device)
        return z_mean + z_std * epsilon
    
    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = h.view(h.size(0), -1)
        z_mean = self.z_mean(h)
        z_logstd = self.z_logstd(h)
        return z_mean, z_logstd, self.__sampling_z(z_mean, z_logstd)

class Decoder(nn.Module):
    def __init__(self, img_size, latent_dim):
        super(Decoder, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        
        self.conv_shape = (32, 8, 8)
        self.latent_linear = nn.Linear(self.latent_dim, np.prod(self.conv_shape))
        
        self.conv1_t = nn.Sequential(
                nn.ConvTranspose2d(self.conv_shape[0], 24, 2, stride=2),
                nn.BatchNorm2d(24, momentum=.8),
                nn.ReLU())
        self.conv2_t = nn.Sequential(
                nn.ConvTranspose2d(24, 12, 2, stride=2),
                nn.BatchNorm2d(12, momentum=.8),
                nn.ReLU())
        self.conv3_t = nn.Sequential(
                nn.ConvTranspose2d(12, 8, 2, stride=2),
                nn.BatchNorm2d(8, momentum=.8),
                nn.ReLU())
        self.conv4_t = nn.ConvTranspose2d(8, self.img_size[-1], 2, stride=2)
        
    def forward(self, latent):
        h = self.latent_linear(latent)
        h = h.view(h.size(0), *self.conv_shape)
        h = self.conv1_t(h)
        h = self.conv2_t(h)
        h = self.conv3_t(h)
        return F.tanh(self.conv4_t(h))
    
vae = VAE((128, 128, 3), latent_dim=48).to(device)
vae.load_state_dict(torch.load('vae.pkl'))

history = vae.fit(300, 64, sample_interval=20)
torch.save(vae.state_dict(), 'vae.pkl')

plt.title('Learning Curve of VAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history)

vae.decoder.eval()
with torch.no_grad():
    r, c = 8, 8
    noise = torch.randn(r*c, vae.latent_dim).to(device)
    gen_img = vae.decoder(noise).cpu().numpy() *.5 + .5
    gen_img = np.transpose(gen_img, (0, 2, 3, 1))
    fig_cnt = 0
    fig = plt.figure(figsize=(14, 14))
    axs = fig.subplots(r, c)
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_img[fig_cnt])
            axs[i, j].axis('off')
            fig_cnt += 1
    
    plt.show()
    plt.close()