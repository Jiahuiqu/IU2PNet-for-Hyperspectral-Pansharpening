import torch
import torch.nn as nn
import torch.optim as optim
from Generator import IU2Pnet
from Discriminator import Discriminator_PAN, Discriminator_HS
from scipy.io import loadmat, savemat
from My_dataset import HS_Dataload
import os
from torch.utils.data import DataLoader
from config import config
import numpy as np
import cv2
device = torch.device("cuda:0")
torch.manual_seed(1)

def filter_downsample(output, data):
    batchsz = output.shape[0]
    final_ouput = np.zeros((batchsz, 31, 32, 32))
    for i in range(batchsz):
        pic_img = output[i,...]
        for j in range(31):
            new_pic = pic_img[j,...]
            out = cv2.filter2D(new_pic,-1,data)
            final_ouput[i,j,...] = out[::4,::4]
    final_ouput = torch.from_numpy(final_ouput)
    return final_ouput

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1).to(device)
    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size()).to(device)
    if torch.cuda.is_available():
        fake = fake.cuda()
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


#### according the wald's protocol
def get_PAN(output):

    return torch.mean(output[:, :31, ...],dim = 1).unsqueeze(1)

def SAM(output, HS):
    data1 = torch.sum(output * HS, dim=1)
    # data2 = output.norm(2,dim = 1) * HS.norm(2, dim = 1)
    data2 = torch.sqrt(torch.sum((output ** 2), dim=1) * torch.sum((HS ** 2), dim=1))
    sam_loss = torch.acos((data1 / data2).clamp(-1,1)).view(-1).mean().type(torch.float32)
    sam_loss = sam_loss.clone().detach().requires_grad_(True)
    return sam_loss

def RMSE_loss(x1, x2):
    x = x1 - x2
    n, c, h, w = x.shape
    x = torch.pow(x, 2)
    out = torch.sum(x, dim=(1, 2, 3))
    out = torch.pow(torch.div(out, c * h * w), 0.5)
    out = torch.sum(out, 0)
    out = torch.div(out, n)
    return out


def adjust_learning_rate(lr, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def smooth_gan_labels(matrix, labels):
    if labels == 0:
        y_out = torch.div(torch.randint(0, 2,size = (matrix.shape[0],1), dtype = torch.float32), 10)
    else:
        y_out = torch.div(torch.randint(9, 12,size = (matrix.shape[0],1),dtype = torch.float32), 10)
    return y_out

def train(max_iter, batchsz, P_critic,HS_critic, batch):
    if torch.cuda.device_count() > 1:
        print("Running!!!")
        #D = nn.DataParallel(Discriminator_PAN().apply(weights_init_normal))
        D_P = nn.DataParallel(Discriminator_PAN())
        D_H = nn.DataParallel(Discriminator_HS())
        HS_G = nn.DataParallel(IU2Pnet(31, 4, 4))
    D_P = D_P.to(device)
    D_H = D_H.to(device)
    HS_G = HS_G.to(device)
    criteron_G = nn.L1Loss()
    # optimizer_D = optim.RMSprop(D.parameters(),lr = 1e-6)
    # optimizer_HSG = optim.RMSprop(HS_G.parameters(), lr = 1e-4)
    lr = 1e-4
    optimizer_D_P = optim.Adam(D_P.parameters(), lr=lr)
    optimizer_D_H = optim.Adam(D_H.parameters(), lr=lr)
    optimizer_HSG = optim.Adam(HS_G.parameters(), lr=lr)
    train_db = HS_Dataload("data","train", 128)
    dataloader = DataLoader(train_db, batch_size = batchsz, shuffle = True,num_workers = 16, pin_memory = True)
    kernel_size = loadmat("BluKer.mat")['ans']
    index = 0
    batches_done = 0
    for epoch in range(max_iter):
        for step, (PAN, LRHS, HS) in enumerate(dataloader):
            LRHS = LRHS.type(torch.float32).to(device)
            PAN = PAN.type(torch.float32).to(device)
            HS = HS.type(torch.float32).to(device)

            ### Dual-Discriminators iterative optimization
            optimizer_D_P.zero_grad()
            if step % P_critic == 0:
                fake_HS = HS_G(LRHS, PAN)
                fake_data_pan = get_PAN(fake_HS)
                real_validity = D_P(PAN).mean()
                fake_validity = D_P(fake_data_pan.detach()).mean()
                gradient_penalty_P = compute_gradient_penalty(D_P, PAN, fake_data_pan)
                loss_D_P = fake_validity - real_validity + 10 * gradient_penalty_P
                loss_D_P.backward()
                optimizer_D_P.step()

            optimizer_D_H.zero_grad()
            if step % HS_critic == 0:
                fake_HS = HS_G(LRHS, PAN)
                fake_data_HS = filter_downsample(fake_HS.detach().cpu().numpy(),kernel_size).type(torch.float32).to(device)
                real_validity_HS = D_H(LRHS).mean()
                fake_validity_HS = D_H(fake_data_HS.detach()).mean()
                gradient_penalty_H = compute_gradient_penalty(D_H, LRHS, fake_data_HS)
                loss_D_H = fake_validity_HS - real_validity_HS + 10 * gradient_penalty_H
                loss_D_H.backward()
                optimizer_D_H.step()

            optimizer_HSG.zero_grad()
            HS_G.train()
            fake_HS = HS_G(LRHS, PAN)
            fake_data_PAN = get_PAN(fake_HS)
            fake_data_HS = filter_downsample(fake_HS.detach().cpu().numpy(), kernel_size).type(torch.float32).to(device)
            adversarial_loss_P = -1 * D_P(fake_data_PAN).mean()
            adversarial_loss_H = -1 * D_H(fake_data_HS).mean()
            loss_G = 5e-4 * adversarial_loss_P + 5e-4 * adversarial_loss_H + criteron_G(fake_HS, HS)
            loss_G.backward()
            optimizer_HSG.step()

        if epoch % batch == 0:
            evaluate(HS_G, index)
            index += 1
        ### learning rate decrease
        if (epoch + 1) % 350 == 0:
            lr /= config.lr_decay
            adjust_learning_rate(lr, optimizer_HSG)
            adjust_learning_rate(lr, optimizer_D_H)
            adjust_learning_rate(lr, optimizer_D_P)

        print("[Epoch %d/%d] [Batch %d/%d] [D_P loss: %f] [D_H loss: %f] [G loss: %f] " % (epoch, max_iter, batches_done % len(dataloader), len(dataloader), loss_D_P.item(), loss_D_H.item(), loss_G.item()))
        ### model save
        if epoch % 20 == 0:
            state = {"net": HS_G.state_dict()}
            torch.save(state, 'model/netG_epoch_%d_gpu.pth' % (epoch))
            if epoch % 100 == 0:
                torch.save(D_P.state_dict(), 'model/netD_P_epoch_%d_gpu.pth' % (epoch))
                torch.save(D_H.state_dict(), 'model/netD_H_epoch_%d_gpu.pth' % (epoch))
                torch.save(optimizer_HSG.state_dict(), 'model/optimizerG_epoch_%d_gpu.pth' % (epoch))
                torch.save(optimizer_D_P.state_dict(), 'model/optimizerD_P_epoch_%d_gpu.pth' % (epoch))
                torch.save(optimizer_D_H.state_dict(), 'model/optimizerD_H_epoch_%d_gpu.pth' % (epoch))


def model_save(epoch, model):
    state = {"net":model.state_dict()}
    torch.save(state,"{0}.mdl".format(epoch))

def evaluate(model,index):
    model.eval()
    with torch.no_grad():
        test_db = HS_Dataload("data","test", 512)
        test_loader = DataLoader(test_db, batch_size= 1, shuffle=False,num_workers = 16, pin_memory = True)
        with torch.no_grad():
            for step, (PAN, LRHS, HS) in enumerate(test_loader):
                LRHS = LRHS.type(torch.float32).to(device)
                PAN = PAN.type(torch.float32).to(device)
                HS = HS.type(torch.float32).to(device)

                output = model(LRHS, PAN)
                if not os.path.exists("cave_SNR30//{}".format(index)):
                    os.makedirs("cave_SNR30//{}".format(index))
                filename = "cave_SNR30//{0}//{1}.mat".format(index,str("out_" + "{0}").format(step + 1))
                savemat(filename, {"data": output.detach().cpu().numpy()})
            print("save success!!!!")


if __name__ == "__main__":
    ## training param set
    train(max_iter = 1000, batchsz = 8, P_critic = 2,HS_critic = 3,batch = 20)