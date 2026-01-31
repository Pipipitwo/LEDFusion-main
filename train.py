
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch.autograd import Variable
import argparse
import datetime
import time
import math
import logging
import os.path as osp
import torch
import torch.nn as nn
import dataloader
import model
#import model_DEConv
import enhancement_loss
from fusion_loss import fusionloss, final_ssim
from logger import setup_logger
from torch.utils.data import DataLoader

from dataloader import rgb2ycbcr, ycbcr2rgb
from torchvision import transforms
from tqdm import tqdm
from thop import profile
import warnings
from torch.utils.tensorboard import SummaryWriter
from LEN.model import enhance_net_nopool
import gc

from deconv import Conv2d_cd, Conv2d_ad, Conv2d_rd, Conv2d_hd, Conv2d_vd
torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings('ignore')

BASE_MODEL_DIR = './model_1'


def weights_init(m):

    if isinstance(m, (Conv2d_cd, Conv2d_ad, Conv2d_hd, Conv2d_vd)):

        m.conv.weight.data.normal_(0.0, 0.02)
        if m.conv.bias is not None:

            m.conv.bias.data.fill_(0)


    elif isinstance(m, nn.Conv2d):

        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)


    elif isinstance(m, nn.BatchNorm2d):

        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_fusion(i, logger=None):

    modelpth = osp.join( BASE_MODEL_DIR , str(i + 1))
    os.makedirs(modelpth, mode=0o777, exist_ok=True)


    image_vis_dir = osp.join(modelpth, 'image_vis')
    vi_en_y_dir = osp.join(modelpth, 'vi_en_y')
    image_vis_en_dir = osp.join(modelpth, 'image_vis_en')
    os.makedirs(image_vis_dir, mode=0o777, exist_ok=True)
    os.makedirs(vi_en_y_dir, mode=0o777, exist_ok=True)
    os.makedirs( image_vis_en_dir, mode=0o777, exist_ok=True)


    writer = SummaryWriter(log_dir=osp.join(modelpth, 'tensorboard_logs'))
    info_log_path = osp.join(modelpth, 'fusion_hyperparameters.txt')
    lr_start = 1e-4


    fusion_batch_size = 5
    n_workers = 4
    ds = dataloader.fusion_dataset_loader('train')
    dl = torch.utils.data.DataLoader(ds, batch_size=fusion_batch_size, shuffle=True, num_workers=n_workers,
                                     pin_memory=False)

    net =model.FusionNet()
    if i == 0:
        net.apply(weights_init)
    if i > 0:

        load_path = osp.join(BASE_MODEL_DIR, str(i), 'fusion_model.pth')
        net.load_state_dict(torch.load(load_path))
        print('Load Pre-trained Fusion Model:{}!'.format(load_path))
    net.cuda()
    net.eval()
    net.train()

    enhancemodel = enhance_net_nopool().cuda()
    optim = torch.optim.Adam(net.parameters(), lr=lr_start, weight_decay=0.0001)
    criteria_fusion = fusionloss()
    st = glob_st = time.time()
    epoch = 20
    grad_step = 5.0
    dl.n_iter = len(dl)

    for epo in range(0, epoch):
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** ((epo / 5) + 1)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_epo

        for it, (image_vis, image_ir) in enumerate(dl):
            net.train()
            image_vis = Variable(image_vis, requires_grad=True).cuda()
            _, image_vis_en, _ = enhancemodel(image_vis)
            image_vis_en_ycbcr = rgb2ycbcr(image_vis_en)
            image_ir = Variable(image_ir, requires_grad=True).cuda()
            vi_en_y = dataloader.clahe(image_vis_en_ycbcr[:, 0:1, :, :], image_vis_en_ycbcr.shape[0])
            _, _, _, Y_f = net(vi_en_y, image_ir)

            fusion_ycbcr = torch.cat((Y_f, image_vis_en_ycbcr[:, 1:2, :, :], image_vis_en_ycbcr[:, 2:, :, :]), dim=1)
            I_f = ycbcr2rgb(fusion_ycbcr)
            ones = torch.ones_like(I_f)
            zeros = torch.zeros_like(I_f)
            I_f = torch.where(I_f > ones, ones, I_f)
            I_f = torch.where(I_f < zeros, zeros, I_f)



            weight_color =0.05
            weight_grad = 10
            weight_image = 120

            ssim_loss_weight = 30
            loss_fusion, loss_image, loss_grad, loss_color = criteria_fusion(vi_en_y, image_vis, image_ir, Y_f, I_f,weight_image,weight_grad,weight_color )
            ssim_loss = 0
            ssim_loss_temp = 1 - final_ssim(image_ir, vi_en_y, Y_f)
            ssim_loss += ssim_loss_temp
            ssim_loss /= len(Y_f)

            loss_fusion = loss_fusion + ssim_loss_weight * ssim_loss
            loss_fusion.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            '''
            
        
            if epo % SAVE_EPOCH_FREQ == 0 and it == 0:
                print(f"\n--- Saving preview images for Epoch {epo} ---") 

               
                for idx in range(min(IMAGES_TO_SAVE, image_vis.size(0))):
                    img = image_vis[idx].detach().cpu()
                    img = img.clamp(0, 1)
                    img = transforms.ToPILImage()(img)
                    img_path = osp.join(image_vis_dir, f'epo_{epo}_idx_{idx}.png')
                    img.save(img_path)

                for idx in range(min(IMAGES_TO_SAVE, image_vis_en.size(0))):
                    img = image_vis_en[idx].detach().cpu()
                    img = img.clamp(0, 1)
                    img = transforms.ToPILImage()(img)
                    img_path = osp.join(image_vis_en_dir, f'epo_{epo}_idx_{idx}.png')
                    img.save(img_path)

               
                for idx in range(min(IMAGES_TO_SAVE, vi_en_y.size(0))):
                    img = vi_en_y[idx].detach().cpu()
                    img = img.repeat(3, 1, 1) 
                    img = img.clamp(0, 1)
                    img = transforms.ToPILImage()(img)
                    img_path = osp.join(vi_en_y_dir, f'epo_{epo}_idx_{idx}.png')
                    img.save(img_path)
            # ==============================================================================
            '''
            if grad_step > 1:
                loss_fusion = loss_fusion / grad_step
            if (it + 1) % grad_step == 0:
                optim.step()
                optim.zero_grad()
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = dl.n_iter * epo + it + 1

            if (it + 1) % 50 == 0:
                writer.add_scalar('Fusion/Total_Loss', loss_fusion.item(), now_it)
                writer.add_scalar('Fusion/Image_Loss', loss_image.item(), now_it)
                writer.add_scalar('Fusion/Gradient_Loss', loss_grad.item(), now_it)
                writer.add_scalar('Fusion/Color_Loss', loss_color.item(), now_it)
                writer.add_scalar('Fusion/SSIM_Loss', ssim_loss.item(), now_it)

            if (it + 1) % 50 == 0:
                lr = optim.param_groups[0]['lr']
                eta = int((dl.n_iter * epoch - now_it) * (glob_t_intv / (now_it)))
                eta = str(datetime.timedelta(seconds=eta))
                msg = ', '.join(
                    ['step: {it}/{max_it}',
                     'loss_fusion:{loss_fusion:.4f}\n',
                     'loss_image: {loss_image:.4f}',
                     'loss_grad: {loss_grad:4f}',
                     'loss_color: {loss_color:4f}',
                     'loss_ssim:{loss_ssim:4f}',
                     'eta: {eta}',
                     'time: {time:.4f}', ]).format(
                    it=now_it, max_it=dl.n_iter * epoch, lr=lr,
                    loss_fusion=loss_fusion.item(), loss_image=loss_image.item(),
                    loss_grad=loss_grad.item(), loss_color=loss_color.item(),
                    loss_ssim=ssim_loss.item(), eta=eta, time=t_intv, )
                logger.info(msg)
                st = ed
    with open(info_log_path, 'w') as f:
        f.write(f"--- Fusion Model Training Hyperparameters for Iteration {i + 1} ---\n")
        f.write(f"initial_learning_rate: {lr_start}\n")
        f.write(f"ssim_loss_weight: {ssim_loss_weight}\n")#weight_color
        f.write(f"weight_color: {weight_color}\n")
        f.write(f"weight_color: {weight_grad}\n")
        f.write(f"weight_color: {weight_image}\n")
        f.write("-" * 20 + "\n")

    writer.close()

    save_pth = osp.join(modelpth, 'fusion_model.pth')
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state, save_pth)
    logger.info('Fusion Model Training done~, The Model is saved to: {}'.format(save_pth))
    logger.info('\n')


def train_enhancement(num, logger=None):
    lr_start = 2 * 1e-5

    modelpth = osp.join(BASE_MODEL_DIR, str(num + 1))
    os.makedirs(modelpth, mode=0o777, exist_ok=True)

    writer = SummaryWriter(log_dir=osp.join(modelpth, 'tensorboard_logs'))
    info_log_path = osp.join(modelpth, 'enhancement_hyperparameters.txt')
    '''
    with open(info_log_path, 'w') as f:
        f.write(f"--- Enhancement Model Training Hyperparameters for Iteration {num + 1} ---\n")
        f.write(f"initial_learning_rate: {lr_start}\n")
        f.write(f"loss_TV_weight: 200\n")
        f.write(f"loss_TV_weight: 20\n")
        f.write(f"loss_spa_weight: 1\n")
        f.write(f"loss_col_weight: 5\n")
        f.write(f"loss_exp_weight: 10\n")
        if num > 0:
            illu_loss_weight = 0.1 * num
            f.write(f"illu_loss_weight: {illu_loss_weight:.2f}\n")
        f.write("-" * 20 + "\n")
    '''
    enhancemodel = enhance_net_nopool().cuda()
    enhancemodel.apply(weights_init)
    enhancemodel.eval()
    enhancemodel.train()
    optimizer = torch.optim.Adam(enhancemodel.parameters(), lr=lr_start, weight_decay=0.0001)

    if num > 0:
        fusionmodel =model.FusionNet()

        if logger == None:
            logger = logging.getLogger()
            setup_logger(modelpth)


        fusionmodel_path = osp.join(BASE_MODEL_DIR, str(num), 'fusion_model.pth')
        fusionmodel.load_state_dict(torch.load(fusionmodel_path), False)


        fusionmodel = fusionmodel.cuda()

        fusionmodel.eval()

        for q in fusionmodel.parameters():
            q.requires_grad = False

    datas = dataloader.enhance_dataset_loader('train')
    datal = torch.utils.data.DataLoader(datas, batch_size=4, shuffle=True, num_workers=4, pin_memory=False)
    print("the training dataset is length:{}".format(datas.length))
    datal.n_iter = len(datal)

    L_color = enhancement_loss.L_color()
    L_spa = enhancement_loss.L_spa()
    L_exp = enhancement_loss.L_exp(8, 0.5)
    L_TV = enhancement_loss.L_TV()
    grad_acc_steps = 4.0
    epoch = 20
    st = glob_st = time.time()
    logger.info('Training Enhancement Model start~')
    for epo in range(0, epoch):
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** ((epo / 5) + 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo

        for it, (image_vis, image_ir) in enumerate(datal):
            enhancemodel.train()
            image_vis = Variable(image_vis, requires_grad=True).cuda()
            image_ir = Variable(image_ir, requires_grad=True).cuda()
            _, enhanced_image, A = enhancemodel(image_vis)
            loss_TV_weight=200
            loss_TV_img_weight=1e-4
            loss_spa_weight = 1
            loss_col_weight = 5
            loss_exp_weight= 10
            loss_TV = loss_TV_weight * L_TV(A)
            loss_spa = loss_spa_weight * torch.mean(L_spa(enhanced_image, image_vis))
            loss_col = loss_col_weight * torch.mean(L_color(enhanced_image))
            loss_exp = loss_exp_weight * torch.mean(L_exp(enhanced_image))
            #loss_TV_img = loss_TV_img_weight * L_TV(enhanced_image)
            loss_enhance = loss_TV + loss_spa + loss_col + loss_exp
            illu_loss = 0
            loss_illu_tensor = torch.tensor(0.0).cuda()

            if num > 0:
                image_vis_en_ycbcr = rgb2ycbcr(enhanced_image)
                vi_clahe_y = dataloader.clahe(image_vis_en_ycbcr[:, 0:1, :, :], image_vis_en_ycbcr.shape[0])
                _, _, _, Y_f = fusionmodel(vi_clahe_y, image_ir)
                fusion_ycbcr = torch.cat((Y_f, image_vis_en_ycbcr[:, 1:2, :, :], image_vis_en_ycbcr[:, 2:, :, :]),
                                         dim=1)
                fusion_image_if = ycbcr2rgb(fusion_ycbcr)
                ones, zeros = torch.ones_like(fusion_image_if), torch.zeros_like(fusion_image_if)
                fusion_image_if = torch.where(fusion_image_if > ones, ones, fusion_image_if)
                fusion_image_if = torch.where(fusion_image_if < zeros, zeros, fusion_image_if)


                loss_total = loss_enhance
            else:
                loss_total = loss_enhance

            if grad_acc_steps > 1:
                loss_total = loss_total / grad_acc_steps
            loss_total.backward()
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = datal.n_iter * epo + it + 1
            eta = str(datetime.timedelta(seconds=int((datal.n_iter * epoch - now_it) * (glob_t_intv / (now_it)))))
            if now_it % grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if now_it % 50 == 0:
                writer.add_scalar('Enhancement/Total_Loss', loss_total.item(), now_it)
                writer.add_scalar('Enhancement/Enhance_Loss', loss_enhance.item(), now_it)
                writer.add_scalar('Enhancement/TV_Loss', loss_TV.item(), now_it)
                #writer.add_scalar('Enhancement/TV_img_Loss', loss_TV_img.item(), now_it)
                writer.add_scalar('Enhancement/Spatial_Loss', loss_spa.item(), now_it)
                writer.add_scalar('Enhancement/Color_Loss', loss_col.item(), now_it)
                writer.add_scalar('Enhancement/Exposure_Loss', loss_exp.item(), now_it)
                if num > 0:
                    writer.add_scalar('Enhancement/Illumination_Loss', illu_loss, now_it)

            if now_it % 50 == 0:
                msg = ', '.join(
                    ['step: {it}/{max_it}', 'loss_total: {loss_total:.4f}', 'loss_enhance: {loss_enhance:.4f}',
                     'loss_illu: {loss_illu:.4f}', 'loss_TV: {loss_TV:.4f}', 'loss_spa: {loss_spa:.6f}',
                     'loss_col: {loss_col:.4f}', 'loss_exp: {loss_exp:.4f}', 'eta: {eta}',
                     'time: {time:.4f}', ]).format(it=now_it, max_it=datal.n_iter * epoch, loss_total=loss_total.item(),
                                                   loss_enhance=loss_enhance.item(), loss_illu=illu_loss,
                                                   loss_TV=loss_TV.item(),loss_spa=loss_spa.item(),
                                                   loss_col=loss_col.item(), loss_exp=loss_exp.item(), time=t_intv,
                                                   eta=eta, )
                logger.info(msg)
                st = ed
    with open(info_log_path, 'w') as f:
        f.write(f"--- Enhancement Model Training Hyperparameters for Iteration {num + 1} ---\n")
        f.write(f"initial_learning_rate: {lr_start}\n")
        f.write(f"loss_TV_weight: {loss_TV_weight}\n")
        #f.write(f"loss_TV_img_weight: {loss_TV_img_weight}\n")
        f.write(f"loss_spa_weight: {loss_spa_weight}\n")
        f.write(f"loss_col_weight: {loss_col_weight}\n")
        f.write(f"loss_exp_weight: {loss_exp_weight}\n")
        if num > 0:
            illu_loss_weight = 0.0 * num
            f.write(f"illu_loss_weight: {illu_loss_weight:.2f}\n")
        f.write("-" * 20 + "\n")
    writer.close()

    enhance_model_file = osp.join(modelpth, 'enhancement_model.pth')
    torch.save(enhancemodel.state_dict(), enhance_model_file)
    logger.info("Enhancement Model Save to: {}".format(enhance_model_file))
    logger.info('\n')



def run_enhance(i):  # LAN eval
    enhance_model_path = osp.join(os.getcwd(), BASE_MODEL_DIR, str(i + 1), 'enhancement_model.pth')
    enhanced_dir = osp.join(BASE_MODEL_DIR, 'eval', 'vi_en')
    os.makedirs(enhanced_dir, mode=0o777, exist_ok=True)
    enhancemodel = enhance_net_nopool().cuda()
    enhancemodel.eval()
    enhancemodel.load_state_dict(torch.load(enhance_model_path))
    print('enhancemodel,done!')

    test_dataset = dataloader.enhance_dataset_loader_test(osp.join(os.getcwd(), 'test','LLVIP'))
    test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=False,
                             drop_last=False)
    test_tqdm = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for images_vis, name in test_tqdm:
            images_vis = Variable(images_vis).cuda()
            _, enhanced_image, _ = enhancemodel(images_vis)
            enhanced_image = enhanced_image.cpu()
            for k in range(len(name)):
                image = enhanced_image[k, :, :, :]
                image = transforms.ToPILImage()(image)
                sub_dir = osp.join(enhanced_dir, str(i + 1))
                os.makedirs(sub_dir, mode=0o777, exist_ok=True)
                save_path = osp.join(sub_dir, name[k])
                image.save(save_path)


def run_fusion(i):  # RFN eval
    fusion_model_path = osp.join(BASE_MODEL_DIR, str(i + 1), 'fusion_model.pth')
    fusion_dir = osp.join(BASE_MODEL_DIR, 'eval')
    os.makedirs(fusion_dir, mode=0o777, exist_ok=True)
    fusionmodel = model.FusionNet().cuda()
    fusionmodel.eval()
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel,done!')
    testdataset = dataloader.fusion_dataset_loader_eval(i, osp.join(os.getcwd(), 'eval'))
    testloader = DataLoader(dataset=testdataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False,
                            drop_last=False)
    testtqdm = tqdm(testloader, total=len(testloader))
    with torch.no_grad():
        for images_vis, images_ir, name in testtqdm:
            images_vis = Variable(images_vis).cuda()
            images_ir = Variable(images_ir).cuda()
            image_vis_en_ycbcr = rgb2ycbcr(images_vis)
            image_vis_en_y = dataloader.clahe(image_vis_en_ycbcr[:, 0:1, :, :], image_vis_en_ycbcr.shape[0])
            _, _, _, Y_f = fusionmodel(image_vis_en_y, images_ir)
            fusion_ycbcr = torch.cat((Y_f, image_vis_en_ycbcr[:, 1:2, :, :], image_vis_en_ycbcr[:, 2:, :, :]), dim=1)
            I_f = ycbcr2rgb(fusion_ycbcr)
            ones, zeros = torch.ones_like(I_f), torch.zeros_like(I_f)
            I_f = torch.where(I_f > ones, ones, I_f)
            I_f = torch.where(I_f < zeros, zeros, I_f)
            I_f = I_f.cpu()
            for k in range(len(name)):
                image_I_f = I_f[k, :, :, :]
                image_I_f = transforms.ToPILImage()(image_I_f)
                sub_dir_1 = osp.join(fusion_dir, 'If', str(i + 1))
                os.makedirs(sub_dir_1, mode=0o777, exist_ok=True)
                save_path1 = osp.join(sub_dir_1, name[k])
                image_I_f.save(save_path1)


def run_full_evaluation(i):

    print(f"--- Running Full End-to-End Evaluation for Iteration {i + 1} ---")
    total_time = 0
    num_images = 0


    model_dir = osp.join(BASE_MODEL_DIR, str(i + 1))

    fusion_model_path = osp.join(model_dir, 'fusion_model.pth')
    enhance_model_path = osp.join(model_dir, 'enhancement_model.pth')

    fusion_dir = osp.join(BASE_MODEL_DIR, 'eval', 'enhance_fused_1.6', str(i + 1))
    os.makedirs(fusion_dir, mode=0o777, exist_ok=True)


    fusionmodel =model.FusionNet().cuda()
    fusionmodel.eval()
    fusionmodel.load_state_dict(torch.load(fusion_model_path), False)

    print(f"Loaded Fusion model from: {fusion_model_path}")

    enhancemodel = enhance_net_nopool().cuda()
    enhancemodel.eval()
    enhancemodel.load_state_dict(torch.load(enhance_model_path))
    print(f"Loaded Enhancement model from: {enhance_model_path}")

    testdataset = dataloader.fusion_dataset_loader_test(osp.join(os.getcwd(),'test/LLVIP/'))
    testloader = DataLoader(
        dataset=testdataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
    )
    testtqdm = tqdm(testloader, total=len(testloader))

    with torch.no_grad():
        for images_vis, images_ir, name in testtqdm:
            images_vis = Variable(images_vis).cuda()
            images_ir = Variable(images_ir).cuda()


            start_time = time.time()

            _, enhanced_image, _ = enhancemodel(images_vis)
            image_vis_en_ycbcr = rgb2ycbcr(enhanced_image)
            image_vis_en_y = dataloader.clahe(image_vis_en_ycbcr[:, 0:1, :, :], image_vis_en_ycbcr.shape[0])
            _, _, _, Y_f = fusionmodel(image_vis_en_y, images_ir)
            fusion_ycbcr = torch.cat((Y_f, image_vis_en_ycbcr[:, 1:2, :, :], image_vis_en_ycbcr[:, 2:, :, :]), dim=1)
            I_f = ycbcr2rgb(fusion_ycbcr)

            end_time = time.time()
            total_time += (end_time - start_time)
            num_images += len(name)


            ones, zeros = torch.ones_like(I_f), torch.zeros_like(I_f)
            I_f = torch.where(I_f > ones, ones, I_f)
            I_f = torch.where(I_f < zeros, zeros, I_f)
            I_f = I_f.cpu()

            for k in range(len(name)):
                image_I_f = I_f[k, :, :, :]
                image_I_f = transforms.ToPILImage()(image_I_f)
                save_path = osp.join(fusion_dir, name[k])
                image_I_f.save(save_path)

    if num_images > 0:
        average_time = total_time / num_images
        print(f"Processed {num_images} images.")
        print(f"Average time per image: {average_time:.4f} s")
    else:
        print("No images were processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    logpath = osp.join(BASE_MODEL_DIR, 'logs')
    logger = logging.getLogger()
    setup_logger(logpath)


    for i in range(0, 5):
        train_enhancement(i, logger)
        print("|{0} Train LAN Sucessfully~!".format(i + 1))
        run_enhance(i)
        print("|{0} Enhancement Image Sucessfully~!".format(i + 1))
        torch.cuda.empty_cache()
        train_fusion(i, logger)
        print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
        gc.collect()
        torch.cuda.empty_cache()
        run_fusion(i)
        print("{0} Fusion Image Sucessfully~!".format(i + 1))
        run_full_evaluation(i)
        print(f" Iteration {i + 1} | Full Pipeline Evaluation Successfully~!")
        print("———————————————————————————")

    print("Training Done!")

