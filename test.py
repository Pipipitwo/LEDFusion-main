import os
from torch.autograd import Variable
import argparse
import time
import logging
import os.path as osp
import torch
import dataloader  # Assuming this is a custom module
import model # Assuming this is a custom module############这里要换模型
from logger import setup_logger  # Assuming this is a custom module
from torch.utils.data import DataLoader
from dataloader import rgb2ycbcr, ycbcr2rgb, clahe  # Assuming these are in your dataloader
from torchvision import transforms
from tqdm import tqdm
from thop import profile
import warnings
from LEN.model import enhance_net_nopool#导入的文件名不能有-，会搜索不到
#from model_DAconv_DEConv_AFM_CBAM_CA_4 import luminance_adjustment#导入的文件名不能有-，会搜索不到
warnings.filterwarnings('ignore')

model=model
# --- Centralized Path Management ---
class Config:
    def __init__(self):
        #self.base_dir = os.getcwd() # Current working directory
        dataset=''
        num='5'
        self.base_dir = '/home/jnu733/LDNFusion'
        self.checkpoint_dir = osp.join(self.base_dir, 'model',num)
        self.data_base_dir = '/home/jnu733/LENFsuion-master/train_1' # Root for your TNO datasets
        self.tno_24_data_dir = osp.join(self.data_base_dir, dataset)
        self.output_base_dir = '/home/jnu733/yolov5-master/datasets/LLVIP/images/train' # Root for your fusion outputs

        # Subdirectories for outputs
        self.enhanced_output_dir = osp.join(self.output_base_dir, 'vi_en', num)
        self.fusion_output_if_dir = osp.join(self.output_base_dir, 'If',num) # For GPU fusion
        self.fusion_output_cpu_dir = osp.join(self.output_base_dir, 'enhance_fused', num) # For CPU fusion

        # Model paths
        self.enhancement_model_path = osp.join(self.checkpoint_dir, 'enhancement_model.pth')
        #self.enhancement_model_path = '/home/jnu733/Zero-DCE-master/Zero-DCE_code/snapshots/Epoch99.pth'
        self.fusion_model_path_base = osp.join(self.checkpoint_dir, 'fusion_model.pth') # For CPU fusion, it seems to iterate based on 'i'

        # Ensure output directories exist
        os.makedirs(self.enhanced_output_dir, mode=0o777, exist_ok=True)
        os.makedirs(self.fusion_output_if_dir, mode=0o777, exist_ok=True)
        os.makedirs(self.fusion_output_cpu_dir, mode=0o777, exist_ok=True)

# Instantiate the config
cfg = Config()

def run_enhance_gpu():
    total_time1 = 0
    ###########这里要改
    #enhancemodel = model.luminance_adjustment().cuda()
    #enhancemodel =model_DAconv_DEConv_AFM_CBAM_CA_4.luminance_adjustment().cuda()
    enhancemodel = enhance_net_nopool().cuda()
    enhancemodel.load_state_dict(torch.load(cfg.enhancement_model_path))

    test_dataset = dataloader.enhance_dataset_loader_test(cfg.tno_24_data_dir)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    test_tqdm = tqdm(test_loader, total=len(test_loader), desc="Enhancing (GPU)")

    with torch.no_grad():
        for images_vis, name in test_tqdm:
            images_vis = images_vis.cuda()
            start_time = time.time()
            _, enhanced_image, _ = enhancemodel(images_vis)
            enhanced_image = enhanced_image.cpu().numpy()

            for k in range(len(name)):
                image = enhanced_image[k, :, :, :]
                image = image.squeeze()
                image = transforms.ToPILImage()(torch.tensor(image))
                save_path = os.path.join(cfg.enhanced_output_dir, name[k])
                image.save(save_path)
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_time1 += elapsed_time

    average_time1 = total_time1 / len(test_loader) # Use actual number of samples
    print(f"Average enhancement time (GPU): {average_time1:.4f} s")


def run_enhance_cpu():
    total_time1 = 0
    #enhancemodel = model_DAconv_DEConv_AFM_CBAM_CA_4.luminance_adjustment()
    enhancemodel = enhance_net_nopool().cuda()
    enhancemodel.eval()
    enhancemodel.load_state_dict(torch.load(cfg.enhancement_model_path, map_location=torch.device('cpu')))

    test_dataset = dataloader.enhance_dataset_loader_test(cfg.tno_24_data_dir)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    test_tqdm = tqdm(test_loader, total=len(test_loader), desc="Enhancing (CPU)")

    with torch.no_grad():
        for images_vis, name in test_tqdm:
            start_time = time.time()
            _, enhanced_image, _ = enhancemodel(images_vis)
            enhanced_image = enhanced_image.cpu().numpy()

            for k in range(len(name)):
                image = enhanced_image[k, :, :, :]
                image = image.squeeze()
                image = transforms.ToPILImage()(torch.tensor(image))
                save_path = os.path.join(cfg.enhanced_output_dir, name[k])
                image.save(save_path)
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_time1 += elapsed_time

    average_time1 = total_time1 / len(test_loader) # Use actual number of samples
    print(f"Average enhancement time (CPU): {average_time1:.4f} s")


def run_fusion_gpu():
    total_time = 0

    fusionmodel = model.FusionNet().cuda()
    fusionmodel.load_state_dict(torch.load(osp.join(cfg.checkpoint_dir, 'fusion_model.pth')), strict=True)
    fusionmodel.eval()

    enhancemodel = enhance_net_nopool().cuda()
    enhancemodel.eval()
    enhancemodel.load_state_dict(torch.load(cfg.enhancement_model_path))

    # 1. 确保路径与 Script 1 完全一致
    # 建议打印一下 cfg.tno_24_data_dir 确认是否指向 test/LLVIP
    test_dataset = dataloader.fusion_dataset_loader_test(cfg.tno_24_data_dir)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    test_tqdm = tqdm(test_loader, total=len(test_loader), desc="Fusing (GPU)")

    with torch.no_grad():
        for images_vis, images_ir, name in test_tqdm:
            images_vis = images_vis.cuda()
            images_ir = images_ir.cuda()
            start_time = time.time()

            _, enhanced_image, _ = enhancemodel(images_vis)
            image_vis_en_ycbcr = rgb2ycbcr(enhanced_image)
            image_vis_en_y = clahe(image_vis_en_ycbcr[:, 0:1, :, :], images_vis.shape[0])


            _, _, _, Y_f = fusionmodel(image_vis_en_y, images_ir)
            fusion_ycbcr = torch.cat((Y_f, image_vis_en_ycbcr[:, 1:2, :, :], image_vis_en_ycbcr[:, 2:, :, :]), dim=1)
            I_f = ycbcr2rgb(fusion_ycbcr)


            ones = torch.ones_like(I_f)
            zeros = torch.zeros_like(I_f)
            I_f = torch.where(I_f > ones, ones, I_f)
            I_f = torch.where(I_f < zeros, zeros, I_f)

            I_f = I_f.cpu()

            for k in range(len(name)):
                image_I_f = I_f[k, :, :, :]
                # image_I_f = image_I_f.squeeze()
                image_I_f = transforms.ToPILImage()(image_I_f)
                save_path = os.path.join(cfg.fusion_output_if_dir, name[k])
                image_I_f.save(save_path)

                end_time = time.time()
                elapsed_time = end_time - start_time
                total_time += elapsed_time

    average_time = total_time / len(test_loader)
    print(f"Average fusion time (GPU): {average_time:.4f} s")


def run_fusion_cpu():
    total_time = 0
    # The original code had an 'i' loop here for fusion_model_path, assuming it's for a single model now or needs an 'i' from outer scope if it's a loop.
    # For now, I'll hardcode '1' for the example. If you need a loop, you'd iterate `i` outside this function.
    # i = 0 # Placeholder for the 'i' in the original path, if you need a specific iteration's model
    fusion_model_path_cpu = cfg.fusion_model_path_base # Assuming this now points to the correct fusion_model.pth for the single test
    fusionmodel = model.FusionNet()
    fusionmodel.eval()
    fusionmodel.load_state_dict(torch.load(fusion_model_path_cpu, map_location=torch.device('cpu')), strict=True)

    enhancemodel = enhance_net_nopool()
    enhancemodel.eval()
    enhancement_model_path_cpu = cfg.enhancement_model_path
    enhancemodel.load_state_dict(torch.load(enhancement_model_path_cpu, map_location=torch.device('cpu')))


    test_dataset = dataloader.fusion_dataset_loader_test(cfg.tno_24_data_dir)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    test_tqdm = tqdm(test_loader, total=len(test_loader), desc="Fusing (CPU)")

    with torch.no_grad():
        for images_vis, images_ir, name in test_tqdm:

            images_vis = images_vis.to(torch.device('cpu'))
            images_ir = images_ir.to(torch.device('cpu'))

            start_time = time.time()


            _, enhanced_image, _ = enhancemodel(images_vis)
            image_vis_en_ycbcr = rgb2ycbcr(enhanced_image)

            image_vis_en_y = clahe(image_vis_en_ycbcr[:, 0:1, :, :], images_vis.shape[0])



            _, _, _, Y_f = fusionmodel(image_vis_en_y, images_ir)

            # image_vis_en_ycbcr = image_vis_en_ycbcr.to(torch.device('cpu'))
            fusion_ycbcr = torch.cat((Y_f, image_vis_en_ycbcr[:, 1:2, :, :], image_vis_en_ycbcr[:, 2:, :, :]), dim=1)
            I_f = ycbcr2rgb(fusion_ycbcr)


            ones = torch.ones_like(I_f)
            zeros = torch.zeros_like(I_f)
            I_f = torch.where(I_f > ones, ones, I_f)
            I_f = torch.where(I_f < zeros, zeros, I_f)



            for k in range(len(name)):
                image_I_f = I_f[k, :, :, :]

                image_I_f = transforms.ToPILImage()(image_I_f)
                save_path = os.path.join(cfg.fusion_output_cpu_dir, name[k])
                image_I_f.save(save_path)
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_time += elapsed_time

    average_time = total_time / len(test_loader) # Use actual number of samples
    print(f"Average fusion time (CPU): {average_time:.4f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test for enhancement and fusion')
    parser.add_argument('--batch_size', '-B', type=int, default=8)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU for inference (default is False)') # Changed default to False for clarity if not specified

    args = parser.parse_args()


    logpath = osp.join(cfg.checkpoint_dir, 'logs')
    setup_logger(logpath)
    logger = logging.getLogger()

    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("Running on GPU...")
        run_enhance_gpu()
        print("|Enhancement Image Sucessfully~!")
        run_fusion_gpu()
        print("Fusion Image Sucessfully~!")
    else:

        print("Running on CPU...")
        run_enhance_cpu()
        print("|Enhancement Image Sucessfully~!")
        run_fusion_cpu()
        print("Fusion Image Sucessfully~!")
    print("————————————————————————————————————————————")
    print("Test Done!")