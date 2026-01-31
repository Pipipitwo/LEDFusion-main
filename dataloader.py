import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
import cv2
import os.path as osp
from torchvision import transforms
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


to_tensor = transforms.Compose([transforms.ToTensor()])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(1143)

def populate_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "*.png")
    train_list = image_list_lowlight
    random.shuffle(train_list)
    return train_list

class lowlight_loader(data.Dataset):
    def __init__(self, lowlight_images_path):
        self.train_list = populate_train_list(lowlight_images_path) 
        self.size = 256
        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]
        data_lowlight = Image.open(data_lowlight_path)
        data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
        data_lowlight = (np.asarray(data_lowlight)/255.0) 
        data_lowlight = torch.from_numpy(data_lowlight).float()
        return data_lowlight.permute(2,0,1)

    def __len__(self):
        return len(self.data_list)

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames

class fusion_dataset_loader(data.Dataset):
    def __init__(self, split, ir_path=None, vi_path=None):
        super(fusion_dataset_loader, self).__init__()
        self.size = 225 # patch_size
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        if split == 'train':
            data_dir_vis = os.path.join(os.getcwd(), 'train/vi')
            data_dir_ir = os.path.join(os.getcwd(), 'train/ir')
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split=='train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis=Image.open(vis_path)
            image_vis=image_vis.resize((self.size,self.size), Image.Resampling.LANCZOS)
            image_vis = np.array(image_vis)
            image_vis = (np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1))/255.0)

            image_inf=Image.open(ir_path)
            image_inf=image_inf.resize((self.size,self.size), Image.Resampling.LANCZOS)
            image_inf = np.array(image_inf)
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            image_vis = torch.tensor(image_vis)
            image_ir = torch.tensor(image_ir)
            return (image_vis,image_ir,)
        
    def __len__(self):
        return self.length

class enhance_dataset_loader(data.Dataset):
    def __init__(self, split, ir_path=None, vi_path=None):
        super(enhance_dataset_loader, self).__init__()
        self.size=224 # patch_size
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        if split == 'train':
            data_dir_vis = os.path.join(os.getcwd(), 'train/vi')
            data_dir_ir = os.path.join(os.getcwd(), 'train/ir')
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))
    def __getitem__(self, index):
        if self.split == 'train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis=Image.open(vis_path)
            image_vis=image_vis.resize((self.size,self.size), Image.LANCZOS)
            image_vis = np.array(image_vis)
            image_vis = (np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1))/255.0)
            image_inf=Image.open(ir_path).convert('L')##1.28
            image_inf=image_inf.resize((self.size,self.size), Image.LANCZOS)
            image_inf = np.array(image_inf)
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            return (torch.tensor(image_vis),torch.tensor(image_ir),)
    def __len__(self):
        return self.length



# for LAN test and eval
class enhance_dataset_loader_test(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        temp_path = os.path.join(data_dir,'vi')
        self.vis_path=temp_path
        self.name_list = os.listdir(self.vis_path)
        self.transform = transform
    def __getitem__(self, index):
        name = self.name_list[index]
        image_vis = Image.open(os.path.join(self.vis_path, name))
        image_vis = self.transform(image_vis)
        return image_vis, name
    def __len__(self):
        return len(self.name_list)
    
class fusion_dataset_loader_eval(data.Dataset):
    def __init__(self,i,data_dir,transform=to_tensor):
        super().__init__()
        dirname=os.listdir(data_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'ir':
                self.inf_path = temp_path
            elif sub_dir == 'vi_en':
                self.vis_path=osp.join(temp_path,str(i+1))
        self.name_list = os.listdir(self.inf_path)
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]


        inf_image = Image.open(os.path.join(self.inf_path, name))

        vis_image = Image.open(os.path.join(self.vis_path, name))
        ir_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        return vis_image, ir_image, name

    def __len__(self):
        return len(self.name_list)

class fusion_dataset_loader_test(data.Dataset):
    def __init__(self,data_dir,transform=to_tensor):
        super().__init__()
        dirname=os.listdir(data_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'ir':
                self.inf_path = temp_path
            elif sub_dir == 'vi':
                self.vis_path=osp.join(temp_path)
        self.name_list = os.listdir(self.inf_path)
        self.transform = transform
    def __getitem__(self,index):
        name = self.name_list[index]
        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')
        vis_image = Image.open(os.path.join(self.vis_path, name))
        ir_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        return vis_image, ir_image, name
    def __len__(self):
        return len(self.name_list)

def rgb2ycbcr(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y=torch.clamp(Y, min=0., max=1.0)
    Cr=torch.clamp(Cr, min=0., max=1.0)
    Cb=torch.clamp(Cb, min=0., max=1.0)
    Y = torch.unsqueeze(Y, 1)#升维
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    # temp = torch.cat((Y, Cr, Cb), dim=1) CPU版本
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (   temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,)
        .transpose(1, 3)
        .transpose(2, 3))
    return out

def ycbcr2rgb(input_im):
    B, C, W, H = input_im.shape
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]])
    bias = torch.tensor([0.0 / 255, -0.5, -0.5])
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3).cuda()
    out = torch.clamp(out, min=0., max=1.0)
    return out

def clahe(image, batch_size):
    image = image.cpu().detach().numpy()
    results = []
    for i in range(batch_size):
        img = np.squeeze(image[i:i+1, :, :, :])
        out = np.array(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX), dtype='uint8')
        #clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(4, 4))
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
        result = clahe.apply(out)[np.newaxis][np.newaxis]
        results.append(result)
    results = np.concatenate(results, axis=0)
    image_hist = (results / 255.0).astype(np.float32)
    image_hist = torch.from_numpy(image_hist).cuda()
    return image_hist


class UnsupervisedStructureDataset(data.Dataset):
    def __init__(self, split, ir_path=None, vi_path=None):
        super(UnsupervisedStructureDataset, self).__init__()
        self.size=224 # patch_size
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        if split == 'train':
            data_dir_vis = os.path.join(os.getcwd(), 'train/vi')
            data_dir_ir = os.path.join(os.getcwd(), 'train/ir')
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))
    def __getitem__(self, index):
        if self.split == 'train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis=Image.open(vis_path)
            image_vis=image_vis.resize((self.size,self.size), Image.ANTIALIAS)
            image_vis = np.array(image_vis)
            image_vis = (np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1))/255.0)
            image_inf=Image.open(ir_path)
            image_inf=image_inf.resize((self.size,self.size), Image.ANTIALIAS)
            image_inf = np.array(image_inf)
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            return (torch.tensor(image_vis),torch.tensor(image_ir),)
    def __len__(self):
        return self.length



# for LAN test and eval
class UnsupervisedStructureDataset_test(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        temp_path = os.path.join(data_dir,'vi')
        self.vis_path=temp_path
        self.name_list = os.listdir(self.vis_path)
        self.transform = transform
    def __getitem__(self, index):
        name = self.name_list[index]
        image_vis = Image.open(os.path.join(self.vis_path, name))
        image_vis = self.transform(image_vis)
        return image_vis, name
    def __len__(self):
        return len(self.name_list)





class PairedImageDataset(data.Dataset):
    def __init__(self, split, image_size=224, base_dir='.'):
        super(PairedImageDataset, self).__init__()
        self.image_size = image_size
        self.split = split
        assert self.split in ['train', 'val'], 'split must be "train" or "val"'


        data_dir_vis = os.path.join(base_dir, self.split, 'vi')
        data_dir_ir = os.path.join(base_dir, self.split, 'ir')
        self.filepath_vis, _ = prepare_data_path(data_dir_vis)
        self.filepath_ir, _ = prepare_data_path(data_dir_ir)
        self.length = min(len(self.filepath_vis), len(self.filepath_ir))
        if self.length == 0:
            raise ValueError(f"No matching image pairs found.")
        print(f"Found {self.length} image pairs for '{self.split}' split.")

    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]

        image_vis = Image.open(vis_path).convert('RGB')
        image_ir = Image.open(ir_path).convert('L')


        image_vis = TF.resize(image_vis, [self.image_size, self.image_size], antialias=True)
        image_ir = TF.resize(image_ir, [self.image_size, self.image_size], antialias=True)


        if self.split == 'train':

            if random.random() > 0.5:
                image_vis = TF.hflip(image_vis)
                image_ir = TF.hflip(image_ir)

            angle = random.uniform(-10, 10)
            image_vis = TF.rotate(image_vis, angle)
            image_ir = TF.rotate(image_ir, angle)


            image_vis = transforms.ColorJitter(brightness=0.2, contrast=0.2)(image_vis)


        image_vis_tensor = TF.to_tensor(image_vis)
        image_ir_tensor = TF.to_tensor(image_ir)

        return image_vis_tensor, image_ir_tensor

    def __len__(self):
        return self.length