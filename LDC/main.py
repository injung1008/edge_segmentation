
from __future__ import print_function

import argparse
import os
import time, platform

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from thop import profile

from dataset import DATASET_NAMES, BipedDataset, dataset_info
from torch.utils.data import Dataset

from modelB4 import LDC


IS_LINUX = True if platform.system()=="Linux" else False


class TestDataset(Dataset):
    def __init__(self,
                 data_root,
                 test_data,
                 mean_bgr,
                 img_height,
                 img_width,
                 test_list=None,
                 arg=None
                 ):


        self.data_root = data_root
        self.test_data = test_data
        self.test_list = test_list

        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.data_index = self._build_index()

        print(f"mean_bgr: {self.mean_bgr}")

    def _build_index(self):
        sample_indices = []
        if self.test_data == "CLASSIC":
            # for single image testing
#             images_path = os.listdir(self.data_root)
            images_path = self.data_root
            labels_path = None
            sample_indices = [images_path, labels_path]
            print('sample_indices',sample_indices)
       
        return sample_indices

    def __len__(self):
        return 1 if self.test_data.upper() == 'CLASSIC' else len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        # image_path, label_path = self.data_index[idx]
        if self.data_index[1] is None:
            image_path = self.data_index[0] if len(self.data_index[0]) > 1 else self.data_index[0]
            print(f'************ image_path {image_path}')
        else:
            image_path = self.data_index[idx][0]
        label_path = None if self.test_data == "CLASSIC" else self.data_index[idx][1]
#         img_name = os.path.basename(image_path)
#         file_name = os.path.splitext(img_name)[0] + ".png"
        img_name = self.data_root.split('/')
        img_name = img_name[-1]
        file_name = os.path.splitext(img_name)[0] + ".png"
        


        gt_dir = None


        image = cv2.imread(self.data_root, cv2.IMREAD_COLOR)

        if not self.test_data == "CLASSIC":
            label = cv2.imread(os.path.join(
                gt_dir, label_path), cv2.IMREAD_COLOR)
        else:
            label = None
        
        im_shape = [image.shape[0], image.shape[1]]
        image, label = self.transform(img=image, gt=label)

        return dict(images=image, labels=label, file_names=file_name, image_shape=im_shape)

    def transform(self, img, gt):
        # gt[gt< 51] = 0 # test without gt discrimination
        if self.test_data == "CLASSIC":
            img_height = self.img_height
            img_width = self.img_width
            print(
                f"actual size: {img.shape}, target size: {(img_height, img_width,)}")
            # img = cv2.resize(img, (self.img_width, self.img_height))
            img = cv2.resize(img, (img_width, img_height))
            gt = None

        # Make images and labels at least 512 by 512
        elif img.shape[0] < 512 or img.shape[1] < 512:
            img = cv2.resize(img, (self.img_width, self.img_height))  # 512
            gt = cv2.resize(gt, (self.img_width, self.img_height))  # 512

        # Make sure images and labels are divisible by 2^4=16
        elif img.shape[0] % 16 != 0 or img.shape[1] % 16 != 0:
            img_width = ((img.shape[1] // 16) + 1) * 16
            img_height = ((img.shape[0] // 16) + 1) * 16
            img = cv2.resize(img, (img_width, img_height))
            gt = cv2.resize(gt, (img_width, img_height))
        else:
            img_width = self.img_width
            img_height = self.img_height
            img = cv2.resize(img, (img_width, img_height))
            gt = cv2.resize(gt, (img_width, img_height))
        # # For FPS
        # img = cv2.resize(img, (496,320))
        # if self.yita is not None:
        #     gt[gt >= self.yita] = 1
        img = np.array(img, dtype=np.float32)
        # if self.rgb:
        #     img = img[:, :, ::-1]  # RGB->BGR
        img -= self.mean_bgr
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        if self.test_data == "CLASSIC":
            gt = np.zeros((img.shape[:2]))
            gt = torch.from_numpy(np.array([gt])).float()
        else:
            gt = np.array(gt, dtype=np.float32)
            if len(gt.shape) == 3:
                gt = gt[:, :, 0]
            gt /= 255.
            gt = torch.from_numpy(np.array([gt])).float()

        return img, gt

def image_normalization(img, img_min=0, img_max=255,
                        epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img

def save_image_batch_to_disk(tensor, output_dir, file_names, img_shape=None,  is_inchannel=False):

    os.makedirs(output_dir, exist_ok=True)
    is_testing = True
   

    fuse_name = 'fused'
    av_name = 'avg'
    tensor2=None
    tmp_img2 = None

    output_dir_f = os.path.join(output_dir, fuse_name)
    output_dir_a = os.path.join(output_dir, av_name)
    os.makedirs(output_dir_f, exist_ok=True)
    os.makedirs(output_dir_a, exist_ok=True)

    # 255.0 * (1.0 - em_a)
    edge_maps = []
    for i in tensor:
        tmp = torch.sigmoid(i).cpu().detach().numpy()
        edge_maps.append(tmp)
    tensor = np.array(edge_maps)
    # print(f"tensor shape: {tensor.shape}")

    image_shape = [x.cpu().detach().numpy() for x in img_shape]
    # (H, W) -> (W, H)
    image_shape = [[y, x] for x, y in zip(image_shape[0], image_shape[1])]

    assert len(image_shape) == len(file_names)

    idx = 0
    for i_shape, file_name in zip(image_shape, file_names):
        tmp = tensor[:, idx, ...]
        tmp2 = tensor2[:, idx, ...] if tensor2 is not None else None
        # tmp = np.transpose(np.squeeze(tmp), [0, 1, 2])
        tmp = np.squeeze(tmp)
        tmp2 = np.squeeze(tmp2) if tensor2 is not None else None

        # Iterate our all 7 NN outputs for a particular image
        preds = []
        fuse_num = tmp.shape[0]-1
        for i in range(tmp.shape[0]):
            tmp_img = tmp[i]
            tmp_img = np.uint8(image_normalization(tmp_img))
            tmp_img = cv2.bitwise_not(tmp_img)
            # tmp_img[tmp_img < 0.0] = 0.0
            # tmp_img = 255.0 * (1.0 - tmp_img)
            if tmp2 is not None:
                tmp_img2 = tmp2[i]
                tmp_img2 = np.uint8(image_normalization(tmp_img2))
                tmp_img2 = cv2.bitwise_not(tmp_img2)

            # Resize prediction to match input image size
            if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
                tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))
                tmp_img2 = cv2.resize(tmp_img2, (i_shape[0], i_shape[1])) if tmp2 is not None else None


            if tmp2 is not None:
                tmp_mask = np.logical_and(tmp_img>128,tmp_img2<128)
                tmp_img= np.where(tmp_mask, tmp_img2, tmp_img)
                preds.append(tmp_img)

            else:
                preds.append(tmp_img)

            if i == fuse_num:
                # print('fuse num',tmp.shape[0], fuse_num, i)
                fuse = tmp_img
                fuse = fuse.astype(np.uint8)
                if tmp_img2 is not None:
                    fuse2 = tmp_img2
                    fuse2 = fuse2.astype(np.uint8)
                    # fuse = fuse-fuse2
                    fuse_mask=np.logical_and(fuse>128,fuse2<128)
                    fuse = np.where(fuse_mask,fuse2, fuse)

                    # print(fuse.shape, fuse_mask.shape)

        # Get the mean prediction of all the 7 outputs
        average = np.array(preds, dtype=np.float32)
        average = np.uint8(np.mean(average, axis=0))
        output_file_name_f = os.path.join(output_dir_f, file_name)
        output_file_name_a = os.path.join(output_dir_a, file_name)

        fuse[fuse >= 200] = 255
        fuse[fuse < 200] = 0
        
        cv2.imwrite(output_file_name_f, fuse)
        cv2.imwrite(output_file_name_a, average)

        idx += 1
        return fuse, average

def test(checkpoint_path, dataloader, model, device, output_dir):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    model.eval()

    with torch.no_grad():
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)

            file_names = sample_batched['file_names']

            image_shape = sample_batched['image_shape']

            print(f"{file_names}: {images.shape}")

            end = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            preds = model(images)


            if device.type == 'cuda':
                torch.cuda.synchronize()

            fuse, average = save_image_batch_to_disk(preds,
                                     output_dir,
                                     file_names,
                                     image_shape)
            torch.cuda.empty_cache()
    return fuse, average





def main(img_path):
    """Main function."""



    checkpoint_path = 'checkpoints/BRIND/11/11_model.pth'


    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')

    # Instantiate model and move it to the computing device
    model = LDC().to(device)



    TEST_DATA = 'CLASSIC'

    test_inf = dataset_info(TEST_DATA, is_linux=IS_LINUX)

    # test_inf {'img_height': 512, 'img_width': 512, 'test_list': None, 'train_list': None, 'data_dir': 'data', 'yita': 0.5}
    
    test_dir = test_inf['data_dir']
#     input_val_dir = test_inf['data_dir']
    input_val_dir = img_path
    test_list = test_inf['test_list']
    test_img_width = test_inf['img_width']
    test_img_height = test_inf['img_height']
    mean_pixel_values = [103.939,116.779,123.68,137.86]    
    workers = 8
    
    dataset_val = TestDataset(input_val_dir,
                              test_data=TEST_DATA,
                              img_width=test_img_width,
                              img_height=test_img_height,
                              mean_bgr=mean_pixel_values[0:3] if len(
                                  mean_pixel_values) == 4 else mean_pixel_values,
                              test_list=test_list
                              )
    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=workers)


    output_dir = 'result/BRIND2CLASSIC'
    print(f"output_dir: {output_dir}")

    fuse, average = test(checkpoint_path, dataloader_val, model, device, output_dir)


    print('-------------------------------------------------------')
    
    return fuse, average




if __name__ == '__main__':
    img_path = 'data/b.jpg'
    fuse, average = main(img_path)
    print(fuse)