import torch
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
from util import BackgroundGenerator
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from pycocotools.coco import COCO


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class CustomDataSet(Dataset):
    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        return img, text, label

    def __len__(self):
        count = len(self.images)
        return count


class SingleModalDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        count = len(self.data)
        return count


def get_loader(path, batch_size, INCOMPLETE=False, USE_INCOMPLETE=False):
    # data_mat = loadmat("labels.COCO.mat")
    # lab_list = data_mat['labels']
    # id_list = data_mat['id']
    # # img_list = data_mat['img_name']

    # caps_file_train = os.path.join("annotations", "captions_train2014.json")
    # caps_file_val = os.path.join("annotations", "captions_val2014.json")

    # coco_caps_train = COCO(caps_file_train)
    # coco_caps_val = COCO(caps_file_val)

    # model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    # path = 'datapath_to_coco/'

    # img_list = []
    # txt_list = []
    # cnt = 0
    # for idx in id_list:
    #     idx = int(idx)
    #     _annIds = coco_caps_train.getAnnIds(imgIds=idx)
    #     if len(_annIds):
    #         _anns = coco_caps_train.loadAnns(_annIds)
    #     else:
    #         _annIds = coco_caps_val.getAnnIds(imgIds=idx)
    #         _anns = coco_caps_val.loadAnns(_annIds)
        
    #     sentences = [_a["caption"] for _a in _anns]

    #     if os.path.exists(path + 'train2014/COCO_train2014_' + str(idx).zfill(12) + '.jpg'):
    #         img = Image.open(path + 'train2014/COCO_train2014_' + str(idx).zfill(12) + '.jpg')
    #     else:
    #         img = Image.open(path + 'val2014/COCO_val2014_' + str(idx).zfill(12) + '.jpg')

    #     inputs = processor(text=sentences, images=img, return_tensors="pt", padding=True).to(torch.device("cuda:0"))

    #     outputs = model(**inputs)
    #     img_list.append(outputs['image_embeds'].cpu().detach().numpy())
    #     txt_list.append(outputs['text_embeds'].mean(dim=0)[None,:].cpu().detach().numpy())
    #     cnt += 1
    #     print(cnt)

    # img_list = np.concatenate(img_list)
    # txt_list = np.concatenate(txt_list)

    # shuffle_idx = np.arange(img_list.shape[0])
    # np.random.shuffle(shuffle_idx)
    # img_list = img_list[shuffle_idx]
    # txt_list = txt_list[shuffle_idx]
    # lab_list = lab_list[shuffle_idx]

    # img_train = img_list[2000:]
    # img_test = img_list[:2000]
    # text_train = txt_list[2000:]
    # text_test = txt_list[:2000]
    # label_train = lab_list[2000:]
    # label_test = lab_list[:2000]

    # savemat("data/train.mat", {'img_train':img_train, 'txt_train':text_train, 'lab_train':label_train}, do_compression=True)
    # savemat("data/test.mat", {'img_test':img_test, 'txt_train':text_test, 'lab_train':label_test}, do_compression=True)
    
    if 0:
        mat_train = loadmat("model/train.mat")
        mat_test = loadmat("model/test.mat")
        
        img_train = mat_train['img_train']
        text_train = mat_train['txt_train']
        label_train = mat_train['lab_train']

        img_test = mat_test['img_test']
        text_test = mat_test['txt_test']
        label_test = mat_test['lab_test']
    else:
        mat_train = loadmat(path + "train_img.mat")
        img_train = mat_train['train_img']
        mat_test = loadmat(path + "test_img.mat")
        img_test = mat_test['test_img']
        text_train = loadmat(path + "train_txt.mat")['train_txt']
        text_test = loadmat(path + "test_txt.mat")['test_txt']
        label_train = loadmat(path + "train_lab.mat")['train_lab']
        label_test = loadmat(path + "test_lab.mat")['test_lab']

    # tr_lab = loadmat(path + "train_lab.mat")['train_lab']
    # tr_id = loadmat(path + "train_lab.mat")['id'][0]
    # tt_lab = loadmat(path + "test_lab.mat")['test_lab']
    # tt_id = loadmat(path + "test_lab.mat")['id'][0]
    # c_lab = loadmat("labels.COCO.mat")['labels']
    # c_name = loadmat("labels.COCO.mat")['img_name']

    # cnt = 0
    # for id in tr_id:
    #     idx = np.where(tr_id == cnt)[0][0]
    #     if (c_lab[cnt] == tr_lab[idx]).all() == False:
    #         print('!!!')
    #         break
    #     cnt+=1
    
    # cnt = 0
    # for id in tt_id:
    #     if (c_lab[id] == tt_lab[cnt]).all() == False:
    #         print('!!!')
    #         break
    #     cnt += 1


    # id_train = mat_train['id'][0]
    # id_test = mat_test['id'][0]


    # Incomplete modal
    split = img_train.shape[0] // 5
    if INCOMPLETE:
        text_train[split * 1: split * 3] = np.zeros_like(text_train[split * 1: split * 3])
        img_train[split * 3: split * 5] = np.zeros_like(img_train[split * 3: split * 5])

    imgs = {'train': img_train, 'test': img_test}
    texts = {'train': text_train, 'test': text_test}
    labels = {'train': label_train, 'test': label_test}

    if USE_INCOMPLETE:
        shuffle = {'train_complete': True, 'train_img': True, 'train_text': True, 'test': False}
        dataset = {'train_complete': CustomDataSet(images=imgs['train'][:split * 1],
                                                   texts=texts['train'][:split * 1],
                                                   labels=labels['train'][:split * 1]),
                   'train_img': SingleModalDataSet(data=imgs['train'][split * 1:split * 3],
                                                   labels=labels['train'][split * 1:split * 3]),
                   'train_text': SingleModalDataSet(data=texts['train'][split * 3: split * 5],
                                                    labels=labels['train'][split * 3: split * 5]),
                   'test': CustomDataSet(images=imgs['test'], texts=texts['test'], labels=labels['test'])}
        dataloader = {'train_complete': DataLoaderX(dataset['train_complete'], batch_size=batch_size // 5,
                                                    shuffle=shuffle['train_complete'], num_workers=0),
                      'train_img': DataLoaderX(dataset['train_img'], batch_size=batch_size // 5 * 2,
                                               shuffle=shuffle['train_img'], num_workers=0),
                      'train_text': DataLoaderX(dataset['train_text'], batch_size=batch_size // 5 * 2,
                                                shuffle=shuffle['train_text'], num_workers=0),
                      'test': DataLoaderX(dataset['test'], batch_size=batch_size,
                                          shuffle=shuffle['test'], num_workers=0),
                      }
    else:
        shuffle = {'train': True, 'test': False}
        dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
                   for x in ['train', 'test']}
        dataloader = {x: DataLoaderX(dataset[x], batch_size=batch_size,
                                     shuffle=shuffle[x], num_workers=0) for x in ['train', 'test']}

    img_dim = img_train.shape[1]
    text_dim = text_train.shape[1]
    num_class = label_train.shape[1]

    input_data_par = {}
    input_data_par['img_test'] = img_test
    input_data_par['text_test'] = text_test
    input_data_par['label_test'] = label_test
    input_data_par['img_train'] = img_train
    input_data_par['text_train'] = text_train
    input_data_par['label_train'] = label_train
    input_data_par['img_dim'] = img_dim
    input_data_par['text_dim'] = text_dim
    input_data_par['num_class'] = num_class
    return dataloader, input_data_par
