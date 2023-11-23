import os
import json
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image
import re
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transform.randaugment import RandomAugment
from torchvision.transforms.functional import InterpolationMode


def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption


class coco_karpathy_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_karpathy_train.json'

        download_url(url,ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root,ann['image'])      
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        # can add prompt in the form of 'str'
        caption = self.prompt + pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']] 
    

    
class coco_karpathy_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = [] # 5 captions for an image
        self.image = [] 
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id) # an image can have 5 captions
                self.txt2img[txt_id] = img_id # one caption only belongs to one image
                txt_id += 1

                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  
        # only return image
        return image, index
    

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(384,scale=(0.5, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])        
transform_test = transforms.Compose([
    transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    normalize,
    ])

data = coco_karpathy_train(transform_train, 'datapath_to_coco/', 'datapath_to_coco/')
print(data[0][0].shape)


from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
