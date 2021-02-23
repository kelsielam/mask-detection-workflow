#!/usr/bin/env python3

"""
MACHINE LEARNING WORKFLOWS 

Model Training
"""
import glob,os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from bs4 import BeautifulSoup
from PIL import Image
import torch
import torchvision

from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.patches as patches


DATASET_DIR = ""
# Read in training data
imgs   = list(sorted(os.listdir("data/images/")))
labels = list(sorted(os.listdir("data/annotations/")))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DATASET_DIR = ""

# ----------- HELPER FUNCTIONS ---------

def generate_box(obj):
    
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)
    
    return [xmin, ymin, xmax, ymax]

def generate_label(obj):
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    return 0

def generate_target(image_id, file): 
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, "html.parser")
        objects = soup.find_all('object')

        num_objs = len(objects)
        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([image_id])
        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id

        
        return target

def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
# ----------------------------------------------------------------


#-------------- DATASET CLASS ------------------------------------
class MaskDataset(object):
    def __init__(self, transforms):
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir("data/images/")))
        self.labels = list(sorted(os.listdir("data/annotations/")))

    def __getitem__(self, idx):
        # load images ad masks
        file_image = 'maksssksksss'+ str(idx) + '.png'
        file_label = 'maksssksksss'+ str(idx) + '.xml'
        img_path = os.path.join("data/images/", file_image)
        label_path = os.path.join("data/annotations/", file_label)
        img = Image.open(img_path).convert("RGB")
        #Generate Label
        target = generate_target(idx, label_path)
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)




def validate(val_loader, model, criterion,device):

    model.eval()
    model.to(device)
    accuracy = 0
    test_loss = 0

    with torch.no_grad():
        for sample_batched in val_loader:
            pass

    return model, test_loss_final, accuracy_final


def train(train_loader, model, criterion, optimizer, epoch, device):

    model.train()
    model.to(device)
    running_loss = 0

    for imgs, annotations in train_loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

        
        optimizer.zero_grad() 
        inputs, labels = sample_batched['image'].float(), sample_batched["label"]

        inputs, labels = inputs.to(device), labels.to(device)     
        log_ps = model(inputs)
        loss   = criterion(log_ps,labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss/len(train_loader)
    print("Train Loss: {}".format(train_loss))

    return model, train_loss




### --------------------TRAIN MODEL--------------------------------
def main():

    data_transform = transforms.Compose([transforms.ToTensor(),])
    dataset = MaskDataset(data_transform)
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4,shuffle=True, collate_fn=collate_fn)
    model = get_model_instance_segmentation(3)
    model.to(device)

    losses_dict= {'train': {}, 'test': {}, 'accuracy': {}}
    criterion = nn.NLLLoss()


    for e in range(epochs):
        print("{} out of {}".format(e+1, epochs))
        time.sleep(1)
        model, train_loss = train(train_dataloader, model, criterion, optimizer, epochs,device)
        model, test_loss, test_accuracy = validate(val_dataloader, model, criterion,device)
        current_metrics = [e,train_loss, test_loss,test_accuracy]
        losses_dict["train"][e] = train_loss
        losses_dict["test"][e] = test_loss
        losses_dict["accuracy"][e] = test_accuracy
        if e % 2 == 0:
            checkpoints_files = save_checkpoint()
    
    return checkpoints_files




if __name__ == "__main__":
    main()
