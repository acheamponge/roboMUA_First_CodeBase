# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 04:43:21 2019

@author: acham
"""
import pandas as pd
import os
from flask import Flask, request, redirect, flash, render_template, send_from_directory, url_for
import torch
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
import torch

import time
import json
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import PIL
from PIL import Image
from collections import OrderedDict
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from werkzeug.utils import secure_filename




app = Flask(__name__)
UPLOAD_FOLDER = "./"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(180),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
# TODO: Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(os.path.join('./content', x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}
# TODO: Using the image datasets and the trainforms, define the dataloaders
batch_size = 102
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'valid']}
class_names = image_datasets['train'].classes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

with open('./content/skin.json', 'r') as f:
    cat_to_name = json.load(f)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    #checkpoint['input_size'] = 25088
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict=(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']

    for param in model.parameters():
        param.requires_grad = False

    return model, checkpoint['class_to_idx']
model, class_to_idx = load_checkpoint('checkpoint_ic_d161.pth')


idx_to_class = { v : k for k,v in class_to_idx.items()}

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    # tensor.numpy().transpose(1, 2, 0)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    return image

model.class_to_idx = image_datasets['train'].class_to_idx



#import plotly.graph_objects as go

rec = pd.read_csv('./content/makeup.csv')

def view_classify(img_path, prob, classes, mapping, rec):
    ''' Function for viewing an image and it's predicted classes.
    '''
    image = Image.open(img_path)
    probs, classes = predict2(img_path, model.to(device))
    nd= {'darkskin': 1, 'olive skin': 2, 'porcelain skin':3}
    fig, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)
    emotion_name = mapping[str(classes[0])]
   # ax1.set_title(emotion_name)
   # ax1.imshow(image)
   # ax1.axis('off')
   # print(type(classes[0]))
   # y_pos = np.arange(len(prob))
   # ax2.barh(y_pos, prob, align='center')
   # ax2.set_yticks(y_pos)
   # ax2.set_yticklabels(emotion_names)
   # ax2.invert_yaxis()  # labels read top-to-bottom
   # ax2.set_title('Class Probability')

    skintype = cat_to_name[str(classes[0])]
    rec1 = rec.loc[rec['SkinTone'] ==skintype]
    #fig = go.Figure(data=[go.Table(
    #header=dict(values=list(rec1.columns),
    #            fill_color='paleturquoise',
    #            align='left'),
    #cells=dict(values=[rec1.Foundation, rec1.HEXColor, rec1.SkinTone,rec1.Company, rec1.ProductURL, rec1.Price, rec1.Image, rec1.VideoTutorial, rec1.YoutubeCode, rec1.InstagramFilter],
    #           fill_color='lavender',
    #           align='left'))
     #   ])
   # fig.show()
    #print(rec1.to_dict())
    def make_clickable(val):
        # target _blank to open new window
        return '<a target="_blank" href="{}">{}</a>'.format(val, val)

    rec1.style.format({'ProductURL': make_clickable})

    return(rec1)

def predict2(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Implement the code to predict the class from an image file
    img = Image.open(image_path)
    img = process_image(img)
    #print("HIIIII")
    # Convert 2D image to 1D vectorcle
    img = np.expand_dims(img, 0)


    img = torch.from_numpy(img)

    model.eval()
    inputs = Variable(img).to(device)
    #print(inputs)
    #model.classifier[0] = nn.Linear(2208, 120, bias = True)

    logits = model.forward(inputs)

    ps = F.softmax(logits,dim=1)
    topk = ps.cpu().topk(topk)

    return (e.data.numpy().squeeze().tolist() for e in topk)

import pandas as pd

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/results", methods=["POST"])
def result():
    text = ''
    filename=''
    if request.method == "POST":
        # Check if the post request has the file part.
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # If user does not select file, browser also
        # submit an empty part without filename.
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file:
            if allowed_file(file.filename):
                probs, classes = predict2(file, model.to(device))
                text = (view_classify(file, probs, class_names, cat_to_name, rec))
                print(type(text))
            else:
                text = "Only Upload *.jpg files"


        def path_to_image_html(path):
            return '<img src="'+ path + '" style=height:200px;width:200px;"/>'


    return render_template("results.html", text=text.to_html(escape=False, index=False))

if __name__ == "__main__":
    app.run()
