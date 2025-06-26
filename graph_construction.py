#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import torch
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import clip
from torch.utils.data import Dataset, DataLoader, BatchSampler
from tqdm import tqdm
import nltk, re, string

device = "cuda:0" if torch.cuda.is_available() else "cpu" 
dataset_name = 'weibo' # twitter or weibo or pheme
saveCSV = True
BATCH_SIZE = 32
Threshold = 0.95  


def preprocess_text(text):
    text = text.strip()
    text = re.sub(re.compile('<.*?>'), ' ', text)
    text = word_tokenize(text)
    text = ' '.join(word for word in text if word.isalpha() or word.isnumeric() or word.isalnum())
    return text

def preprocess_event(text):
    text = text.split('_')[0]
    return text
if dataset_name == 'weibo':
    df = pd.read_csv('dataset/weibo/weibo_train.csv')
    df_test = pd.read_csv('dataset/weibo/weibo_test.csv')
    df['event'] = df['image_id']
    df_test['event'] = df_test['image_id']
    for i in  range(0, len(df['event'])):
        df['event'][i] = 1 
    for i in  range(0, len(df_test['event'])):
        df_test['event'][i] = 1 
    IMG_ROOT_train = "dataset/weibo/images"
    IMG_ROOT_test = "dataset/weibo/images"
if dataset_name == 'twitter':
    df= pd.read_csv('dataset/twitter/train_posts_clean.csv')
    df_test = pd.read_csv('dataset/twitter/test_posts.csv')
    df['event'] = df['image_id']
    for i in  range(0, len(df['label'])):
        df['label'][i] = 1 if df['label'][i] == 'real' else 0
    df.event = np.array([preprocess_event(text) for text in df.event])
    df_test['event'] = df_test['image_id']
    for i in  range(0, len(df_test['label'])):
        df_test['label'][i] = 1 if df_test['label'][i] == 'real' else 0
    df_test.event = np.array([preprocess_event(text) for text in df_test.event])
    IMG_ROOT_train = "dataset/twitter/twitter_cleaned/images_train"
    IMG_ROOT_test = "dataset/twitter/twitter_cleaned/images_test"
if dataset_name == 'pheme':
    df = pd.read_csv('dataset/pheme/pheme_train.csv')
    df_test = pd.read_csv('dataset/pheme/pheme_test.csv')
    IMG_ROOT_train = "dataset/pheme/pheme_image/images"
    IMG_ROOT_test = "dataset/pheme/pheme_image/images" 
    
df.rename(columns={'post_text': 'text'}, inplace=True)
df.text = np.array([preprocess_text(text) for text in df.text])
df_test.rename(columns={'post_text': 'text'}, inplace=True)
df_test.text = np.array([preprocess_text(text) for text in df_test.text])
if saveCSV and dataset_name == 'weibo':
    df.to_csv('dataset/weibo/dataforGCN_train.csv',index=False)
    df_test.to_csv('dataset/weibo/dataforGCN_test.csv',index=False)
if saveCSV and dataset_name == 'twitter':
    df.to_csv('dataset/twitter/dataforGCN_train.csv',index=False)
    df_test.to_csv('dataset/twitter/dataforGCN_test.csv',index=False)
if saveCSV and dataset_name == 'pheme':
    df.to_csv('dataset/pheme/dataforGCN_train.csv',index=False)
    df_test.to_csv('dataset/pheme/dataforGCN_test.csv',index=False)
df


class image_caption_dataset(Dataset):
    def __init__(self, df, IMG_ROOT,  name="twitter"):
        self.dataset_name = name
        self.img_root = IMG_ROOT
        if name == "twitter":
            self.images = df["image_id"].tolist()
            self.caption = df["text"].tolist()
            self.label = df["label"].tolist()
            self.event = df["event"].tolist()
        elif name == "weibo":
            self.images = df["image_id"].tolist()
            self.caption = df["text"].tolist()
            self.label = df["label"].tolist()
            self.event = df["event"].tolist()
        elif name == "pheme":
            self.images = df["imgnum"].tolist()
            self.caption = df["text"].tolist()
            self.label = df["label"].tolist()
            self.event = df["event"].tolist()

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        if self.dataset_name == "twitter":
            images = preprocess(Image.open(self.img_root+'/'+self.images[idx]+'.jpg')) 
        elif self.dataset_name == "weibo":
            images = preprocess(Image.open(self.img_root+'/'+self.images[idx])) 
        elif self.dataset_name == "pheme":
            images = preprocess(Image.open(self.img_root+'/'+str(self.images[idx])+'.jpg')) 
        caption = self.caption[idx]
        label = self.label[idx]
        event = self.event[idx]
        return images, caption, label, event, idx
    
dataset = image_caption_dataset(df, IMG_ROOT_train, dataset_name)
dataset_test = image_caption_dataset(df_test, IMG_ROOT_test, dataset_name)

clip_model, preprocess = clip.load("ViT-B/32",device=device,jit=False)
data_dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=False)
pbar = tqdm(data_dataloader, leave=False)
ALLimage_embeds = []
ALLtext_embeds = []
ALLlabels = []
ALLids = []
ALLevents = []
for batch in pbar:
        images, texts, labels, events, idxs = batch
        images = images.to(device)
        texts = clip.tokenize(texts,truncate=True).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(images).float()
            text_features = clip_model.encode_text(texts).float()
        ALLimage_embeds.append(image_features)
        ALLtext_embeds.append(text_features)
        ALLlabels.extend(list(labels))
        ALLevents.extend(list(events))
        ALLids.append(idxs)
ALLimage_embeds_train = torch.cat(ALLimage_embeds, dim=0)
ALLtext_embeds_train = torch.cat(ALLtext_embeds, dim=0)
ALLids = torch.cat(ALLids, dim=0)

data_dataloader = DataLoader(dataset_test, BATCH_SIZE, shuffle=False)
pbar = tqdm(data_dataloader, leave=False)
ALLimage_embeds = []
ALLtext_embeds = []
ALLlabels = []
ALLids = []
ALLevents = []
for batch in pbar:
        images, texts, labels, events, idxs = batch
        images = images.to(device)
        texts = clip.tokenize(texts,truncate=True).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(images).float()
            text_features = clip_model.encode_text(texts).float()
        ALLimage_embeds.append(image_features)
        ALLtext_embeds.append(text_features)
        ALLlabels.extend(list(labels))
        ALLevents.extend(list(events))
        ALLids.append(idxs)
ALLimage_embeds_test = torch.cat(ALLimage_embeds, dim=0)
ALLtext_embeds_test = torch.cat(ALLtext_embeds, dim=0)
ALLids = torch.cat(ALLids, dim=0)

def calculate_cosine_similarity_matrix(h_emb, eps=1e-5):
    a_n = h_emb.norm(dim=1).unsqueeze(1)
    a_norm = h_emb / torch.max(a_n, eps * torch.ones_like(a_n))
    sim_matrix = torch.einsum('bc,cd->bd', a_norm, a_norm.transpose(0,1))
    return sim_matrix
ALLCAT_embeds_train = torch.cat((ALLimage_embeds_train, ALLtext_embeds_train), 1)
ALLCAT_embeds_test = torch.cat((ALLimage_embeds_test, ALLtext_embeds_test), 1)
ALLCAT_embeds = torch.cat((ALLCAT_embeds_train, ALLCAT_embeds_test), 0)

ALLTEXT_embeds = torch.cat((ALLtext_embeds_train, ALLtext_embeds_test), 0)

ALLIMAGE_embeds = torch.cat((ALLimage_embeds_train, ALLimage_embeds_test), 0)

ALLimage_embeds = torch.cat((ALLimage_embeds_train, ALLimage_embeds_test), 0)
ALLtext_embeds = torch.cat((ALLtext_embeds_train, ALLtext_embeds_test), 0)

ALLCAT_embeds /= ALLCAT_embeds.norm(dim=-1, keepdim=True)
ALLimage_embeds /= ALLimage_embeds.norm(dim=-1, keepdim=True)
ALLtext_embeds /= ALLtext_embeds.norm(dim=-1, keepdim=True)

ALLCAT_similarity = (ALLCAT_embeds @ ALLCAT_embeds.T)
i2t_similarity = (ALLimage_embeds @ ALLtext_embeds.T)
t2i_similarity = (ALLtext_embeds @ ALLimage_embeds.T)
i2i_similarity = (ALLimage_embeds @ ALLimage_embeds.T)
t2t_similarity = (ALLtext_embeds @ ALLtext_embeds.T)

torch.save(ALLCAT_similarity, 'dataset/'+dataset_name+'/ALLCAT_similarity.pt')
torch.save(i2t_similarity, 'dataset/'+dataset_name+'/i2t_similarity.pt')
torch.save(t2i_similarity, 'dataset/'+dataset_name+'/t2i_similarity.pt')
torch.save(i2i_similarity, 'dataset/'+dataset_name+'/i2i_similarity.pt')
torch.save(t2t_similarity, 'dataset/'+dataset_name+'/t2t_similarity.pt')

result_tensor = torch.zeros_like(ALLCAT_similarity)
result_tensor[(i2t_similarity > Threshold) | (t2t_similarity > Threshold)] = 2
result_tensor[(t2i_similarity > Threshold) | (i2i_similarity > Threshold)] = 3
result_tensor[ALLCAT_similarity > Threshold] = 1
edge = result_tensor
edge_sparse = edge.to_sparse()

torch.save(edge_sparse, 'dataset/'+dataset_name+'/TweetGraph.pt')
torch.save(ALLCAT_embeds, 'dataset/'+dataset_name+'/TweetEmbeds.pt')
torch.save(ALLTEXT_embeds, 'dataset/'+dataset_name+'/TweetTextEmbeds.pt')
torch.save(ALLIMAGE_embeds, 'dataset/'+dataset_name+'/TweetImageEmbeds.pt')