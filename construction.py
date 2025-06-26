#!/usr/bin/env python
# coding: utf-8

import torch
from torch_geometric.data import Data
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
from nltk.tokenize import word_tokenize
import numpy as np
import clip
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re
import argparse
import openai
import os
import time


def get_args():
    parser = argparse.ArgumentParser(description='GLPN-LLM')
    parser.add_argument('--dataset_name', type=str, default='weibo', help='dataset name (weibo, twitter, pheme)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--threshold', type=float, default=0.95, help='similarity threshold')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--label_rate', type=float, default=0.65, help='label rate for training')
    parser.add_argument('--psesudo_mask_rate', type=float, default=0.05, help='pseudo mask rate for test set')
    parser.add_argument('--llm_label_rate', type=float, default=0.35, help='rate for using LLM labels')
    parser.add_argument('--hidden_channels', type=int, default=64, help='hidden channels for UniMP')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers for UniMP')
    parser.add_argument('--heads', type=int, default=2, help='number of heads for UniMP')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--save_csv', action='store_true', help='save processed data to CSV')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    return parser.parse_args()

args = get_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(args.seed)

device = torch.device(args.device)

def preprocess_text(text):
    text = text.strip()
    text = re.sub(re.compile('<.*?>'), ' ', text)
    text = word_tokenize(text)
    text = ' '.join(word for word in text if word.isalpha() or word.isnumeric() or word.isalnum())
    return text

def preprocess_event(text):
    text = text.split('_')[0]
    return text
if args.dataset_name == 'weibo':
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
if args.dataset_name == 'twitter':
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
if args.dataset_name == 'pheme':
    df = pd.read_csv('dataset/pheme/pheme_train.csv')
    df_test = pd.read_csv('dataset/pheme/pheme_test.csv')
    IMG_ROOT_train = "dataset/pheme/pheme_image/images"
    IMG_ROOT_test = "dataset/pheme/pheme_image/images" 
    
df.rename(columns={'post_text': 'text'}, inplace=True)
df.text = np.array([preprocess_text(text) for text in df.text])
df_test.rename(columns={'post_text': 'text'}, inplace=True)
df_test.text = np.array([preprocess_text(text) for text in df_test.text])
if args.save_csv and args.dataset_name == 'weibo':
    df.to_csv('dataset/weibo/dataforGCN_train.csv',index=False)
    df_test.to_csv('dataset/weibo/dataforGCN_test.csv',index=False)
if args.save_csv and args.dataset_name == 'twitter':
    df.to_csv('dataset/twitter/dataforGCN_train.csv',index=False)
    df_test.to_csv('dataset/twitter/dataforGCN_test.csv',index=False)
if args.save_csv and args.dataset_name == 'pheme':
    df.to_csv('dataset/pheme/dataforGCN_train.csv',index=False)
    df_test.to_csv('dataset/pheme/dataforGCN_test.csv',index=False)


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
    
dataset = image_caption_dataset(df, IMG_ROOT_train, args.dataset_name)
dataset_test = image_caption_dataset(df_test, IMG_ROOT_test, args.dataset_name)

clip_model, preprocess = clip.load("ViT-B/32",device=device,jit=False)
data_dataloader = DataLoader(dataset, args.batch_size, shuffle=False)
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

data_dataloader = DataLoader(dataset_test, args.batch_size, shuffle=False)
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

result_tensor = torch.zeros_like(ALLCAT_similarity)
result_tensor[(i2t_similarity > args.threshold) | (t2t_similarity > args.threshold)] = 2
result_tensor[(t2i_similarity > args.threshold) | (i2i_similarity > args.threshold)] = 3
result_tensor[ALLCAT_similarity > args.threshold] = 1
edge = result_tensor
edge_sparse = edge.to_sparse()

torch.save(edge_sparse, 'dataset/'+args.dataset_name+'/TweetGraph.pt')
torch.save(ALLCAT_embeds, 'dataset/'+args.dataset_name+'/TweetEmbeds.pt')
torch.save(ALLTEXT_embeds, 'dataset/'+args.dataset_name+'/TweetTextEmbeds.pt')
torch.save(ALLIMAGE_embeds, 'dataset/'+args.dataset_name+'/TweetImageEmbeds.pt')


train_data = pd.read_csv('dataset/' + args.dataset_name + '/dataforGCN_train.csv')
test_data = pd.read_csv('dataset/' + args.dataset_name + '/dataforGCN_test.csv')

tweet_embeds = torch.load('dataset/' + args.dataset_name + '/TweetEmbeds.pt', map_location=device)
tweet_graph = torch.load('dataset/' + args.dataset_name + '/TweetGraph.pt', map_location=device)

# you can use the psesudo labels generated by GPT to train the model, {args.dataset_name}_analysis_results.csv is the file that stores the psesudo labels
analysis_file_path = f'dataset/{args.dataset_name}/{args.dataset_name}_analysis_results.csv'

psesudo_labels = torch.tensor(psesudo_data["analysis"].tolist(), dtype=torch.long).to(device)

label_list_train = train_data["label"].tolist()
label_list_test = test_data["label"].tolist()

labels = []
for label_list in [label_list_train, label_list_test]:
    labels_i = torch.tensor(label_list, dtype=torch.long)
    labels.append(labels_i)

labels = torch.cat(labels, 0)

data = Data(
    x=tweet_embeds.float(),
    edge_index=tweet_graph.coalesce().indices(),
    edge_attr=None,
    train_mask=torch.tensor([True]*len(label_list_train) + [False]*(len(labels)-len(label_list_train))).bool(),
    test_mask=torch.tensor([False]*len(label_list_train) + [True]*(len(labels)-len(label_list_train))).bool(),
    y=labels
).to(device)
num_features = tweet_embeds.shape[1]
num_classes = 2

data.x = torch.cat([data.x, torch.zeros((data.num_nodes, num_classes), device=device)], dim=1)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels, num_layers,
                    heads, dropout=0.3):
        super().__init__()

        self.num_classes = num_classes

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            if i < num_layers:
                out_channels = hidden_channels // heads
                concat = True
            else:
                out_channels = num_classes
                concat = False
            conv = GCNConv(in_channels, out_channels, heads,
                                    concat=concat, beta=True, dropout=dropout)
            self.convs.append(conv)
            in_channels = hidden_channels

            if i < num_layers:
                self.norms.append(torch.nn.LayerNorm(hidden_channels))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index)).relu()
        x = self.convs[-1](x, edge_index)
        return x

data.y = data.y.view(-1)
model = GCN(num_features + num_classes, num_classes, hidden_channels=args.hidden_channels,
                num_layers=args.num_layers, heads=args.heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

train_mask = data.train_mask
test_mask = data.test_mask
test_mask_idx = data.test_mask.nonzero(as_tuple=False).view(-1)

psesudo_mask_rate = args.psesudo_mask_rate
psesudo_mask = torch.rand(test_mask_idx.shape[0], device=test_mask_idx.device) < psesudo_mask_rate
test_psesudo_idx = test_mask_idx[psesudo_mask]
selected_psesudo_labels = psesudo_labels[psesudo_mask]

def train(label_rate=args.label_rate):
    model.train()

    data.x[:, -num_classes:] = 0

    train_mask_idx = train_mask.nonzero(as_tuple=False).view(-1)
    mask = torch.rand(train_mask_idx.shape[0]) < label_rate
    train_labels_idx = train_mask_idx[mask]
    train_unlabeled_idx = train_mask_idx[~mask]

    data.x[train_labels_idx, -num_classes:] = F.one_hot(data.y[train_labels_idx], num_classes).float()

    llm_label_rate = args.llm_label_rate  
    llm_mask = torch.rand(test_psesudo_idx.shape[0], device=test_psesudo_idx.device) < llm_label_rate

    llm_indices = test_psesudo_idx[llm_mask]
    llm_labels = selected_psesudo_labels[llm_mask]

    data.x[llm_indices, -num_classes:] = F.one_hot(llm_labels, num_classes).float()

    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_unlabeled_idx], data.y[train_unlabeled_idx])
    loss.backward()
    optimizer.step()

    use_labels = True
    n_label_iters = 1

    if use_labels and n_label_iters > 0:
        unlabel_idx = torch.cat([train_unlabeled_idx, data.test_mask.nonzero(as_tuple=False).view(-1)])
        with torch.no_grad():
            for _ in range(n_label_iters):
                torch.cuda.empty_cache()
                out = out.detach()
                data.x[unlabel_idx, -num_classes:] = F.softmax(out[unlabel_idx], dim=-1)
                out = model(data.x, data.edge_index)

    return loss.item()

max_test_acc = 0

@torch.no_grad()
def test():
    model.eval()

    data.x[:, -num_classes:] = 0

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)

    data.x[train_idx, -num_classes:] = F.one_hot(data.y[train_idx], num_classes).float()

    data.x[test_psesudo_idx, -num_classes:] = F.one_hot(selected_psesudo_labels, num_classes).float()

    unlabel_idx = test_idx
    n_label_iters = 1
    for _ in range(n_label_iters):
        out = model(data.x, data.edge_index)
        data.x[unlabel_idx, -num_classes:] = F.softmax(out[unlabel_idx], dim=-1)

    out = model(data.x, data.edge_index)
    pred = out[test_mask].argmax(dim=-1)

    y_true = data.y[test_mask].cpu().numpy()
    y_pred = pred.cpu().numpy()

    test_acc = int((pred == data.y[test_mask]).sum()) / pred.size(0)

    precision = precision_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0)

    val_acc = 0

    return val_acc, test_acc, precision, recall, f1

best_precision = 0
best_recall = 0
best_f1 = 0
max_test_acc = 0

for epoch in range(args.epochs):
    loss = train()
    val_acc, test_acc, precision, recall, f1 = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, '
            f'Test Acc: {test_acc:.4f}, Precision: {precision:.4f}, '
            f'Recall: {recall:.4f}, F1: {f1:.4f}')

    if test_acc > max_test_acc:
        max_test_acc = test_acc
        best_precision = precision
        best_recall = recall
        best_f1 = f1
        best_epoch = epoch

print(f'Best Epoch: {best_epoch}, Max Test Acc: {max_test_acc:.4f}, '
        f'Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}')
