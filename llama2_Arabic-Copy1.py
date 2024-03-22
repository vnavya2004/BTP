#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # To disable GPU

import torch

# Check if CUDA (GPU) is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)


# In[2]:


import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,AutoModelForCausalLM, AutoModelForSequenceClassification, LlamaTokenizer ,LlamaForCausalLM, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import torch.nn.functional as F


# In[3]:


#!pip install openpyxl
import pandas as pd
df = pd.read_excel('Arabic_Depression_10.000_Tweets.xlsx')


# In[4]:


df.head()


# In[5]:


cd /home/abcd/Navya


# In[6]:


model_name = "meta-llama/Llama-2-7b-hf"
api_token = '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'
vocab_size = 50000
tokenizer = AutoTokenizer.from_pretrained(model_name, vocab_size=vocab_size, token=api_token,padding=True)
model = AutoModelForCausalLM.from_pretrained(model_name, vocab_size=vocab_size, token=api_token, ignore_mismatched_sizes=True)


# In[11]:


tokenizer.add_special_tokens({'pad_token': '[PAD]'})
import torch
import torch.nn as nn
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

# Assuming 'df' is your DataFrame containing tweet text and labels
for index, row in df.iterrows():
    # Tokenize input tweet
    inputs = tokenizer(row["tweet"], return_tensors="pt", truncation=True, padding=True)
    
    # Get label
    label = torch.tensor(row["label"])  # Assuming single label per tweet
    print(label)
    # Forward pass
    outputs = model(**inputs)
    
    # Compute loss
    loss = loss_fn(outputs.logits, label)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Iteration {index + 1}: Loss = {loss.item()}")


# In[9]:


outputs.logits.shape


# In[10]:


outputs


# In[12]:


model.config


# In[13]:


tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# Tokenize the text data in the dataset

tokenized_texts = tokenizer(df['tweet'].tolist(), padding=True, truncation=True, max_length=32, return_tensors='pt')

# Convert tokenized texts to PyTorch tensors
input_ids = tokenized_texts['input_ids']
attention_masks = tokenized_texts['attention_mask']
labels = torch.tensor(df['label'].tolist())

# Specify the columns for features (tweets) and labels
tweets_column = 'tweet'
labels_column = 'label'
NUM_LABELS = len(df[labels_column].unique())
df.head()
possible_labels = df[labels_column].unique()
label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
X_train, X_test, y_train, y_test = train_test_split(df[tweets_column], df[labels_column], stratify=df[labels_column])


# Tokenize the training data
encoded_data_train = tokenizer.batch_encode_plus(
    X_train.tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length=32,
    truncation=True,
    return_tensors='pt'
)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(y_train.values)

# Tokenize the validation data
encoded_data_val = tokenizer.batch_encode_plus(
    X_test.tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length=32,
    truncation=True,
    return_tensors='pt'
)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(y_test.values)

dataset_train = TensorDataset(input_ids_train,
                              attention_masks_train,
                              labels_train)

dataset_val = TensorDataset(input_ids_val,
                            attention_masks_val,
                            labels_val)


# In[14]:


# Find the number of samples in X_train and y_train
num_samples_train = len(X_train)
num_labels_train = len(y_train)

print("Number of samples in X_train:", num_samples_train)
print("Number of samples in y_train:", num_labels_train)
# Find the number of samples in the encoded data tensors
num_samples_encoded_train = input_ids_train.shape[0]
num_samples_labels_train = labels_train.shape[0]

print("Number of samples in input_ids_train:", num_samples_encoded_train)
print("Number of samples in labels_train:", num_samples_labels_train)
print(input_ids_train.shape)
print(labels_train.shape)


# In[15]:


optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8,no_deprecation_warning=True)
epochs = 5
batch_size = 32
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataset_train)*epochs)


# In[16]:


def binary_accuracy(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, preds_flat)

def binary_f1_score(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat)

def binary_precision(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return precision_score(labels_flat, preds_flat)

def binary_recall(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return recall_score(labels_flat, preds_flat)


# In[17]:


import os
import torch

# Function to enable or disable GPU
def set_cuda_visible_devices(enable_gpu):
    if enable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set to the GPU index you want to use, e.g., "0" for the first GPU
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # To disable GPU
set_cuda_visible_devices(True)
# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# In[ ]:


import torch.nn.functional as F
batch_size=32
# Calculate total batches
total_batches_train = len(dataset_train) // batch_size
total_batches_val = len(dataset_val) // batch_size

# Training loop
for epoch in range(1, epochs + 1):
    model.train()
    loss_train_total = 0
    progress_bar = tqdm(DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size), desc='Epoch {:1d}'.format(epoch), total=total_batches_train, leave=False, disable=False)
    
    for batch in progress_bar:
        optimizer.zero_grad()
        batch = tuple(b.to(device) for b in batch)
    
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
        outputs = model(**inputs) 
        logits = outputs.logits
        targets = batch[2]          
        targets_expanded = targets.unsqueeze(1).unsqueeze(2).expand(-1,32, 50000)
        print("Logits shape:", logits.shape)
        print("Flattened targets shape:", targets_expanded.float().shape)
        # Calculate binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(logits, targets_expanded.float())
        loss_train_total += loss.item()
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(logits_flat))})


    loss_train_avg = loss_train_total.item() / total_batches_train

    tqdm.write('\nEpoch {epoch}')
    tqdm.write(f'Training loss: {loss_train_avg}')

    # Evaluation on validation data
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []

    progress_bar_val = tqdm(DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size), desc='Evaluating', total=total_batches_val, leave=False, disable=False)
    for batch in progress_bar_val:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
        with torch.no_grad():
            outputs = model(**inputs)
        print(ouputs)
        # Calculate validation loss
        loss = F.binary_cross_entropy_with_logits(outputs.logits, batch[2].float())  # Assuming batch[2] contains labels
        loss_val_total += loss.item()
        
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = batch[2].cpu().numpy()  # Assuming batch[2] contains labels
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / total_batches_val
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    val_accuracy = binary_accuracy(predictions, true_vals)
    val_f1 = binary_f1_score(predictions, true_vals)
    val_precision = binary_precision(predictions, true_vals)
    val_recall = binary_recall(predictions, true_vals)
    tqdm.write(f'Validation loss: {loss_val_avg}')
    tqdm.write(f'Accuracy: {val_accuracy}')
    tqdm.write(f'F1 Score: {val_f1}')
    tqdm.write(f'Precision: {val_precision}')
    tqdm.write(f'Recall: {val_recall}')


# In[57]:


print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)
print("Flattened logits shape:", logits_flat.float().shape)

print("Flattened targets shape:", targets_flat.shape)

print("Flattened targets shape:", target_flat.float().shape)
targets_expanded = targets.unsqueeze(1).unsqueeze(2).expand(-1,32, 50000)
print("Flattened targets shape:", targets_expanded.float().shape)


# In[ ]:





# In[ ]:




