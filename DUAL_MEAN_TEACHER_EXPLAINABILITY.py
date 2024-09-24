#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install lime


# In[2]:


import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer  # LIME Import
import random

# Load the XLM-RoBERTa tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")


# Read the Excel file
df = pd.read_excel("./Bangla2.xlsx", header=0)
df = df.sample(frac=0.4, random_state=42)
tweets_column = 'tweets'
labels_column = 'labels'
NUM_LABELS = len(df[labels_column].unique())
possible_labels = df[labels_column].unique()
label_dict = {possible_label: index for index, possible_label in enumerate(possible_labels)}
df['labels'] = df[labels_column].map(label_dict)

# Split the dataset into labeled (20%), unlabeled (60%), and test (20%) sets
df_labeled, df_temp = train_test_split(df, stratify=df[labels_column], test_size=0.8)
df_unlabeled, df_test = train_test_split(df_temp, stratify=df_temp[labels_column], test_size=0.25)

# Tokenize the labeled data for training
encoded_data_train = tokenizer.batch_encode_plus(
    df_labeled[tweets_column].tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df_labeled['labels'].values)

# Tokenize the unlabeled data
encoded_data_unlabeled = tokenizer.batch_encode_plus(
    df_unlabeled[tweets_column].tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)

input_ids_unlabeled = encoded_data_unlabeled['input_ids']
attention_masks_unlabeled = encoded_data_unlabeled['attention_mask']

# Tokenize the test data
encoded_data_test = tokenizer.batch_encode_plus(
    df_test[tweets_column].tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)

input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(df_test['labels'].values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_unlabeled = TensorDataset(input_ids_unlabeled, attention_masks_unlabeled)
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

# Define two XLM-RoBERTa models for sequence classification (students and teachers)
student_model1 = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=NUM_LABELS)
teacher_model1 = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=NUM_LABELS)

student_model2 = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=NUM_LABELS)
teacher_model2 = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=NUM_LABELS)


# In[4]:


# Set up the device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
student_model1.to(device)
teacher_model1.to(device)
student_model2.to(device)
teacher_model2.to(device)

# Set up the optimizer and scheduler for both student models
optimizer1 = AdamW(student_model1.parameters(), lr=1e-5, eps=1e-8)
optimizer2 = AdamW(student_model2.parameters(), lr=1e-5, eps=1e-8)

epochs = 7
scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=0, num_training_steps=len(dataset_train) * epochs)
scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=0, num_training_steps=len(dataset_train) * epochs)

# Training loop with Dual Mean Teacher logic
def calculate_alpha(epoch, total_epochs, base_alpha=0.95, final_alpha=0.999):
    alpha = base_alpha + (final_alpha - base_alpha) * (epoch / total_epochs)
    return alpha

for epoch in range(1, epochs + 1):
    student_model1.train()
    student_model2.train()
    loss_train_total = 0
    progress_bar = tqdm(DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=4), desc=f'Epoch {epoch}', leave=False, disable=False)
    alpha = calculate_alpha(epoch, epochs)

    for batch in progress_bar:
        student_model1.zero_grad()
        student_model2.zero_grad()
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}

        # Supervised loss for both students
        outputs_student1 = student_model1(**inputs)
        outputs_student2 = student_model2(**inputs)

        loss_supervised1 = outputs_student1.loss
        loss_supervised2 = outputs_student2.loss

        # Get consistency loss for both student-teacher pairs
        unlabeled_batch = next(iter(DataLoader(dataset_unlabeled, sampler=RandomSampler(dataset_unlabeled), batch_size=4)))
        unlabeled_batch = tuple(b.to(device) for b in unlabeled_batch)

        with torch.no_grad():
            logits_teacher1 = teacher_model1(input_ids=unlabeled_batch[0], attention_mask=unlabeled_batch[1]).logits
            logits_teacher2 = teacher_model2(input_ids=unlabeled_batch[0], attention_mask=unlabeled_batch[1]).logits

        logits_student1_unlabeled = student_model1(input_ids=unlabeled_batch[0], attention_mask=unlabeled_batch[1]).logits
        logits_student2_unlabeled = student_model2(input_ids=unlabeled_batch[0], attention_mask=unlabeled_batch[1]).logits

        # Consistency loss between students and teachers on unlabeled data
        consistency_loss1 = F.mse_loss(logits_student1_unlabeled, logits_teacher1)
        consistency_loss2 = F.mse_loss(logits_student2_unlabeled, logits_teacher2)

        # Cross-consistency loss between students on unlabeled data
        cross_consistency_loss = F.mse_loss(logits_student1_unlabeled, logits_student2_unlabeled)

        # Total loss = Supervised loss + Consistency loss + Cross-consistency loss
        total_loss = loss_supervised1 + loss_supervised2 + consistency_loss1 + consistency_loss2 + cross_consistency_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(student_model1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(student_model2.parameters(), 1.0)
        # Update parameters
        optimizer1.step()
        optimizer2.step()
        scheduler1.step()
        scheduler2.step()

        loss_train_total += total_loss.item()
        progress_bar.set_postfix({'training_loss': f'{loss_train_total / len(progress_bar):.3f}'})

        # Update teacher models via Exponential Moving Average (EMA)
        with torch.no_grad():
            for teacher_param1, student_param1 in zip(teacher_model1.parameters(), student_model1.parameters()):
                teacher_param1.data.mul_(alpha).add_(student_param1.data, alpha=(1 - alpha))

            for teacher_param2, student_param2 in zip(teacher_model2.parameters(), student_model2.parameters()):
                teacher_param2.data.mul_(alpha).add_(student_param2.data, alpha=(1 - alpha))

    # Validation
    student_model1.eval()
    student_model2.eval()
    teacher_model1.eval()
    teacher_model2.eval()
    loss_val_total = 0
    predictions_student1, predictions_student2, predictions_teacher1, predictions_teacher2, true_vals = [], [], [], [], []
    
    for batch in DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=4):
        batch = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            # Forward pass through student models
            outputs_student1 = student_model1(input_ids=batch[0], attention_mask=batch[1])
            outputs_student2 = student_model2(input_ids=batch[0], attention_mask=batch[1])
            
            # Forward pass through teacher models
            outputs_teacher1 = teacher_model1(input_ids=batch[0], attention_mask=batch[1])
            outputs_teacher2 = teacher_model2(input_ids=batch[0], attention_mask=batch[1])

            # Loss from student_model1 (can add student_model2 loss if necessary)
            loss = F.cross_entropy(outputs_student1.logits, batch[2])
            loss_val_total += loss.item()

            # Extract logits from all models
            logits_student1 = outputs_student1.logits.detach().cpu().numpy()
            logits_student2 = outputs_student2.logits.detach().cpu().numpy()
            logits_teacher1 = outputs_teacher1.logits.detach().cpu().numpy()
            logits_teacher2 = outputs_teacher2.logits.detach().cpu().numpy()
            label_ids = batch[2].cpu().numpy()

            # Collect predictions from students and teachers
            predictions_student1.append(logits_student1)
            predictions_student2.append(logits_student2)
            predictions_teacher1.append(logits_teacher1)
            predictions_teacher2.append(logits_teacher2)
            true_vals.append(label_ids)

    # Concatenate predictions and true labels
    predictions_student1 = np.concatenate(predictions_student1, axis=0)
    predictions_student2 = np.concatenate(predictions_student2, axis=0)
    predictions_teacher1 = np.concatenate(predictions_teacher1, axis=0)
    predictions_teacher2 = np.concatenate(predictions_teacher2, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    # Metrics calculation for all models
    def compute_metrics(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        accuracy = accuracy_score(labels_flat, preds_flat)
        f1 = f1_score(labels_flat, preds_flat, average='weighted')
        precision = precision_score(labels_flat, preds_flat, average='weighted')
        recall = recall_score(labels_flat, preds_flat, average='weighted')
        return accuracy, f1, precision, recall

    # Metrics for student_model1
    accuracy1, f1_score1, precision1, recall1 = compute_metrics(predictions_student1, true_vals)
    
    # Metrics for student_model2
    accuracy2, f1_score2, precision2, recall2 = compute_metrics(predictions_student2, true_vals)
    
    # Metrics for teacher_model1
    accuracy_teacher1, f1_score_teacher1, precision_teacher1, recall_teacher1 = compute_metrics(predictions_teacher1, true_vals)
    
    # Metrics for teacher_model2
    accuracy_teacher2, f1_score_teacher2, precision_teacher2, recall_teacher2 = compute_metrics(predictions_teacher2, true_vals)

    # Print metrics for each student and teacher model
    print(f'Epoch {epoch}: Val Loss: {loss_val_total / len(dataset_test):.3f}')
    
    # Student Model 1
    print(f'Student Model 1 - Accuracy: {accuracy1:.3f}, F1 Score: {f1_score1:.3f}, Precision: {precision1:.3f}, Recall: {recall1:.3f}')
    
    # Student Model 2
    print(f'Student Model 2 - Accuracy: {accuracy2:.3f}, F1 Score: {f1_score2:.3f}, Precision: {precision2:.3f}, Recall: {recall2:.3f}')
    
    # Teacher Model 1
    print(f'Teacher Model 1 - Accuracy: {accuracy_teacher1:.3f}, F1 Score: {f1_score_teacher1:.3f}, Precision: {precision_teacher1:.3f}, Recall: {recall_teacher1:.3f}')
    
    # Teacher Model 2
    print(f'Teacher Model 2 - Accuracy: {accuracy_teacher2:.3f}, F1 Score: {f1_score_teacher2:.3f}, Precision: {precision_teacher2:.3f}, Recall: {recall_teacher2:.3f}')
    
print('Training complete!')



# In[5]:


def compute_metrics(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    accuracy = accuracy_score(labels_flat, preds_flat)
    f1 = f1_score(labels_flat, preds_flat, average='weighted')
    precision = precision_score(labels_flat, preds_flat, average='weighted')
    recall = recall_score(labels_flat, preds_flat, average='weighted')
    return accuracy, f1, precision, recall


# In[6]:


# ---- Testing using the average of both teacher models ----

def test_using_teacher_average(dataset_test):
    teacher_model1.eval()
    teacher_model2.eval()
    
    predictions, true_vals = [], []
    
    for batch in DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=4):
        batch = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            # Forward pass through teacher models
            outputs_teacher1 = teacher_model1(input_ids=batch[0], attention_mask=batch[1])
            outputs_teacher2 = teacher_model2(input_ids=batch[0], attention_mask=batch[1])

            # Average the logits of both teacher models
            avg_logits = (outputs_teacher1.logits + outputs_teacher2.logits) / 2.0

            logits = avg_logits.detach().cpu().numpy()
            label_ids = batch[2].cpu().numpy()

            predictions.append(logits)
            true_vals.append(label_ids)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    
    # Calculate accuracy and other metrics
    accuracy, f1, precision, recall = compute_metrics(predictions, true_vals)
    
    # Print metrics
    print(f'Teacher Model Average - Accuracy: {accuracy:.3f}, F1 Score: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}')

# Call the function to test using teacher average
test_using_teacher_average(dataset_test)


# In[7]:


# ---- Testing using the average of both student models ----

def test_using_student_average(dataset_test):
    student_model1.eval()
    student_model2.eval()
    
    predictions, true_vals = [], []
    
    for batch in DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=4):
        batch = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            # Forward pass through student models
            outputs_student1 = student_model1(input_ids=batch[0], attention_mask=batch[1])
            outputs_student2 = student_model2(input_ids=batch[0], attention_mask=batch[1])

            # Average the logits of both student models
            avg_logits = (outputs_student1.logits + outputs_student2.logits) / 2.0

            logits = avg_logits.detach().cpu().numpy()
            label_ids = batch[2].cpu().numpy()

            predictions.append(logits)
            true_vals.append(label_ids)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    
    # Calculate accuracy and other metrics
    accuracy, f1, precision, recall = compute_metrics(predictions, true_vals)
    
    # Print metrics
    print(f'Student Model Average - Accuracy: {accuracy:.3f}, F1 Score: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}')

# Call the function to test using student average
test_using_student_average(dataset_test)


# In[10]:


# ---- LIME explainability with average of both student models ----
# LIME for explainability: explain the output using the average of both student models
class_names = list(label_dict.keys())
explainer = LimeTextExplainer(class_names=class_names, split_expression='\s+')

def predict_average_students(texts):
    encodings = tokenizer.batch_encode_plus(texts, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True, max_length=256, return_tensors='pt')
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        # Forward pass through both student models
        outputs_student1 = student_model1(input_ids=input_ids, attention_mask=attention_mask)
        outputs_student2 = student_model2(input_ids=input_ids, attention_mask=attention_mask)

        # Average the logits of both student models
        avg_logits = (outputs_student1.logits + outputs_student2.logits) / 2.0

        # Convert logits to probabilities
        probs = F.softmax(avg_logits, dim=1).detach().cpu().numpy()

    return probs

# Explain a random sample from the test set
random_idx = random.randint(0, len(df_test) - 1)
text_sample = df_test.iloc[random_idx][tweets_column]
true_label = df_test.iloc[random_idx][labels_column]
print(f"Sample text: {text_sample}")
print(f"True label: {class_names[true_label]}")

# Use the new prediction function for LIME
exp = explainer.explain_instance(text_sample, predict_average_students, num_features=10,num_samples=100)
exp.show_in_notebook()


# In[ ]:




