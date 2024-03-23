#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install datasets')
get_ipython().system('pip install transformers[torch]')
get_ipython().system('pip install accelerate -U')
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Model(ABC):
    """
    Abstract class for a machine learning model. Whenever it is needed to
    implement a new model it should inherit and implement each of its methods.
    Each inheritted model might be implemented differently but should respect
    the signature of the abstract class.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    @abstractmethod
    def fit(self,
            x_train: pd.Series,
            y_train: pd.Series):
        """
        Abstract fit method that takes training text documents `x_train` and
        their labels `y_train` and train a model. `x_dev` and `y_dev` can be
        used to obtain cross-validation insights, early stopping, or simply
        ignore them.

        parameters:
            - `x_train` (pd.Series[str]) training text documents.
            - `y_train` (pd.Series[int]) training labels.
            - `x_dev` (pd.Series[str]) dev text documents.
            - `y_dev` (pd.Series[int]) dev labels.
        """
        pass

    @abstractmethod
    def predict(self, x: pd.Series) -> np.array:
        """
        Abstract method to perform classification on samples in `x`.

        parameters:
            - `x` (pd.Series[str]) sample to predict.

        returns:
            - `y_pred` (np.array[int]) class labels for sample `x`.
        """
        pass

    @abstractmethod
    def predict_proba(self, x: pd.Series) -> np.array:
        """
        Abstract method to estimate classification probabilities on samples in
        `x`.

        parameters:
            - `x` (pd.Series[str]) sample to predict.

        returns:
            - `y_pred` (np.array of floats with n classes columns) probability
              labels for sample `x`.
        """
        pass

    @abstractmethod
    def save_model(self) -> None:
        """
        Save model weights as a pickle python file in `self.output_dir` using
        its identifier `self.model_name`.
        """
        pass

    @abstractmethod
    def load_model(self, model_dirpath: str) -> None:
        """
        Load model weights. It takes directory path `model_dirpath` where the
        model necessary data is in.

        parameters:
            - `model_dirpath` (str) Directory path where the model is saved.
        """
        pass


# In[11]:


from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          TextClassificationPipeline, TrainingArguments,
                          Trainer, DataCollatorWithPadding)
from datasets import Dataset
import pandas as pd
import numpy as np
import os
import warnings


class TransformerModel(Model):
    """
    Huggingface Transformer model for classification such as BERT, DeBERTa,
    RoBERTa, etc.

    parameters:
        - `output_dir` (str) Directory path where the model outputs will be
          recorded. That is weights, predictions, etc.

        - `model_name` (str) Identifier of the model. It is used to recognize an
          instance of the class. For example, if multiple runs are executed with
          different parameters, `model_name` can be used to assign a different
          name. Also, when saving an instance of the model, it will create a
          directory using this parameters as its name and will be saved in
          `output_dir`.

        - `huggingface-path` (str) the name of the model in the hub of
          huggingface. For example: `bert-base-uncased` or
          `microsoft/deberta-v3-large`.

        - `checkpoint-path` (str) [optional] path to a huggingface checkpoint
        directory containing its configuration.

        - `epochs` (int) number of epochs for training the transformer.

        - `batch-size` (int) batch size used for training the transformer.

        - `random_state` (int) integer number to initialize the random state
          during the training process.

        - `lr` (float) learning rate for training the transformer.

        - `weight-decay` (float) weight decay penalty applied to the
          transformer.

        - `device` (str) Use `cpu` or `gpu`.
    """

    def __init__(self,
                 huggingface_path: str = "google/mt5-base",
                 checkpoint_path: str = None,
                 epochs: int = 4,
                 batch_size: int = 16,
                 random_state: int = 42,
                 lr: float = 2e-5,
                 weight_decay: float = 0.01,
                 num_labels: int = 2,
                 output_dir: str = "./default_output_dir",
                 device: str = "cpu") -> None:
        super(TransformerModel, self).__init__(output_dir)

        # Load model from hugginface hub.
        model = AutoModelForSequenceClassification.from_pretrained(
            huggingface_path,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )

        # Load tokenizer from huggingface hub.
        tokenizer = AutoTokenizer.from_pretrained(huggingface_path,
                                                  do_lower_case=True)
        # Set class attributes.
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_path = checkpoint_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.num_labels = num_labels
        self.args = None
        self.trainer = None

    def set_training_args(self):
        self.args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            seed=self.random_state,
            #data_seed=self.random_state,
            optim="adamw_hf")

    def tokenize(self, example: str):
        """
        Tokenize a sentence using the model tokenizer.
        """
        return self.tokenizer(example["text"], truncation=True)

    def build_loader(self, sentences: pd.Series, labels: pd.Series = None):
        """
        Create a Dataset loader from huggingface tokenizing each sentence.

        parameters:
            - `sentences` (pd.Series[str])
            - `labels` (pd.Series[int])
        """
        dataset_dict = {"text": sentences}
        if labels is not None:
            dataset_dict.update({"label": labels})
    
        dataset = Dataset.from_dict(dataset_dict)
        return dataset.map(self.tokenize, batched=True)

    def fit(self,
            x_train: pd.Series,
            y_train: pd.Series) -> None:
        """
        Fit method that takes training text documents `x_train` and their labels
        `y_train` and train a transformer based model. In this case the `x_dev`
        and `y_dev` are used to evaluate the model in each epoch. When saving
        the model, train and dev losses are saved too.

        parameters:
            - `x_train` (pd.Series[str]) training text documents.
            - `y_train` (pd.Series[int]) training labels.
            - `x_dev` (pd.Series[str]) dev text documents.
            - `y_dev` (pd.Series[int]) dev labels.
        """
        self.set_training_args()

        # Create data collator.
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer,
                                                padding=True)

        train_size = int(0.8 * len(x_train))  # 80-20 split
        train_sentences, eval_sentences = x_train[:train_size], x_train[train_size:]
        train_labels, eval_labels = y_train[:train_size], y_train[train_size:]

        # Create dataset loaders for train and eval sets.
        train_dataset = self.build_loader(sentences=train_sentences, labels=train_labels)
        eval_dataset = self.build_loader(sentences=eval_sentences, labels=eval_labels)

        # Move huggingface model to the device indicated.
        self.model = self.model.to(self.device)

        # Instance huggingface Trainer.
        self.trainer = Trainer(model=self.model,
                               args=self.args,
                               train_dataset=train_dataset,
                               eval_dataset=eval_dataset,
                               tokenizer=self.tokenizer,
                               data_collator=data_collator)

        # If there is any checkpoint provided, training is resumed from it.
        if self.checkpoint_path is not None:
            self.trainer.train(self.checkpoint_path)
        else:
            self.trainer.train()

    def predict_proba(self, x: pd.Series) -> np.array:
        """
        Estimate classification probabilities on samples in `x`.

        parameters:
            - `x` (pd.Series[str]) sample to predict.

        returns:
            - `y_pred` (np.array of floats with n classes columns) probability
              labels for sample `x`.
        """
        # Use text classification pipeline to make predictions.
        pipe = TextClassificationPipeline(model=self.model,
                                          tokenizer=self.tokenizer,
                                          return_all_scores=True,
                                          framework="pt")
        preds = pipe(x.tolist())
        y_prob = np.array([[pred[i]["score"] for i in range(self.num_labels)]
                           for pred in preds])
        return y_prob

    def predict(self, x: pd.Series) -> np.array:
        """
        Perform classification on samples in `x`.

        parameters:
            - `x` (pd.Series[str]) sample to predict.

        returns:
            - `y_pred` (np.array[int]) class labels for sample `x`.
        """
        y_prob = predict_proba(x)
        y_pred = np.argmax(y_prob, axis=1)
        return y_pred

    def save_model(self):
        """
        Save model weights and its configuration in `self.output_dir`. It
        follows huggingface save standards so the model can be re-loaded using
        huggingface `from_pretrained()` functionality.
        """
        if self.trainer is not None:
            os.makedirs(f"{self.output_dir}/model", exist_ok=True)
            self.trainer.save_model(output_dir=f"{self.output_dir}/model")
        else:
            warnings.warn(
                "Method ignored. Trying to save model without training it."
                "Please use `fit` before `save_model`",
                UserWarning,
            )

    def load_model(self, model_dirpath):
        """
        Load model weights. It takes directory path `model_dirpath` where the
        model necessary data is in.

        parameters:
            - `model_dirpath` (str) Directory path where the model is saved.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dirpath)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dirpath)

    def embed(self, x: pd.Series) -> np.array:
        inputs = self.tokenizer(x.tolist(),
                                truncation=True,
                                max_length=256,
                                padding="max_length",
                                return_tensors="pt")
        outputs = self.model(**inputs, output_hidden_states=True)

        # Get the last hidden state
        last_hidden_states = outputs.hidden_states[-1]

        # Get only the CLS token for each instance in `x` (the one used for classification).
        # cls = last_hidden_states[:, 0, :]

        # Detach Pytorch tensor to Numpy array.
        return last_hidden_states.detach().numpy()


# In[3]:


# Step 1: Load data from CSV files
arabic_data = pd.read_excel("Arabic_Depression_10.000_Tweets.xlsx",header=0)
# dev_data = pd.read_csv("/kaggle/input/translated-datasets/malayalam_only_dev.csv")
# test_data = pd.read_csv("Dataset/Transliterated Only/Tamil/tamil_transliterated_test.csv")

arabic_data.columns = ['id','tweet', 'label']

# Remove any leading or trailing spaces from the labels
# = arabic_data['label'].str.strip()

# Split the dataset into train and test sets
# Assuming you want to split it into 80% train and 20% test
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(arabic_data, test_size=0.2, random_state=42)

# Optionally, you can also split a development/validation set if needed
# dev_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Again, ensure labels are stripped of any leading or trailing spaces
# train_data['label'] = train_data['label'].str.strip()
# test_data['label'] = test_data['label'].str.strip()


# In[4]:


train_data.columns = ['id','text', 'label']
# dev_data.columns = ['text', 'label']
test_data.columns = ['id','text', 'label']
train_data['label'] = train_data['label']
# dev_data['label'] = dev_data['label'].str.strip()
test_data['label'] = test_data['label']


# In[5]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_data['label'] = le.fit_transform(train_data['label'])
# dev_data['label'] = le.transform(dev_data['label'])
test_data['label'] = le.transform(test_data['label'])


# In[13]:


import pandas as pd
# Step 2: Preprocess the data, separating sentences and labels
x_train, y_train = train_data["text"], train_data["label"]
# x_dev, y_dev = dev_data["text"], dev_data["label"]
x_test, y_test = test_data["text"], test_data["label"]

# Step 3: Initialize the TransformerModel
model = TransformerModel(huggingface_path="xlm-roberta-base",
                         epochs=4,
                         batch_size=16,
                         random_state=42,
                         lr=2e-5,
                         weight_decay=0.01,
                         num_labels=2,
                         device="cuda")

# # Step 4: Train the model on the training data


# In[14]:


model.fit(x_train, y_train)


# In[15]:


# Set the output directory where you want to save the model
output_dir = "Models/arabic_xlmr.h5"  # Replace this with your desired output directory

# Set the output_dir in the model instance
model.output_dir = output_dir

model.save_model()


# In[16]:


# Instantiate the TransformerModel class
model = TransformerModel()

# Load the saved model from the specified directory
model.load_model("Models/arabic_xlmr.h5/model")


# In[17]:


from tqdm import tqdm  # Import tqdm

# Sample sentences for demonstration
sample_sentences = x_train

# Define batch size
batch_size = 32

# Calculate the number of batches
num_batches = (len(sample_sentences) + batch_size - 1) // batch_size

# Initialize an empty list to store the embeddings
embeddings_list = []

# Use tqdm to create a progress bar
for i in tqdm(range(num_batches), desc="Processing batches", total=num_batches):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
#     sentences_batch = sample_sentences[start_idx:end_idx]

    # Get embeddings for the current batch
#     embeddings_batch = model.embed(sentences_batch)

    # Append the embeddings to the list
    embeddings_list.append(model.embed(sample_sentences[start_idx:end_idx]))

# Concatenate embeddings from all batches into a single numpy array.
embeddings_list = np.concatenate(embeddings_list, axis=0)
# embeddings = embeddings.reshape(256,768)

# The `embeddings` variable now contains the embeddings for all sentences.
# It will be a numpy array with shape (number_of_sentences, embedding_dim).
# You can use these embeddings for further downstream tasks or analysis.
print(embeddings_list.shape)


# In[18]:


embeddings_list.shape


# In[19]:


import os

# Define the directory path
directory = "Embeddings/MT5/ARABIC"

# Create the directory if it does not exist
if not os.path.exists(directory):
    os.makedirs(directory)


# In[21]:


# Save the embeddings to a file
# Embeddings/Muril/Tamil/Transliterated Only/tamil_muril_transliterated_train_embeds.npy
np.save("Embeddings/MT5/ARABIC/arabic_mt5_transliterated_train_embeds.npy", embeddings_list)


# In[ ]:




