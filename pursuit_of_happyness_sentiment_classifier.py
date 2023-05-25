# -*- coding: utf-8 -*-
"""Pursuit of Happyness - Sentiment classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DWUsJZITdOAfX3B1YRf89v13DR6VQxjP
"""

from google.colab import drive
drive.mount('/content/drive')

#  install all the requirements
!pip install transformers
!pip install emoji
!pip install nltk

MH_path = "Model/train_MH"
NMH_path = "Model/train_NMH"

# Open the file in read mode
with open(MH_path, 'r', encoding="utf-8") as file:
    # Read the lines of the file and store them in a list
    MHtext = file.readlines()

# Print the lines
print(len(MHtext))
MHlabels = [1]*len(MHtext)

# Open the file in read mode
with open(NMH_path, 'r', encoding="utf-8") as file:
    # Read the lines of the file and store them in a list
    NMHtext = file.readlines()

# Print the lines
print(len(NMHtext))
NMHlabels = [0]*len(NMHtext)

from sklearn.model_selection import train_test_split

texts = MHtext + NMHtext
labels = MHlabels + NMHlabels

# Split data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

print(f"Train data: {len(train_texts)}")
print(f"Test data: {len(test_texts)}")


"""
Binary Sentiment classifier
"""

# Import the libraries
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import DataLoader, Dataset

# Define the Sentiment Analysis Model
class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Freeze the BERT model's parameters
        for param in bert_model.parameters():
            param.requires_grad = False
        self.bert = bert_model              # DistilBert
        self.dropout = nn.Dropout(0.1)      # Dropout layer to prevent overfitting
        self.linear = nn.Linear(768, 1)     # linear layer
        # self.linear = nn.Linear(512, 1)   #MobileBert
        self.sigmoid = nn.Sigmoid()         # Sigmoid activation function

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)     # Feed input to BERT
        pooled_output = outputs.last_hidden_state[:, 0, :]                          # Use the [CLS] token representation
        dropout_output = self.dropout(pooled_output)                                # Pass the pooled output to the dropout layer
        logits = self.linear(dropout_output)                                        # Pass the dropout output to the linear layer
        probabilities = self.sigmoid(logits)                                        # Pass the logits to the sigmoid activation function
        return probabilities

# Custom Dataset for Sentiment Analysis
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts              # Input texts
        self.labels = labels            # Labels
        self.tokenizer = tokenizer      # Tokenizer for encoding the text
        self.max_len = max_len          # Maximum length of the tokenized input

    # Return the length of the dataset
    def __len__(self):
        return len(self.texts)
    
    # Create a tokenized input and its corresponding label
    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,                           # Text to be tokenized
            add_special_tokens=True,        # Add special tokens
            max_length=self.max_len,        # Maximum length of the tokenized input
            padding='max_length',           # Pad the sequences to the maximum length
            truncation=True,                # Truncate the input to the maximum length
            return_attention_mask=True,     # Return the attention mask
            return_tensors='pt'             # Return PyTorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
batch_size = 64    # Number of samples in each batch
max_length = 512    # Max length of the text that can go to BERT
learning_rate = 16e-4   # Learning rate
num_epochs = 1  # Number of epochs

# Prepare the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')  # Load the DistilBERT tokenizer

# Initialize the model
model = SentimentClassifier()   # Sentiment classifier
model.to(device)    # Push the model to GPU

# Prepare the data
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)  # Training dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # Training dataloader

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.linear.parameters(), lr=learning_rate)   # Adam optimizer
criterion = nn.BCELoss()    # Cross entropy loss
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.5)  # Scheduler for the learning rate

# Training loop
for epoch in range(num_epochs):
    model.train()   # Set the model to training mode
    total_loss = 0  # Total loss
    intermediate_loss = 0   # Intermediate loss
    i = 0
    for batch in train_loader:
        # print(f"Batch {i}")
        i += 1
        input_ids = batch['input_ids'].to(device)               # Push the batch to GPU
        attention_mask = batch['attention_mask'].to(device)     # Push the batch to GPU
        labels = batch['label'].to(device)                      # Push the batch to GPU

        optimizer.zero_grad()                               # Clear the previous gradients
        probabilities = model(input_ids, attention_mask)    # Feed the input to the model
        loss = criterion(probabilities.squeeze(), labels)   # Calculate the loss

        loss.backward()     # Backpropagate the loss
        optimizer.step()    # Update the weights

        loss_item = loss.item()             # Get the loss value from the loss tensor
        total_loss += loss_item             # Add the loss for this batch to the total loss
        intermediate_loss += loss_item      # Add the loss for this batch to the intermediate loss

        if i%100 == 0:
            print(f"Batch average loss: {intermediate_loss/200} | Learning Rate: {scheduler.get_last_lr()[0]}")
            intermediate_loss = 0
            if i%500 == 0:
                scheduler.step()

    average_loss = total_loss / len(train_loader)                                   # Calculate the average loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}')      # Print the average loss

# - test_texts: List of test texts
# - test_labels: List of corresponding test labels (0 or 1)
# - tokenizer: Tokenizer instance for text encoding
# - max_length: Maximum sequence length for encoding

# Create the test dataset
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length)     # Test dataset
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)        # Test dataloader

# Evaluate the model
model.eval()        # Set the model to evaluation mode
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)               # Push the batch to GPU
        attention_mask = batch['attention_mask'].to(device)     # Push the batch to GPU
        labels = batch['label'].to(device)                      # Push the batch to GPU

        probabilities = model(input_ids, attention_mask)                        # Feed the input to the model
        predicted_labels = (probabilities > 0.5).squeeze().cpu().numpy()        # Get the predicted labels

        predictions.extend(predicted_labels)        # Append the predicted labels to a list
        true_labels.extend(labels.cpu().numpy())    # Append the true labels to a list

# Calculate evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(true_labels, predictions)         # Calculate the accuracy
precision = precision_score(true_labels, predictions)       # Calculate the precision
recall = recall_score(true_labels, predictions)             # Calculate the recall
f1 = f1_score(true_labels, predictions)                     # Calculate the F1-score

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Specify the file path where you want to save the weights
weights_path = "Model/model_weights.pth"

# Save the model's state_dict (containing the weights)
torch.save(model.state_dict(), weights_path)

print("Model weights saved successfully!")

# Download the saved model to your local machine
from google.colab import files
files.download(weights_path)

# Load the weights into the model
model.load_state_dict(torch.load(weights_path))

# Set the model in evaluation mode
model.eval()

# import libraries and modules
import os
from utils import preprocessRec

# Function to evaluate records from files
def evaluate_records_from_universities(files):
    for file in files:
        sum = 0
        count = 0
        file_path = os.path.join(directory, file)                           # File path
        uni = file[:-10]                                                    # University name
        output_file_path = os.path.join(out_directory, uni+"scores")        # Output file path
        print(f"Evaluating {file}, writing into {output_file_path}..")      # Print the file being evaluated
        fo = open(output_file_path, 'w')
        with open(file_path, 'r') as file:
            for line in file:
                # Preprocess the line if needed
                rec = preprocessRec(line, True)     # Preprocess the record
                text = rec[1]                       # Get the text

                # Encode the text
                inputs = tokenizer.encode_plus(
                    text,                           # Sentence to encode.
                    add_special_tokens=True,        # Add '[CLS]' and '[SEP]'
                    max_length=max_length,          # Pad & truncate all sentences.
                    padding='max_length',           # Pad all sentences.
                    truncation=True,                # Truncate all sentences.
                    return_attention_mask=True,     # Construct attention masks.
                    return_tensors='pt'             # Return pytorch tensors.
                )

                # Push the input to GPU
                input_ids = inputs['input_ids'].to(device)

                # Push the attention mask to GPU
                attention_mask = inputs['attention_mask'].to(device)

                # Forward pass through the model
                with torch.no_grad():
                    logit = model(input_ids, attention_mask).to('cpu').item()       # Get the logit
                sum += logit                                                        # Add the logit to the sum
                count += 1                                                          # Increment the count
                fo.write(f"{uni}DELIM{rec[0]}DELIM{rec[1]}DELIM{logit}\n")          # Write the record to the output file
        print(f"University: {uni}, Score: {sum/count}\n")                           # Print the average score
        fo.close()


directory = "Data/University/"
out_directory = "Data/University-Scores/"

# List of files in the directory
files = os.listdir(directory)                   

# Evaluate records from files
evaluate_records_from_universities(files)

# Import the combineFiles module
import combineFiles                             

score_dir = "Data/University-Scores/"                                           # Directory containing the scores
files = os.listdir(score_dir)                                                   # List of files in the directory
combineFiles(score_dir, files, "Data/University-Scores/combined-scores")        # Combine the scores into a single file