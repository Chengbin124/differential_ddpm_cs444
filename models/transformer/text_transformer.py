import os
import torch
import sklearn.preprocessing
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from dataset import TextDataset
from transformers import get_scheduler

text_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), download_if_missing=True)

texts = text_data.data
labels = text_data.target
#some preprocessing
maxLen = 512
#change to int
label_num_encoder = sklearn.preprocessing.LabelEncoder()
labels = label_num_encoder.fit_transform(labels)
#download some pre-trained BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_strategy="longest_first")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(set(labels)))
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay = 1e-2)

loss_function = torch.nn.CrossEntropyLoss()

#prepare data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
train_data = TextDataset(train_texts, train_labels, tokenizer, maxLen)
val_data = TextDataset(val_texts, val_labels, tokenizer, maxLen)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
scheduler = get_scheduler(
    "cosine", optimizer=optimizer, num_warmup_steps=100, num_training_steps=len(train_loader) * 20
)
val_loader = DataLoader(val_data, batch_size=16)
device = torch.device("cpu")
model.to(device)

def train_one_epoch(device):
    model.train()
    total_loss = 0
    for bat in train_loader:
        optimizer.zero_grad()
        outputs = model(bat["id"].to(device), attention_mask=bat["attention_mask"].to(device), labels=bat["labels"].to(device))
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()  
        total_loss += loss.item()
    return total_loss / 16


model.eval()
for epoch in range(20):
    train_loss = train_one_epoch(device)
    val_loss = 0
    cor_label = 0
    for bat in val_loader:
        outputs = model(bat["id"].to(device), attention_mask=bat["attention_mask"].to(device), labels=bat["labels"].to(device))
        loss = outputs.loss
        val_loss += loss.item()
        correct += (preds == outputs.logits.argmax(dim=-1)).sum().item()
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

