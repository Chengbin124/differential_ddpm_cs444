import os
import torch
import sklearn.preprocessing
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from dataset import TextDataset
from transformers import get_scheduler

data_dir = 'text_dataset'
os.makedirs(data_dir, exist_ok=True)


text_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), download_if_missing=True)

for idx, text in enumerate(text_data.data):
    file_path = os.path.join(data_dir, f'textdoc_{idx}.txt')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)


def load_text_data(data_dir):
    texts = []
    labels = []
    for file_name in os.listdir(data_dir):
        index = int(file_name.split('_')[1].split('.')[0])  
        with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as file:
            texts.append(file.read().strip())
        labels.append(text_data.target[index])  

    return texts, labels

texts, labels = load_text_data(data_dir)
max_length = 256
batch_size = 16

label_encoder = sklearn.preprocessing.LabelEncoder()
labels = label_encoder.fit_transform(labels)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_strategy="longest_first")
train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(set(labels)))
device = torch.device("cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

num_epoches = 20
num_training_steps = len(train_loader) * num_epoches
scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
)

def train_one_epoch( data_loader, model, scheduler,optimizer,  criterion, device):
    model.train()
    total_loss = 0
    for bat in data_loader:
        optimizer.zero_grad()
        ids = bat["input_ids"].to(device)
        attention_mask = bat["attention_mask"].to(device)
        labels = bat["labels"].to(device)
        outputs = model(ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  
        total_loss += loss.item()
    return total_loss / len(data_loader)

#def train_one_epoch(model, data_loader, optimizer, criterion, device):
#    model.train()
#    total_loss = 0
#    for batch in data_loader:
#        optimizer.zero_grad()
#        input_ids = batch["input_ids"].to(device)
#        attention_mask = batch["attention_mask"].to(device)
#        labels = batch["labels"].to(device)
#        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#        loss = outputs.loss
#        loss.backward()
#        optimizer.step()
#        total_loss += loss.item()
#    return total_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for bat in data_loader:
            input_ids = bat["input_ids"].to(device)
            attention_mask = bat["attention_mask"].to(device)
            labels = bat["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
    accuracy = correct / len(data_loader.dataset)
    return (total_loss / len(data_loader)), accuracy

num_epochs = 20
for epoch in range(num_epochs):
    train_loss = train_one_epoch(train_loader,model, scheduler, optimizer, criterion, device)
    val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

model.save_pretrained("text_transformer_model")
tokenizer.save_pretrained("text_transformer_model") 
