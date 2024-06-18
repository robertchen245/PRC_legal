from transformers import AutoTokenizer, AutoModelForSequenceClassification,BertForSequenceClassification
from transformers import pipeline
import torch.nn as nn
from matplotlib import pyplot as plt
from dataset_extractor import *
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import get_scheduler
from torch.optim import AdamW
def main():
    tokenizer=AutoTokenizer.from_pretrained('/data/chenhaohua/PRC_legal/bert-base-chinese')
    model_reg=AutoModelForSequenceClassification.from_pretrained('/data/chenhaohua/PRC_legal/bert-base-legal-chinese-epoch-8')
    model_reg.classifier = nn.Linear(model_reg.config.hidden_size, 1) 
    criterion = nn.MSELoss()
    for name, param in model_reg.named_parameters():
        param.requires_grad=False
        if 'bert.encoder.layer.6.attention.self.query.weight' in name:
            break
    for name, param in model_reg.named_parameters():
        if param.requires_grad==True:
            print(name)
    batch_size=60
    train_pair=extract("/data/chenhaohua/PRC_legal_dataset/data_train.json")
    print(train_pair["imprisonment"][0])
    train_batch=tokenizer(train_pair["content"],max_length=512,truncation=True,padding="max_length",return_tensors="pt")
    train=TensorDataset(train_batch["input_ids"],train_batch["attention_mask"],torch.tensor(train_pair["imprisonment"],dtype=torch.float32))
    train_sampler=RandomSampler(train)
    train_dataloader=DataLoader(train,sampler=train_sampler,batch_size=batch_size)
    optimizer=AdamW(model_reg.parameters(),lr=2e-5)
    num_epochs=50
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    def train(model):
        model.to(device)
        for epoch in range(num_epochs):
            model.train() #切换成训练模式
            total_loss=0
            for step,batch in enumerate(train_dataloader):
                if step % 10 == 0 and not step == 0:
                    print("step: ",step, "  loss:",total_loss/(step*batch_size))
                b_input_ids=batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                model.zero_grad()
                outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
                loss = criterion(outputs.logits.reshape(-1), b_labels)     # include cross-entropy loss or MSE loss when label=1
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #防止梯度爆炸
                optimizer.step()
                lr_scheduler.step()
            avg_train_loss = total_loss / len(train_dataloader)      
            print("avg_loss:",avg_train_loss)
            model.save_pretrained(f"bert-base-legal-chinese-regression-frozen_first_6layer-epoch-{epoch+1}")
    train(model_reg)
if __name__=="__main__":
    device=torch.device("cuda")
    main()