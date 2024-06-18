from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer
import torch
import numpy as np
from dataset_extractor import *
from utils import *
import json as js
def infer_accusation(model,case_text,tokenizer,device=None):
    case_text=tokenizer(case_text,padding=True,return_tensors='pt')
    if device:
        model.to(device)
        case_text=case_text.to(device)
    model.eval()
    output=model(**case_text)
    output_logits = output.logits.to("cpu")
    prediction=np.array(torch.argmax(output_logits,dim=1))[0]
    return accusation_table[prediction],article_to_detail[str(accu_to_article[accusation_table[prediction]])]
def infer_imprisonment(model,case_text,tokenizer,device=None):
    case_text=tokenizer(case_text,padding=True,return_tensors='pt')
    if device:
        model.to(device)
        case_text=case_text.to(device)
    model.eval()
    output=model(**case_text)
    output_logits = output.logits.to("cpu").detach().numpy()
    prediction=output_logits[0][0]
    return prediction
def criminal_recognition(model,case_text,tokenizer,device=None):
    case_text=tokenizer(case_text,padding=True,return_tensors='pt')
    if device:
        model.to(device)
        case_text=case_text.to(device)
    model.eval()
    output=model(**case_text)
    entity_list = output.logits.to("cpu").detach().numpy()
    entity_list=np.argmax(entity_list,axis=2)
    criminals=NERextract(entity_list[0],case_text['input_ids'][0],tokenizer)
    return criminals
def case_analyze(path_dataset,path_tok,path_cls,path_imp,path_rec,num_entries=1,device=None):
    model_cls=AutoModelForSequenceClassification.from_pretrained(path_cls)
    model_imp=AutoModelForSequenceClassification.from_pretrained(path_imp)
    model_rec=AutoModelForTokenClassification.from_pretrained(path_rec)
    tokenizer=AutoTokenizer.from_pretrained(path_tok)
    dataset=extract(path_dataset,num_entries=num_entries)
    case_result=[]
    for idx in range(num_entries):
        if idx%10==0:
            print(f'{idx} case analyzed')
        case_text=dataset['content'][idx]
        accu,article=infer_accusation(model_cls,case_text,tokenizer,device=device)
        imp=infer_imprisonment(model_imp,case_text,tokenizer,device=device)
        criminal=criminal_recognition(model_rec,case_text,tokenizer,device=device)
        case_result.append({"案情事实":case_text,"犯罪嫌疑人识别":criminal,"预测罪名":accu,"匹配条文":article,"预测刑期":[float(imp),int(round(imp))]})
    return case_result
if __name__=="__main__":
    with torch.no_grad():
        case_dict=case_analyze('../PRC_legal_dataset/data_valid.json',
                           'bert-base-chinese',
                           'bert-base-legal-chinese-epoch-8',
                           'bert-base-legal-chinese-regression-epoch-18',
                           'bert-base-legal-chinese-NER-epoch8',
                           num_entries=50
                           )
    with open("Sample_result.json","w") as fp:
        js.dump(case_dict,fp,ensure_ascii=False,indent=1)