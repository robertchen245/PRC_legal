import json as js
from collections import Counter

def most_frequent_element(arr):
    # 使用Counter统计数组中每个元素的频数
    counter = Counter(arr)
    
    # 使用Counter的most_common方法找到频数最高的元素及其频数
    most_common_element, frequency = counter.most_common(1)[0]

    return most_common_element
accu_article_map={}
with open("/data/chenhaohua/PRC_legal_dataset/data_train.json",'r') as fp:
    for line_num,line in enumerate(fp.readlines()):
        data=js.loads(line)
        if data["meta"]['accusation'][0] not in accu_article_map:
            accu_article_map[data["meta"]['accusation'][0]]=[]
            accu_article_map[data["meta"]['accusation'][0]]+=(data["meta"]["relevant_articles"])
        elif data["meta"]["relevant_articles"] not in accu_article_map[data["meta"]['accusation'][0]]:
            accu_article_map[data["meta"]['accusation'][0]]+=(data["meta"]["relevant_articles"])
for key in accu_article_map:
    accu_article_map[key]= most_frequent_element(accu_article_map[key])
with open("/data/chenhaohua/PRC_legal_dataset/data_train_accu_article.json",'w') as fp:
    js.dump(accu_article_map,fp,indent=1,ensure_ascii=False)

