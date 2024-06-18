import json as js
def accusation_statistics(fp):
    crime_dict={}
    accu_nubmber=0
    for line in fp.readlines():
        info=js.loads(line)
        for accu in info["meta"]["accusation"]:
            if accu not in crime_dict.values():
                crime_dict[accu_nubmber]=accu
                accu_nubmber+=1
    return crime_dict

def main():
    file="/Users/chenhaohua/Desktop/projects/NLP/PRC_legal_match/final_all_data/exercise_contest/data_train.json"
    with open(file,"r") as fp:
        crime_map=accusation_statistics(fp)
        print(crime_map)
    with open("/Users/chenhaohua/Desktop/projects/NLP/PRC_legal_match/final_all_data/accusation_map.js","w+") as nfp:
        js.dump(crime_map,nfp,indent=1,ensure_ascii=False)
if __name__=='__main__':
    main()