import json as js
import re


def extract(path: str, info: str = "criminal", num_entries: int = None) -> dict:
    regularized_content: str
    stop_words = [str(2000 + i) + "年" for i in range(23)] + [
        "审理查明",
        "人民检察院指控",
        "（",
        "）",
        "。",
        "，",
        "？",
        "(",
        ")",
        "早晨",
        "凌晨",
        "清晨",
        "晚上",
        "*",
        "：",
        ",",
        ".",
        "?",
        "、",
        "《",
        "》",
    ]
    table = {
        "故意伤害": 0,
        "盗窃": 1,
        "危险驾驶": 2,
        "非法[持有、私藏][枪支、弹药]": 3,
        "交通肇事": 4,
        "寻衅滋事": 5,
        "[窝藏、包庇]": 6,
        "放火": 7,
        "故意毁坏财物": 8,
        "绑架": 9,
        "赌博": 10,
        "妨害公务": 11,
        "合同诈骗": 12,
        "[走私、贩卖、运输、制造]毒品": 13,
        "抢劫": 14,
        "非法拘禁": 15,
        "诬告陷害": 16,
        "非法采矿": 17,
        "容留他人吸毒": 18,
        "强奸": 19,
        "[伪造、变造、买卖]国家机关[公文、证件、印章]": 20,
        "故意杀人": 21,
        "诈骗": 22,
        "聚众斗殴": 23,
        "[掩饰、隐瞒][犯罪所得、犯罪所得收益]": 24,
        "敲诈勒索": 25,
        "[组织、强迫、引诱、容留、介绍]卖淫": 26,
        "[引诱、容留、介绍]卖淫": 27,
        "开设赌场": 28,
        "重大责任事故": 29,
        "抢夺": 30,
        "破坏电力设备": 31,
        "[制造、贩卖、传播]淫秽物品": 32,
        "传播淫秽物品": 33,
        "虐待": 34,
        "非法[采伐、毁坏]国家重点保护植物": 35,
        "非法[制造、买卖、运输、邮寄、储存][枪支、弹药、爆炸物]": 36,
        "受贿": 37,
        "脱逃": 38,
        "行贿": 39,
        "破坏[广播电视设施、公用电信设施]": 40,
        "[伪造、变造]居民身份证": 41,
        "拐卖[妇女、儿童]": 42,
        "强迫交易": 43,
        "拒不支付劳动报酬": 44,
        "帮助[毁灭、伪造]证据": 45,
        "爆炸": 46,
        "污染环境": 47,
        "非法持有毒品": 48,
        "破坏易燃易爆设备": 49,
        "妨害信用卡管理": 50,
        "[引诱、教唆、欺骗]他人吸毒": 51,
        "非法处置[查封、扣押、冻结]的财产": 52,
        "贪污": 53,
        "职务侵占": 54,
        "帮助犯罪分子逃避处罚": 55,
        "盗伐林木": 56,
        "挪用资金": 57,
        "重婚": 58,
        "侵占": 59,
        "[窝藏、转移、收购、销售]赃物": 60,
        "妨害作证": 61,
        "挪用公款": 62,
        "伪造[公司、企业、事业单位、人民团体]印章": 63,
        "[窝藏、转移、隐瞒][毒品、毒赃]": 64,
        "[虚开增值税专用发票、用于骗取出口退税、抵扣税款发票]": 65,
        "非法侵入住宅": 66,
        "信用卡诈骗": 67,
        "非法获取公民个人信息": 68,
        "滥伐林木": 69,
        "非法经营": 70,
        "招摇撞骗": 71,
        "以危险方法危害公共安全": 72,
        "[盗窃、侮辱]尸体": 73,
        "过失致人死亡": 74,
        "[持有、使用]假币": 75,
        "传授犯罪方法": 76,
        "猥亵儿童": 77,
        "逃税": 78,
        "非法吸收公众存款": 79,
        "非法[转让、倒卖]土地使用权": 80,
        "骗取[贷款、票据承兑、金融票证]": 81,
        "破坏生产经营": 82,
        "高利转贷": 83,
        "[盗窃、抢夺][枪支、弹药、爆炸物]": 84,
        "[盗窃、抢夺][枪支、弹药、爆炸物、危险物质]": 85,
        "假冒注册商标": 86,
        "[伪造、变造]金融票证": 87,
        "强迫卖淫": 88,
        "扰乱无线电通讯管理秩序": 89,
        "虚开发票": 90,
        "非法占用农用地": 91,
        "[组织、领导、参加]黑社会性质组织": 92,
        "[隐匿、故意销毁][会计凭证、会计帐簿、财务会计报告]": 93,
        "保险诈骗": 94,
        "强制[猥亵、侮辱]妇女": 95,
        "非国家工作人员受贿": 96,
        "伪造货币": 97,
        "拒不执行[判决、裁定]": 98,
        "[生产、销售]伪劣产品": 99,
        "非法[收购、运输][盗伐、滥伐]的林木": 100,
        "冒充军人招摇撞骗": 101,
        "组织卖淫": 102,
        "持有伪造的发票": 103,
        "[生产、销售][有毒、有害]食品": 104,
        "非法[制造、出售]非法制造的发票": 105,
        "[伪造、变造、买卖]武装部队[公文、证件、印章]": 106,
        "[组织、领导]传销活动": 107,
        "强迫劳动": 108,
        "走私": 109,
        "贷款诈骗": 110,
        "串通投标": 111,
        "虚报注册资本": 112,
        "侮辱": 113,
        "伪证": 114,
        "聚众扰乱社会秩序": 115,
        "聚众扰乱[公共场所秩序、交通秩序]": 116,
        "劫持[船只、汽车]": 117,
        "集资诈骗": 118,
        "盗掘[古文化遗址、古墓葬]": 119,
        "失火": 120,
        "票据诈骗": 121,
        "经济犯": 122,
        "单位行贿": 123,
        "投放危险物质": 124,
        "过失致人重伤": 125,
        "破坏交通设施": 126,
        "聚众哄抢": 127,
        "走私普通[货物、物品]": 128,
        "收买被拐卖的[妇女、儿童]": 129,
        "非法狩猎": 130,
        "销售假冒注册商标的商品": 131,
        "破坏监管秩序": 132,
        "拐骗儿童": 133,
        "非法行医": 134,
        "协助组织卖淫": 135,
        "打击报复证人": 136,
        "强迫他人吸毒": 137,
        "非法[收购、运输、加工、出售][国家重点保护植物、国家重点保护植物制品]": 138,
        "[生产、销售]不符合安全标准的食品": 139,
        "非法买卖制毒物品": 140,
        "滥用职权": 141,
        "聚众冲击国家机关": 142,
        "[出售、购买、运输]假币": 143,
        "对非国家工作人员行贿": 144,
        "[编造、故意传播]虚假恐怖信息": 145,
        "玩忽职守": 146,
        "私分国有资产": 147,
        "非法携带[枪支、弹药、管制刀具、危险物品]危及公共安全": 148,
        "过失以危险方法危害公共安全": 149,
        "走私国家禁止进出口的[货物、物品]": 150,
        "违法发放贷款": 151,
        "徇私枉法": 152,
        "非法[买卖、运输、携带、持有]毒品原植物[种子、幼苗]": 153,
        "动植物检疫徇私舞弊": 154,
        "重大劳动安全事故": 155,
        "走私[武器、弹药]": 156,
        "破坏计算机信息系统": 157,
        "[制作、复制、出版、贩卖、传播]淫秽物品牟利": 158,
        "单位受贿": 159,
        "[生产、销售]伪劣[农药、兽药、化肥、种子]": 160,
        "过失损坏[武器装备、军事设施、军事通信]": 161,
        "破坏交通工具": 162,
        "包庇毒品犯罪分子": 163,
        "[生产、销售]假药": 164,
        "非法种植毒品原植物": 165,
        "诽谤": 166,
        "传播性病": 167,
        "介绍贿赂": 168,
        "金融凭证诈骗": 169,
        "非法[猎捕、杀害][珍贵、濒危]野生动物": 170,
        "徇私舞弊不移交刑事案件": 171,
        "巨额财产来源不明": 172,
        "过失损坏[广播电视设施、公用电信设施]": 173,
        "挪用特定款物": 174,
        "[窃取、收买、非法提供]信用卡信息": 175,
        "非法组织卖血": 176,
        "利用影响力受贿": 177,
        "非法捕捞水产品": 178,
        "对单位行贿": 179,
        "遗弃": 180,
        "徇私舞弊[不征、少征]税款": 181,
        "提供[侵入、非法控制计算机信息系统][程序、工具]": 182,
        "非法进行节育手术": 183,
        "危险物品肇事": 184,
        "非法[制造、买卖、运输、储存]危险物质": 185,
        "非法[制造、销售]非法制造的注册商标标识": 186,
        "侵犯著作权": 187,
        "倒卖[车票、船票]": 188,
        "过失投放危险物质": 189,
        "走私废物": 190,
        "非法出售发票": 191,
        "走私[珍贵动物、珍贵动物制品]": 192,
        "[伪造、倒卖]伪造的有价票证": 193,
        "招收[公务员、学生]徇私舞弊": 194,
        "非法[生产、销售]间谍专用器材": 195,
        "倒卖文物": 196,
        "虐待被监管人": 197,
        "洗钱": 198,
        "非法[生产、买卖]警用装备": 199,
        "非法获取国家秘密": 200,
        "非法[收购、运输、出售][珍贵、濒危野生动物、珍贵、濒危野生动物]制品": 201,
    }
    pair = {
        "content": [],
        "label": [],
        "articles": [],
        "punish_of_money": [],
        "criminals": [],
        "death_penalty": [],
        "imprisonment": [],
        "life_imprisonment": [],
        "NER": [],
    }
    with open(path, "r") as fp:
        for line_num, line in enumerate(fp.readlines()):
            if num_entries != None and len(pair["content"]) >= num_entries:
                break
            line_json = js.loads(line)
            regularized_content = line_json["fact"]
            for item in stop_words:
                regularized_content = regularized_content.replace(item, "")
            if len(regularized_content) < 512:
                pair["content"].append(regularized_content)
                pair["label"].append(table[line_json["meta"]["accusation"][0]])
                pair["articles"].append(line_json["meta"]["relevant_articles"])
                pair["punish_of_money"].append(line_json["meta"]["punish_of_money"])
                pair["criminals"].append(line_json["meta"]["criminals"])
                pair["death_penalty"].append(
                    line_json["meta"]["term_of_imprisonment"]["death_penalty"]
                )
                pair["imprisonment"].append(
                    line_json["meta"]["term_of_imprisonment"]["imprisonment"]
                )
                pair["life_imprisonment"].append(
                    line_json["meta"]["term_of_imprisonment"]["life_imprisonment"]
                )
        print(f"{len(pair['content'])} lines have been extracted")
        return pair


accusation_table = {
    0: "故意伤害",
    1: "盗窃",
    2: "危险驾驶",
    3: "非法[持有、私藏][枪支、弹药]",
    4: "交通肇事",
    5: "寻衅滋事",
    6: "[窝藏、包庇]",
    7: "放火",
    8: "故意毁坏财物",
    9: "绑架",
    10: "赌博",
    11: "妨害公务",
    12: "合同诈骗",
    13: "[走私、贩卖、运输、制造]毒品",
    14: "抢劫",
    15: "非法拘禁",
    16: "诬告陷害",
    17: "非法采矿",
    18: "容留他人吸毒",
    19: "强奸",
    20: "[伪造、变造、买卖]国家机关[公文、证件、印章]",
    21: "故意杀人",
    22: "诈骗",
    23: "聚众斗殴",
    24: "[掩饰、隐瞒][犯罪所得、犯罪所得收益]",
    25: "敲诈勒索",
    26: "[组织、强迫、引诱、容留、介绍]卖淫",
    27: "[引诱、容留、介绍]卖淫",
    28: "开设赌场",
    29: "重大责任事故",
    30: "抢夺",
    31: "破坏电力设备",
    32: "[制造、贩卖、传播]淫秽物品",
    33: "传播淫秽物品",
    34: "虐待",
    35: "非法[采伐、毁坏]国家重点保护植物",
    36: "非法[制造、买卖、运输、邮寄、储存][枪支、弹药、爆炸物]",
    37: "受贿",
    38: "脱逃",
    39: "行贿",
    40: "破坏[广播电视设施、公用电信设施]",
    41: "[伪造、变造]居民身份证",
    42: "拐卖[妇女、儿童]",
    43: "强迫交易",
    44: "拒不支付劳动报酬",
    45: "帮助[毁灭、伪造]证据",
    46: "爆炸",
    47: "污染环境",
    48: "非法持有毒品",
    49: "破坏易燃易爆设备",
    50: "妨害信用卡管理",
    51: "[引诱、教唆、欺骗]他人吸毒",
    52: "非法处置[查封、扣押、冻结]的财产",
    53: "贪污",
    54: "职务侵占",
    55: "帮助犯罪分子逃避处罚",
    56: "盗伐林木",
    57: "挪用资金",
    58: "重婚",
    59: "侵占",
    60: "[窝藏、转移、收购、销售]赃物",
    61: "妨害作证",
    62: "挪用公款",
    63: "伪造[公司、企业、事业单位、人民团体]印章",
    64: "[窝藏、转移、隐瞒][毒品、毒赃]",
    65: "[虚开增值税专用发票、用于骗取出口退税、抵扣税款发票]",
    66: "非法侵入住宅",
    67: "信用卡诈骗",
    68: "非法获取公民个人信息",
    69: "滥伐林木",
    70: "非法经营",
    71: "招摇撞骗",
    72: "以危险方法危害公共安全",
    73: "[盗窃、侮辱]尸体",
    74: "过失致人死亡",
    75: "[持有、使用]假币",
    76: "传授犯罪方法",
    77: "猥亵儿童",
    78: "逃税",
    79: "非法吸收公众存款",
    80: "非法[转让、倒卖]土地使用权",
    81: "骗取[贷款、票据承兑、金融票证]",
    82: "破坏生产经营",
    83: "高利转贷",
    84: "[盗窃、抢夺][枪支、弹药、爆炸物]",
    85: "[盗窃、抢夺][枪支、弹药、爆炸物、危险物质]",
    86: "假冒注册商标",
    87: "[伪造、变造]金融票证",
    88: "强迫卖淫",
    89: "扰乱无线电通讯管理秩序",
    90: "虚开发票",
    91: "非法占用农用地",
    92: "[组织、领导、参加]黑社会性质组织",
    93: "[隐匿、故意销毁][会计凭证、会计帐簿、财务会计报告]",
    94: "保险诈骗",
    95: "强制[猥亵、侮辱]妇女",
    96: "非国家工作人员受贿",
    97: "伪造货币",
    98: "拒不执行[判决、裁定]",
    99: "[生产、销售]伪劣产品",
    100: "非法[收购、运输][盗伐、滥伐]的林木",
    101: "冒充军人招摇撞骗",
    102: "组织卖淫",
    103: "持有伪造的发票",
    104: "[生产、销售][有毒、有害]食品",
    105: "非法[制造、出售]非法制造的发票",
    106: "[伪造、变造、买卖]武装部队[公文、证件、印章]",
    107: "[组织、领导]传销活动",
    108: "强迫劳动",
    109: "走私",
    110: "贷款诈骗",
    111: "串通投标",
    112: "虚报注册资本",
    113: "侮辱",
    114: "伪证",
    115: "聚众扰乱社会秩序",
    116: "聚众扰乱[公共场所秩序、交通秩序]",
    117: "劫持[船只、汽车]",
    118: "集资诈骗",
    119: "盗掘[古文化遗址、古墓葬]",
    120: "失火",
    121: "票据诈骗",
    122: "经济犯",
    123: "单位行贿",
    124: "投放危险物质",
    125: "过失致人重伤",
    126: "破坏交通设施",
    127: "聚众哄抢",
    128: "走私普通[货物、物品]",
    129: "收买被拐卖的[妇女、儿童]",
    130: "非法狩猎",
    131: "销售假冒注册商标的商品",
    132: "破坏监管秩序",
    133: "拐骗儿童",
    134: "非法行医",
    135: "协助组织卖淫",
    136: "打击报复证人",
    137: "强迫他人吸毒",
    138: "非法[收购、运输、加工、出售][国家重点保护植物、国家重点保护植物制品]",
    139: "[生产、销售]不符合安全标准的食品",
    140: "非法买卖制毒物品",
    141: "滥用职权",
    142: "聚众冲击国家机关",
    143: "[出售、购买、运输]假币",
    144: "对非国家工作人员行贿",
    145: "[编造、故意传播]虚假恐怖信息",
    146: "玩忽职守",
    147: "私分国有资产",
    148: "非法携带[枪支、弹药、管制刀具、危险物品]危及公共安全",
    149: "过失以危险方法危害公共安全",
    150: "走私国家禁止进出口的[货物、物品]",
    151: "违法发放贷款",
    152: "徇私枉法",
    153: "非法[买卖、运输、携带、持有]毒品原植物[种子、幼苗]",
    154: "动植物检疫徇私舞弊",
    155: "重大劳动安全事故",
    156: "走私[武器、弹药]",
    157: "破坏计算机信息系统",
    158: "[制作、复制、出版、贩卖、传播]淫秽物品牟利",
    159: "单位受贿",
    160: "[生产、销售]伪劣[农药、兽药、化肥、种子]",
    161: "过失损坏[武器装备、军事设施、军事通信]",
    162: "破坏交通工具",
    163: "包庇毒品犯罪分子",
    164: "[生产、销售]假药",
    165: "非法种植毒品原植物",
    166: "诽谤",
    167: "传播性病",
    168: "介绍贿赂",
    169: "金融凭证诈骗",
    170: "非法[猎捕、杀害][珍贵、濒危]野生动物",
    171: "徇私舞弊不移交刑事案件",
    172: "巨额财产来源不明",
    173: "过失损坏[广播电视设施、公用电信设施]",
    174: "挪用特定款物",
    175: "[窃取、收买、非法提供]信用卡信息",
    176: "非法组织卖血",
    177: "利用影响力受贿",
    178: "非法捕捞水产品",
    179: "对单位行贿",
    180: "遗弃",
    181: "徇私舞弊[不征、少征]税款",
    182: "提供[侵入、非法控制计算机信息系统][程序、工具]",
    183: "非法进行节育手术",
    184: "危险物品肇事",
    185: "非法[制造、买卖、运输、储存]危险物质",
    186: "非法[制造、销售]非法制造的注册商标标识",
    187: "侵犯著作权",
    188: "倒卖[车票、船票]",
    189: "过失投放危险物质",
    190: "走私废物",
    191: "非法出售发票",
    192: "走私[珍贵动物、珍贵动物制品]",
    193: "[伪造、倒卖]伪造的有价票证",
    194: "招收[公务员、学生]徇私舞弊",
    195: "非法[生产、销售]间谍专用器材",
    196: "倒卖文物",
    197: "虐待被监管人",
    198: "洗钱",
    199: "非法[生产、买卖]警用装备",
    200: "非法获取国家秘密",
    201: "非法[收购、运输、出售][珍贵、濒危野生动物、珍贵、濒危野生动物]制品",
}
if __name__ == "__main__":
    with open("../PRC_legal_dataset/data_val_format.json", "w") as fp:
        train = extract("../PRC_legal_dataset/data_valid.json")
        js.dump(train, fp, ensure_ascii=False, indent=1)


def extract_format(path: str, info: str = "criminal", num_entries: int = None) -> dict:
    with open(path, "r") as fp:
        pair = js.load(fp)
    return pair
