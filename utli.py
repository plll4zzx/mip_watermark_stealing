import json

import logging
from logging import handlers
import numpy as np
import torch
from datetime import datetime
import random
import copy
import os

def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def remove_repeat(token_ids, repeat_len=10):
    sen_dict={}
    len_ids=len(token_ids)
    for idx in range(len_ids):
        tmp_array=np.array(token_ids[idx:min(idx+repeat_len, len_ids)])
        tmp_idx=np.arange(0, min(idx+repeat_len, len_ids)-idx)
        tmp_sum_past=np.sum(tmp_array*tmp_idx)
        # tmp_sum_pre=sum(token_ids[max(0, idx-repeat_len):idx])
        if tmp_sum_past not in sen_dict:
             sen_dict[tmp_sum_past]=1
        else:
            return token_ids[0:idx]
    return token_ids

def count_green(token_ids, green_list):
    green_num=0
    if isinstance(green_list, list):
        for token in token_ids:
            if token in green_list:
                green_num+=1
    elif isinstance(green_list, dict):
        for token in token_ids:
            green_num+=green_list[token]
    return green_num

def count_z_score_bound(len_sentence, z_threshold, gamma):
    z_score_bound=np.sqrt(
        len_sentence*gamma*(1-gamma)
    )*z_threshold + len_sentence*gamma
    return np.floor(z_score_bound)

def get_key(sentence, key_num):
    return (sentence[0]%key_num)+1
    
def print_info(info):
    split_line_len=40
    if len(info)>32:
        split_line_len=int(len(info)+2*5)
    print('*'*split_line_len)
    left_len=(split_line_len-len(info)-2)/2
    left_len=int(left_len)
    right_len=split_line_len-left_len-len(info)-2
    print('*'*left_len+' '+info+' '+'*'*right_len)
    print('*'*split_line_len)

def check_num(x_str):
    num_str_list=[str(idx) for idx in range(10)]
    if x_str in num_str_list:
        return True
    return False

def split_sentence(sentence):
        sentence=sentence.replace('\n', ' ')
        last_idx=0
        sub_sentence_list=[]
        for idx in range(len(sentence)):
            char_=sentence[idx]
            if idx>0 and idx<len(sentence)-1 and char_=='.' and check_num(sentence[idx-1]) and check_num(sentence[idx+1]):
                continue
            if char_ in ['.', '?', ';', '!', '•']:#,' – ',',"'
                sub_sentence=sentence[last_idx:idx+1]
                if sub_sentence[0]==' ':
                    sub_sentence=sub_sentence[1:]
                if sub_sentence[-1]==' ':
                    sub_sentence=sub_sentence[:-1]
                if len(sub_sentence)<=20:
                    continue
                last_idx=idx+1
                sub_sentence_list.append(sub_sentence)
        if last_idx<len(sentence)-1:
            sub_sentence_list.append(sentence[last_idx:])
        return sub_sentence_list

def reorder_sentence_list(sub_sentence_list, re_order=0.1):
    k=int(len(sub_sentence_list)*re_order)
    head_list=sub_sentence_list[0:3]
    random.shuffle(head_list)
    sub_sentence_list[0:3]=head_list
    reorder_idx=list(set(random.choices(sub_sentence_list,k=k)))
    reorder_idx=sorted(reorder_idx)
    random.shuffle(reorder_idx)
    sub_sentence_list_new=copy.deepcopy(sub_sentence_list)
    idy=0
    for idx in range(len(sub_sentence_list)):
        if idx in reorder_idx:
            sub_sentence_list_new[idx]=sub_sentence_list[reorder_idx[idy]]
            idy+=1
    sub_sentence_list_new
    return sub_sentence_list_new

def get_timestamp():
    now = datetime.now()
    timestamp = now.timestamp()
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_time = dt_object.strftime("%Y_%m_%d_%H_%M_%S_%f")[:-4]
    return formatted_time

def project(x, lp, epsilon):
    if isinstance(lp, int):
        x_norm = torch.norm(x.reshape((x.shape[0], -1)), lp,dim=-1)
        x_norm = torch.maximum(x_norm, torch.ones_like(x_norm)*1e-12)
        factors=epsilon/x_norm
        factors=torch.minimum(factors, torch.ones_like(factors))
        x=x*factors
    elif isinstance(lp, str) and lp=='inf':
        x=torch.clip(x, min=-epsilon, max=epsilon)
    else:
        x=torch.zeros_like(x)
    return x

def project_lp(x, lp, epsilon):
    return x/(torch.norm(x, lp)+torch.ones_like(x)*1e-12)*epsilon

def uniform_np(x, lp):
    x_norm=np.linalg.norm(x, ord=lp, axis=-1, keepdims=True)+1e-12
    return x/x_norm

def uniform_torch(x, lp):
    x_norm=torch.norm(x, p=lp, dim=-1, keepdim=True)+1e-12
    return x/x_norm

def cosine_similarity(a, b):
    return a.dot(b)/((np.linalg.norm(a,ord=2)*np.linalg.norm(b,ord=2))+1e-12)

def hamming_distance(a,b):
    return np.sum(np.logical_xor(a,b))

def load_json(file_path):
    with open(file_path, 'r') as file:
        dict_list=json.load(file)
    return dict_list

def save_json(data, file_path):
    json_data = json.dumps(data)
    with open(file_path, 'w') as file:
        file.write(json_data)

def save_model(model, tokenizer, model_path):
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

def to_string(inputs):
    output_str=''
    for input in inputs:
        if isinstance(input, str):
            output_str+=input
        else:
            output_str+=str(input)
        output_str+=' '
    return output_str
class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', screen=True):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        if screen:
            self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

if __name__=='__main__':
    a=list(range(10))
    for r in np.arange(0.1,1,0.1):
        b=reorder_sentence_list(a,r)
        print(r, b)