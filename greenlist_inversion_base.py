import pyomo.environ as pyomo
import os
from transformers import AutoTokenizer
import numpy as np
from get_greenlist import get_greenlist,get_greenlist_filename
import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utli import get_key, print_info
from utli import Logger,to_string
from utli import count_z_score_bound
from utli import load_json, save_json, remove_repeat
import random

class GreenlistInversorBase:

    def __init__(self, tokenizer_tag="", logger=None,):

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_tag, token=False)
        self.vocab_size=self.tokenizer.vocab_size
        self.true_green = None
        self.device='cuda:0'
        self.token_color_list=[]
        if logger is None:
            self.log=Logger(
                'log/GreenlistInversor'+'-'+str(datetime.datetime.now().date())+'.log',
                level='debug', 
                screen=False
            )
        else:
            self.log=logger
        self.log_info('\n')

    def log_info(self, info):
        print_info(info)
        self.log.logger.info(info)

    def get_z_score_bound(self, len_sentence):
        return count_z_score_bound(len_sentence, self.z_threshold, self.gamma)
    
    def load_true_green(self, file_path):
        self.true_green=load_json(file_path)
        self.true_green_dict={}
        for key in self.true_green:
            self.true_green_dict[key]={}
            for idx in range(self.vocab_size):
                if idx in self.true_green[key]:
                    self.true_green_dict[key][idx]=1
                else:
                    self.true_green_dict[key][idx]=0
        # self.true_green_vector={}
        # for key_id in self.true_green:
        #     self.true_green_vector[key_id]=torch.sum(
        #         F.one_hot(
        #             torch.tensor(self.true_green[key_id]).to(self.device), 
        #             self.vocab_size
        #         ).to(self.device),
        #         dim=0
        #     )
        #     torch.cuda.empty_cache()
        return

    def count_rate(self, sentence_index_list):
        token_rate={}
        for sen_id in sentence_index_list:
            for tmp_token in self.dataset[sen_id]:
                if tmp_token in token_rate:
                    token_rate[tmp_token]+=1
                else:
                    token_rate[tmp_token]=1
        
        token_num=len(token_rate)
        for tmp_token in token_rate:
            token_rate[tmp_token]=token_rate[tmp_token]/token_num
        return token_rate

    def count_green_key(self, token_ids, key):
        green_num=0
        for token in token_ids:
            if self.token_dict[token]==0:
                continue
            green_num+=self.true_green_dict[key][token]
        return green_num

    def count_green(self, token_ids):
        green_num=0
        key='1'
        for tmp_key in self.true_green:
            tmp_num=self.count_green_key(token_ids,tmp_key)
            # tmp_num=self.count_green_vector(token_ids,key)
            # torch.cuda.empty_cache()
            if green_num<tmp_num:
                green_num=tmp_num
                key=tmp_key
        return green_num, key
    
    def count_green_vector(self, token_ids, key_id):
        tmp_vector=F.one_hot(torch.tensor(token_ids), self.vocab_size).to(self.device)
        return torch.sum(torch.mul(
            tmp_vector,
            self.true_green_vector[key_id]#.to(self.device)
        ))

    def set_dataset(self, 
        dir_path, file_name, input_type, 
        wm_data_num=5000, 
        nl_data_num=5000,
        gamma=0.25, 
        z_threshold=4, key_num=1,
        wm_seed=123, nl_seed=123,
        rand_num=0,
        token_prop=(0,1),
        rand_seed=123,
        true_key_list=[],
        query_flag=True,
        min_len=20,
        # wm_data=None, nl_data=None,
    ):
        # if self.true_green is None:
        #     self.true_green={
        #         str(key_id):get_greenlist(
        #             key_token=key_id, vocab_size=self.vocab_size, 
        #             gamma= gamma, device='cuda'
        #         )
        #         for key_id in range(key_num)
        #     }

        self.gamma=gamma
        self.z_threshold=z_threshold
        self.key_num=key_num
        self.true_key_list=true_key_list
        self.query_flag=query_flag
        self.log_info(to_string(('gamma', gamma)))
        self.log_info(to_string(('z_threshold', z_threshold)))
        self.log_info(to_string(('true_key_list', true_key_list)))
        self.log_info(to_string(('token_prop', token_prop)))
        self.log_info(to_string(('wm_data_num，nl_data_num', wm_data_num, nl_data_num)))
        self.log_info(to_string(('query_flag', query_flag)))
        
        input_file_path=os.path.join(dir_path,file_name+input_type)
        data_json=load_json(input_file_path)
        # data_json=data_json[0:data_num]

        if isinstance(wm_data_num, int):
            wm_rand_num=0
        else:
            wm_rand_num=wm_data_num-int(wm_data_num)
            wm_data_num=int(wm_data_num)
        if isinstance(nl_data_num, int):
            nl_rand_num=0
        else:
            nl_rand_num=nl_data_num-int(nl_data_num)
            nl_data_num=int(nl_data_num)#*(1+nl_rand_num))

        with_watermark_list=[]
        without_watermark_list=[]
        # re_sentence_list=[]
        for data in data_json:#
            if (
                ('redecoded_input' in data) 
                and ('output_with_watermark' in data) 
                and ('output_without_watermark' in data)
            ):
                with_watermark_list.append(data['output_with_watermark'])
                without_watermark_list.append(data['output_without_watermark'])
                # re_sentence_list.append(data['redecoded_input'])
        
        tmp_wm_data=self.tokenizer.batch_encode_plus(with_watermark_list).data['input_ids']
        random.seed(wm_seed)
        random.shuffle(tmp_wm_data)
        tmp_wm_data=tmp_wm_data[0:wm_data_num]
        
        tmp_nl_data=self.tokenizer.batch_encode_plus(without_watermark_list).data['input_ids']
        random.seed(nl_seed)
        random.shuffle(tmp_nl_data)
        tmp_nl_data=tmp_nl_data[0:nl_data_num]
        # raw_data=self.tokenizer.batch_encode_plus(re_sentence_list).data['input_ids']

        token_rate={
            idx:0
            for idx in range(self.vocab_size)
        }
        # for sen in tmp_wm_data+tmp_nl_data:
        #     for token in sen:
        #         token_rate[token]+=1
        
        all_token_index=sorted(token_rate.items(), key=lambda x: x[1], reverse=True)
        all_token_index0=all_token_index[0:int(self.vocab_size*0.5)]
        all_token_index1=all_token_index[int(self.vocab_size*0.5):]
        token_start=int(token_prop[0]*self.vocab_size*0.5)
        token_end=int(token_prop[1]*self.vocab_size*0.5)
        random.seed(rand_seed)
        random.shuffle(all_token_index0)
        random.seed(rand_seed)
        random.shuffle(all_token_index1)
        tmp_token_index=all_token_index0[token_start:token_end]+all_token_index1[token_start:token_end]
        
        self.token_index=[t[0] for t in tmp_token_index]
        self.token_dict={
            id:0
            for id in range(self.vocab_size)
        }
        for token_id in self.token_index:
            self.token_dict[token_id]=1
        
        wm_data,nl_data=[],[]
        for idx in tqdm(range(len(tmp_wm_data)), ncols=100, desc='load wm', leave=True):
            token_ids=tmp_wm_data[idx]
            # key=get_key(token_ids, key_num)
            token_ids=remove_repeat(token_ids)
            if len(token_ids)<min_len:
                continue
            
            if query_flag:
                green_num, key=self.count_green(token_ids)
                z_num=self.get_z_score_bound(self.sentence_len(token_ids))
                if green_num<z_num:
                    nl_data.append(token_ids)
                else:
                    wm_data.append(token_ids)
            else:
                wm_data.append(token_ids)
            # if self.count_green(wm_data[-1])[0]<=self.get_z_score_bound(self.sentence_len(wm_data[-1])):
            #     print()
            
        for idx in tqdm(range(len(tmp_nl_data)), ncols=100, desc='load nl', leave=True):
            token_ids=tmp_nl_data[idx]
            token_ids=remove_repeat(token_ids)
            # key=get_key(token_ids, key_num)
            if len(token_ids)<min_len:
                continue
            
            if query_flag:
                green_num, key=self.count_green(token_ids)
                z_num=self.get_z_score_bound(self.sentence_len(token_ids))
                if green_num>=z_num:
                    wm_data.append(token_ids)
                else:
                    nl_data.append(token_ids)
            else:
                nl_data.append(token_ids)
        
        if rand_num>0:
            rand_token=np.random.choice(self.vocab_size, size=rand_num, replace=False)
            rand_sentence=np.ones((rand_num,50))
            rand_sentence=(rand_token*rand_sentence.T).T
            rand_sentence=rand_sentence.astype(np.int32)

            for idx in tqdm(range(rand_num), ncols=100, desc='load random', leave=True):
                rs=rand_sentence[idx]
                green_num, key=self.count_green(rs)
                z_num=self.get_z_score_bound(len(rs))
                if green_num>=z_num:
                    wm_data.append(rs.tolist())
                else:
                    nl_data.append(rs.tolist())

        if wm_rand_num>0 and nl_rand_num>0:
            perb_num=min(int(len(wm_data)*wm_rand_num),int(len(nl_data)*nl_rand_num))
            tmp_rand_wm=wm_data[-perb_num:]
            tmp_rand_nl=nl_data[-perb_num:]
            wm_data=wm_data[0:-perb_num]+tmp_rand_nl
            nl_data=nl_data[0:-perb_num]+tmp_rand_wm


        self.dataset=(
            wm_data
            +nl_data
        )
        self.dataset_size=len(self.dataset)
        self.wm_size=len(wm_data)
        self.nl_size=len(nl_data)

        self.wm_sentence_index=list(range(len(wm_data)))
        self.nl_sentence_index=list(range(len(wm_data), self.dataset_size,1))
        self.sentence_len_list=[len(sen) for sen in self.dataset]
        
        self.label=np.hstack((
            np.ones(len(wm_data)),
            np.zeros(len(nl_data)),
        ))

        print_info('log info: dataset loaded')
    
    def sentence_len(self, sentence):
        len_sen=sum([
            self.token_dict[token]
            for token in sentence
        ])
        return len_sen

    def set_solver(self, TimeLimit=300, solver_type='gurobi', MIPGap= 1e-4):
        self.solver = pyomo.SolverFactory(solver_type, solver_io="python")
        self.solver.options['TimeLimit'] = TimeLimit
        self.solver.options['LogToConsole'] = 1
        self.solver.options['OutputFlag'] = 1
        self.solver.options['Seed'] = 123
        self.solver.options['MIPGap'] = MIPGap
        # self.solver.options['LogFile'] = 'log/gurobi.log'
    
    def solve(self, TimeLimit=None, MIPGap=None, warmstart=True, MIPGapAbs=None):
        if TimeLimit is not None:
            self.solver.options['TimeLimit'] = TimeLimit
        if MIPGap is not None:
            self.solver.options['MIPGap'] = MIPGap
        if MIPGapAbs is not None:
            self.solver.options['MIPGapAbs'] = MIPGapAbs
        # self.solver.options['IntegralityFocus'] = 1
        solution = self.solver.solve(self.model, warmstart=warmstart)
    
    def save_solution(self, dir_path, file_name):
        green_num={
            key_id:0
            for key_id in self.true_key_list
        }#[0]*self.key_num
        true_green_num={
            key_id:0
            for key_id in self.true_key_list
        }#[0]*self.key_num
        if self.solver._solver_model.SolCount==0:
            token_color={
                key_id:{
                    token_id:(
                        self.tokenizer.decode(token_id),
                        0
                    )
                    for token_id in self.token_index
                }
                for key_id in self.key_index
            }
        else:
            tmp_token_color={}
            for idx in range(len(self.key_index)):
                key_id=self.key_index[idx]
                tmp_token_color[key_id]={}
                for token_id in self.token_index:
                    try:
                        tmp_color=pyomo.value(self.model.key_token_color[key_id, token_id])
                    except:
                        tmp_color=0
                    if tmp_color>=0.5:
                        tmp_color=1
                    else:
                        tmp_color=0
                    # if self.key_num>1 and (
                    #     token_id in self.nl_token_rate
                    # ) and (
                    #     token_id in self.wm_token_rate
                    # ) and self.nl_token_rate[token_id]/self.wm_token_rate[token_id]<=1:
                    #     tmp_color=1
                    tmp_token_color[key_id][token_id]=(
                        self.tokenizer.decode(token_id),
                        tmp_color
                    )
            token_color, green_num, true_green_num=self.match_token_color(tmp_token_color)

        self.log_info(to_string(('green_num', green_num)))
        
        for idx in range(len(self.true_key_list)):
            key_id=self.true_key_list[idx]
            if key_id not in green_num:
                continue
            if green_num[key_id]==0:
                self.log_info(to_string(('true_green_num', key_id, true_green_num[key_id], 0)))
            else:
                green_acc=np.round(true_green_num[key_id]/green_num[key_id],4)
                self.log_info(to_string(('true_green_num', key_id, true_green_num[key_id], green_acc)))
        file_path=os.path.join(dir_path, file_name)
        for key_id in self.true_key_list:
            if key_id not in token_color:
                tmp_key=list(token_color.keys())[0]
                token_color[key_id]=token_color[tmp_key]
        self.token_color_list.append(token_color)
        save_json(token_color, file_path=file_path)
        
        # token_color_list={
        #     (key_id, token_id):
        #     token_color[key_id][token_id][1]
        #     for token_id in self.token_index
        #     for key_id in self.key_index
        # }
        # return token_color_list

    def find_key4token_dict(self, tmp_token_color, token_color):
        same_num=[0]*len(self.true_key_list)
        green_num=0
        for token in tmp_token_color:
            green_num+=tmp_token_color[token][1]
            for idx in range(len(self.true_key_list)):
                key=self.true_key_list[idx]
                if key in token_color:
                    continue
                if tmp_token_color[token][1]==1 and self.true_green_dict[str(key)][token]==1:
                    same_num[idx]+=1
        return self.true_key_list[same_num.index(max(same_num))], max(same_num), green_num
    
    def match_token_color(self, tmp_token_color):
        green_num={}
        for key1 in tmp_token_color:
            green_num[key1]=0
            for token in tmp_token_color[key1]:
                if tmp_token_color[key1][token][1]==1:
                    green_num[key1]+=1
        
        tmp_true_green_num={}
        for key1 in tmp_token_color:
            for key2 in self.true_green_dict:
                tmp_true_green_num[(key1, key2)]=0
                for token in tmp_token_color[key1]:
                    if (
                        tmp_token_color[key1][token][1]==1
                    ) and (
                        tmp_token_color[key1][token][1]==self.true_green_dict[key2][token]
                    ):
                        tmp_true_green_num[(key1, key2)]+=1
        
        token_color={}
        true_green_num={}
        for key2 in self.true_green_dict:
            tmp_score=0
            tmp_key=-1
            for key1 in tmp_token_color:
                if key1 in token_color:
                    continue
                if tmp_true_green_num[(key1, key2)]>=tmp_score:
                    tmp_score=tmp_true_green_num[(key1, key2)]
                    tmp_key=key1
            if tmp_key==-1:
                continue
            token_color[tmp_key]=tmp_token_color[tmp_key]
            true_green_num[tmp_key]=tmp_true_green_num[(tmp_key, key2)]
        return token_color, green_num, true_green_num
    
    def compare_token_color(self, dir_path, file_name):
        file_path=os.path.join(dir_path, file_name)
        token_color=load_json(file_path=file_path)
        self.z_score_bound_list=[
            self.get_z_score_bound(self.sentence_len(sen))
            for sen in self.dataset
        ]
        self.green_num_list=[
            self.count_green(sen)[0]
            for sen in self.dataset
        ]
