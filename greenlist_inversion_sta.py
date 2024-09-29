
import pyomo.environ as pyomo
import os
from utli import load_json, save_json
import numpy as np
from get_greenlist import get_greenlist,get_greenlist_filename
from utli import to_string, print_info, check_dir
from greenlist_inversion_base import GreenlistInversorBase
import random
from utli import Logger
import datetime
from model_inversion_config import config_sta_max_num

class GreenlistInversorSta(GreenlistInversorBase):

    def find_green_list(self, key_list, dir_path='', file_name='', gamma_flag=False, max_num=0):

        self.wm_token_rate=self.count_rate(self.wm_sentence_index)
        self.nl_token_rate=self.count_rate(self.nl_sentence_index)

        token_color, green_num,true_green_num={},{},{}
        for key_id in key_list:
            token_color[key_id]={}
            green_num[key_id]=0
            true_green_num[key_id]=0
        # if gamma_flag:
        token_scores=[]
        token_scores_dict={}
        min_nl=min(self.nl_token_rate.values())
        for tmp_token in self.wm_token_rate:
            token_str=self.tokenizer.decode(tmp_token)
            if len(token_str)<=2:
                token_scores.append(0)
                token_scores_dict[tmp_token]=0
                continue
            if token_str in ['that', 'the', 'to', 'of', 'is', 'are','be','on','in','it','an','and','for']:
                token_scores.append(0)
                token_scores_dict[tmp_token]=0
                continue
            if (tmp_token in self.nl_token_rate):
                tmp_score=self.wm_token_rate[tmp_token]/self.nl_token_rate[tmp_token]
                token_scores_dict[tmp_token]=tmp_score
                token_scores.append(tmp_score)
            else:
                tmp_score=self.wm_token_rate[tmp_token]/min_nl
                token_scores.append(tmp_score)
                token_scores_dict[tmp_token]=tmp_score
        token_scores=sorted(token_scores, reverse=True)
        if max_num>0:
            threshold=max(token_scores[max_num],1)
        elif len(token_scores)>int(self.vocab_size*self.gamma) and gamma_flag:
            threshold=max(token_scores[int(self.vocab_size*self.gamma)],1)
        else:
            threshold=1#token_scores[-1]
        
        for key_id in key_list:
            for tmp_token in self.wm_token_rate:
                token_str=self.tokenizer.decode(tmp_token)
                if token_scores_dict[tmp_token]<threshold :
                    token_color[key_id][tmp_token]=(token_str,0)
                else:
                    token_color[key_id][tmp_token]=(token_str,1)
                    green_num[key_id]+=1

                    if green_num[key_id]>=max_num and max_num>0:
                        break

                    if self.true_green_dict[str(key_id)][tmp_token]==1:
                        true_green_num[key_id]+=1

        # else:    
        #     for tmp_token in self.wm_token_rate:
        #         token_str=self.tokenizer.decode(tmp_token)
        #         if len(token_str)<=2:
        #             continue
        #         if token_str in ['that', 'the', 'to', 'of', 'is', 'are','be','on','in','it','an','and','for']:
        #             continue
        #         # if tmp_token not in self.nl_token_rate:
        #         #     continue
        #         if (tmp_token in self.nl_token_rate) and (
        #             self.wm_token_rate[tmp_token]<=self.nl_token_rate[tmp_token]
        #         ):
        #             # if self.wm_token_rate[tmp_token]>self.nl_token_rate[tmp_token]:
        #             #     token_color[key_id][tmp_token]=(self.tokenizer.decode(tmp_token),1)
        #             # else:
        #             token_color[key_id][tmp_token]=(token_str,0)
        #         else:
        #             token_color[key_id][tmp_token]=(token_str,1)
        #             green_num[key_id]+=1
        #             if self.true_green_dict[str(key_id)][tmp_token]==1:
        #                 true_green_num[key_id]+=1
        
        self.log_info(to_string(('green_num', green_num)))
        for idx in range(len(self.true_key_list)):
            tmp_key=self.true_key_list[idx]
            if tmp_key not in green_num:
                continue
            if green_num[tmp_key]==0:
                self.log_info(to_string(('true_green_num', tmp_key, true_green_num[tmp_key], 0)))
            else:
                green_acc=np.round(true_green_num[tmp_key]/green_num[tmp_key],4)
                self.log_info(to_string(('true_green_num', tmp_key, true_green_num[tmp_key], green_acc)))
        if len(file_name)>0:
            for tmp_key in key_list[1:]:
                token_color[tmp_key]=token_color[key_id]
            file_path=os.path.join(dir_path, file_name)
            save_json(token_color, file_path=file_path)

if __name__=='__main__':
    
    dir_path='saved_data'
    save_path='saved_res'
    check_dir(dir_path)
    check_dir(save_path)
    input_type='.json'
    # time.sleep(1440)
    wm_type='o'
    dataset_name='c4_realnewslike'
    model_name_list= ['facebook/opt-1.3b']#, 'facebook/opt-1.3b','../model/llama-2-7b'
    wm_level = 'model'#'sentence_fi'#
    dataset_num=str(30000)
    key_token_list=[123]
    key_num=1
    wm_seed, nl_seed=123, 456
    
    rand_num=0
    expect_green_size=1
    # gamma, delta=0.25, 2
    
    z_threshold=4
    query_flag=True
    # sentence_up_num, sentence_down_num=0.99, 0.98
    # wm_bound, nl_bound=0.95, 0.95
    gamma_flag=True
    max_num_flag=False
    perb_rate=0

    check_dir('log')
    logger=Logger(
        'log/GreenlistInversorSta-'+'-'.join([
            str(query_flag), 
        ])+'-'+str(datetime.datetime.now().date())+'.log',
        level='debug', 
        screen=False
    )

    model_name = model_name_list[0]
    for (gamma, delta, max_num) in [
        (0.25, 2, 0),
        (0.25, 4,0),
        (0.5, 2, 0),
        (0.5, 4, 0),##
    ]:#
        for (wm_data_num, nl_data_num, perb_rate) in [ 
            # (100,100),#(500,500)
            (2000,2000,0),
            (5000,5000,0),
            # # (5000,5000,0.3),
            (10000,10000,0),
            # (10000,10000,0.7),
            (20000,20000,0),
        #    (2000,2000),
        # (3000,3000,0),
        # (6000,6000,0),
        # (20000,20000),# 
        ]:
            print()

            # if max_num_flag:
            #     max_num=2000#config_sta_max_num[model_name][(gamma, delta)][wm_data_num]
            # else:
            #     max_num=0
            
            green_inversor = GreenlistInversorSta(tokenizer_tag=model_name, logger=logger)
            
            green_inversor.log_info(to_string((wm_level, rand_num, gamma, delta)))
            # green_inversor.log_info(to_string(('native_flag', native_flag)))
            
            # green_list_name= get_greenlist_filename(key_token_list, gamma, model_name)
            # green_inversor.load_true_green(os.path.join(dir_path, green_list_name))
            if wm_level=='model':
                green_list_name= get_greenlist_filename(key_token_list, gamma, model_name)
            else:
                green_list_name= get_greenlist_filename(key_token_list, gamma+delta, model_name+'_'+wm_level)#
            green_inversor.load_true_green(os.path.join(dir_path, green_list_name))

            file_name='_'.join((
                dataset_name, 
                model_name.replace('/','_'),
                wm_level,wm_type,
                dataset_num,
                str(key_num),
                str(gamma), str(delta)
            ))

            if perb_rate>0:
                wm_data_num+=perb_rate
                nl_data_num+=perb_rate
            
            green_inversor.set_dataset(
                dir_path=dir_path, file_name=file_name, input_type=input_type,
                wm_data_num=wm_data_num, nl_data_num=nl_data_num,
                gamma=gamma,
                z_threshold=z_threshold,
                key_num=key_num, rand_num=rand_num,
                wm_seed=wm_seed, nl_seed=nl_seed,
                token_prop=(0,1),
                true_key_list=key_token_list,
                query_flag=query_flag
            )

            green_inversor.log_info(to_string(('gamma_flag:', gamma_flag)))
            # green_inversor.find_green_list(key_id=key_token_list[0],  dir_path=save_path, gamma_flag=gamma_flag,)
            # green_inversor.find_green_list(key_id=key_token_list[1], gamma_flag=gamma_flag,max_num=max_num)
            # green_inversor.find_green_list(key_id=key_token_list[2], gamma_flag=gamma_flag,max_num=max_num)
            # if max_num_flag:
            #     expect_green_size+=max_num
            green_inversor.find_green_list(
                key_list=key_token_list, 
                dir_path=save_path, 
                gamma_flag=gamma_flag,
                max_num=max_num,
                file_name='_'.join([
                    'sta',wm_level, wm_type, dataset_name, model_name.replace('/','_'), 
                    dataset_num,
                    str(key_token_list), str(key_num), 
                    str(wm_seed), str(nl_seed), str(rand_num), 
                    str(wm_data_num), str(nl_data_num),
                    str(gamma), str(delta),
                    str(expect_green_size),
                    str(z_threshold), 
                    str(query_flag), 
                    str(gamma_flag),
                    'token_color.json'
                ])
            )