
import pyomo.environ as pyomo
import os
from utli import load_json, save_json
from transformers import AutoTokenizer
import numpy as np
from get_greenlist import get_greenlist,get_greenlist_filename
from utli import Logger,to_string
import datetime
from utli import count_green, count_z_score_bound,print_info
from greenlist_inversion_base import GreenlistInversorBase

class GreenlistInversor(GreenlistInversorBase):

    def set_model(self, sentence_count_num=0.9, key_list=[]):
        
        self.key_index=key_list
        self.log_info(to_string(('key_list', key_list)))

        self.log_info(to_string(('sentence_count_num',sentence_count_num)))

        self.z_score_bound_list=[
            self.get_z_score_bound(len(sen))
            for sen in self.dataset
        ]
        # self.green_num_list=[
        #     count_green(sen, self.true_green)
        #     for sen in self.dataset
        # ]
        self.token_index=list(range(self.vocab_size))
        self.token_weight=[]
        for token_id in self.token_index:
            token=self.tokenizer.decode(token_id)
            # if token in ['that', 'the', 'to', 'of', 'is', 'are','be','on','in','it','an','and','for']:
            #     self.token_weight.append(0)
            #     continue
            if len(token)<=1:
                self.token_weight.append(0)
                continue
            self.token_weight.append(1)
        self.sentence_index=list(range(self.dataset_size))

        self.model = pyomo.ConcreteModel()

        self.model.token_color=pyomo.Var(self.token_index, domain=pyomo.Boolean)
        self.model.whether_sentence=pyomo.Var(self.sentence_index, domain=pyomo.Boolean, initialize=1)

        def sum_green_expr(model, sen_idx):
            return sum([
                model.token_color[token_idx]*self.token_weight[token_idx]
                for token_idx in self.dataset[sen_idx]
            ])
        self.model.sum_green=pyomo.Expression(self.sentence_index, rule=sum_green_expr)

        def whether_sum_green_wm_expr(model, sen_idx):
            return model.sum_green[sen_idx]>=self.z_score_bound_list[sen_idx]*model.whether_sentence[sen_idx]
        self.model.whether_sum_green_wm=pyomo.Constraint(self.wm_sentence_index, rule=whether_sum_green_wm_expr)

        def whether_sum_green_nl_expr(model, sen_idx):
            return model.sum_green[sen_idx]<=(
                self.z_score_bound_list[sen_idx]+(
                    1-model.whether_sentence[sen_idx]
                )*self.sentence_len_list[sen_idx]
            )
        self.model.whether_sum_green_nl=pyomo.Constraint(self.nl_sentence_index, rule=whether_sum_green_nl_expr)

        def sum_whether_sentence_expr(model):
            return sum([
                model.whether_sentence[sen_idx]
                for sen_idx in self.sentence_index
            ])
        self.model.sum_whether_sentence=pyomo.Expression(rule=sum_whether_sentence_expr)
        
        def sum_whether_sentence_con_expr(model):
            return model.sum_whether_sentence>=sentence_count_num*self.dataset_size
        self.model.sum_whether_sentence_con=pyomo.Constraint(rule=sum_whether_sentence_con_expr)

        def sum_green_token_expr(model):
            return sum([
                model.token_color[token_idx]#
                for token_idx in self.token_index
            ])
        self.model.sum_green_token=pyomo.Expression(rule=sum_green_token_expr)
    
    def solve_green_num(self, max_min=False, TimeLimit=None):
        if max_min:
            self.model.sum_green_token_obj=pyomo.Objective(
                expr=self.model.sum_green_token, 
                sense = pyomo.maximize
            )
        else:
            self.model.sum_green_token_obj=pyomo.Objective(
                expr=self.model.sum_green_token, 
                sense = pyomo.minimize
            )
        self.solve(TimeLimit=TimeLimit)
        self.model.sum_green_token_obj.deactivate()
        sum_green_token=pyomo.value(self.model.sum_green_token)
        
        if max_min:
            self.model.sum_green_token_obj_con=pyomo.Constraint(
                expr=self.model.sum_green_token>=sum_green_token*0.95
            )
        else:
            self.model.sum_green_token_obj_con=pyomo.Constraint(
                expr=self.model.sum_green_token<=sum_green_token*1.05
            )
        print_info(to_string(('sum_green_token:', sum_green_token)))
        return

if __name__=='__main__':

    dir_path='saved_data'
    input_type='.json'
    # time.sleep(1440)
    wm_type='o'
    dataset_name='c4_realnewslike'
    model_name_list= ['facebook/opt-1.3b']#, 'facebook/opt-2.7b'
    wm_level_list = ['model', ]#'model','sentence',  'token', 
    dataset_num=str(12500)
    key_num=1
    key_token_list=[123]
    for model_name in model_name_list:
        for wm_level in wm_level_list:
            for data_num in [2000]:#, 4000, 6000, 8000, 10000
                file_name='_'.join((dataset_name,model_name.replace('/','_'),wm_level,wm_type,dataset_num))
                green_inversor=GreenlistInversor(tokenizer_tag=model_name)
                green_inversor.log_info(to_string((wm_level, data_num)))
                green_list_name=get_greenlist_filename(key_token_list)
                green_inversor.load_true_green(os.path.join(dir_path, green_list_name))
                try:
                    wm_data=load_json(os.path.join(dir_path, 'wm_data_'+file_name+'_'+str(data_num)+'.json'))
                    nl_data=load_json(os.path.join(dir_path, 'nl_data_'+file_name+'_'+str(data_num)+'.json'))
                except:
                    wm_data, nl_data=None, None
                green_inversor.set_dataset(
                    dir_path=dir_path, 
                    file_name=file_name, 
                    input_type=input_type,
                    data_num=data_num,
                    gamma=0.25,
                    z_threshold=4,
                    key_num=key_num,
                    wm_data=wm_data, nl_data=nl_data,
                )
                green_inversor.set_model(
                    sentence_count_num=0.95,
                    key_list=key_token_list
                )
                green_inversor.set_solver()
                green_inversor.solve_green_num(max_min=False, TimeLimit=50)
                green_inversor.save_solution(dir_path, wm_level+'_'+str(data_num)+'_'+str(key_num)+'_token_color.json')