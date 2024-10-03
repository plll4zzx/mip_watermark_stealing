
import pyomo.environ as pyomo
import os
from utli import load_json, save_json
import numpy as np
from get_greenlist import get_greenlist,get_greenlist_filename
from utli import to_string, print_info
from greenlist_inversion_base import GreenlistInversorBase
import random

class GreenlistInversorPlus(GreenlistInversorBase):

    def set_model(
        self, 
        sentence_up_num=1, 
        sentence_down_num=1,
        perb_rate=0,
        expect_green_size=1,
        key_list=[],
        token_len=1, gamma_flag=True, oracle_flag=False,
        reals_flag=False,
        token_color_ini=0,
        min_max_green_bound=-1,
        min_green_num=2000,
    ):
        
        self.log_info(to_string(('sentence_up_num, sentence_down_num',sentence_up_num, sentence_down_num)))
        self.log_info(to_string(('expect_green_size',expect_green_size)))
        self.log_info(to_string(('key_list',key_list)))
        self.log_info(to_string(('token_len',token_len)))
        self.log_info(to_string(('gamma_flag, oracle_flag',gamma_flag, oracle_flag)))
        self.log_info(to_string(('reals_flag',reals_flag)))

        self.key_index=key_list
        self.gamma_flag=gamma_flag

        self.wm_token_rate=self.count_rate(self.wm_sentence_index)
        self.nl_token_rate=self.count_rate(self.nl_sentence_index)

        if isinstance(token_color_ini, dict):
            tmp_token_color_ini={
                (key_id, token_id): token_color_ini[str(key_id)][token_id]
                for key_id in self.key_index
                for token_id in self.token_index
            }
            token_color_ini=tmp_token_color_ini
            self.log_info(to_string(('ground truth','True')))
        else:
            random.seed(123)
            randm_green=list(range(self.vocab_size))
            random.shuffle(randm_green)
            step=0.2
            step1=0.3
            tmp_token_color_ini={}
            for idx, key_id in enumerate(self.key_index):
                for token_id in self.token_index:
                    tmp_start=int(idx*step*self.vocab_size)
                    tmp_end=int(tmp_start+step1*self.vocab_size)
                    if tmp_start<self.vocab_size and tmp_end>self.vocab_size:
                        tmp_list=randm_green[tmp_start:]+randm_green[0:tmp_end%self.vocab_size]
                    elif tmp_start>self.vocab_size and tmp_end>self.vocab_size:
                        tmp_end=tmp_end%self.vocab_size
                        tmp_start=tmp_start%self.vocab_size
                        tmp_list=randm_green[tmp_start:tmp_end]
                    else:
                        tmp_list=randm_green[tmp_start:tmp_end]
                    if token_id in tmp_list:
                        tmp_token_color_ini[(key_id, token_id)]=1
                    else:
                        tmp_token_color_ini[(key_id, token_id)]=0
            token_color_ini=tmp_token_color_ini
            self.log_info(to_string(('ground truth','False')))

        self.z_score_bound_list=[
            self.get_z_score_bound(self.sentence_len(sen))
            for sen in self.dataset
        ]
        self.green_num_list=[
            self.count_green(sen)[0]
            for sen in self.dataset
        ]

        self.key_token_index=[
            (key_id, token_id)
            for key_id in self.key_index
            for token_id in self.token_index
        ]
        self.sentence_index=list(range(self.dataset_size))
        self.sentence_key_index=[
            (sentence_id, key_id)
            for sentence_id in self.sentence_index
            for key_id in self.key_index
        ]
        self.wm_sentence_key_index=[
            (sentence_id, key_id)
            for sentence_id in self.wm_sentence_index
            for key_id in self.key_index
        ]
        self.nl_sentence_key_index=[
            (sentence_id, key_id)
            for sentence_id in self.nl_sentence_index
            for key_id in self.key_index
        ]

        self.token_weight={}
        for token_id in self.token_index:
            token=self.tokenizer.decode(token_id)
            token=token.replace(' ','')
            if token in ['that', 'the', 'to', 'of', 'is', 'are','be','on','in','it','an','and','for','</s>']:
                self.token_weight[token_id]=2
                continue
            if len(token)<=1:
                self.token_weight[token_id]=5
                continue
            if len(token)<=2:
                self.token_weight[token_id]=4
                continue
            
            self.token_weight[token_id]=1

            if (
                token_id in self.nl_token_rate
            ) and (
                token_id in self.wm_token_rate
            ) and perb_rate==0 and len(key_list)==1:#
                rate=self.nl_token_rate[token_id]/self.wm_token_rate[token_id]
                self.token_weight[token_id]=rate

        self.model = pyomo.ConcreteModel()

        if reals_flag:
            self.model.key_token_color  = pyomo.Var(self.key_token_index, domain=pyomo.Reals, bounds=(0,1), initialize=token_color_ini)
        else:
            self.model.key_token_color  = pyomo.Var(self.key_token_index, domain=pyomo.Boolean, initialize=token_color_ini)
        self.model.sentence_key     = pyomo.Var(self.sentence_key_index, domain=pyomo.Boolean)
        self.model.whether_sentence = pyomo.Var(self.sentence_index,     domain=pyomo.Boolean)
        self.model.whether_key      = pyomo.Var(self.key_index,          domain=pyomo.Boolean)
        self.model.sen_green_bound  = pyomo.Var(self.sentence_index,     domain=pyomo.NonNegativeReals)

        def sum_whether_key_expr(model):
            return sum([
                model.whether_key[key_id]
                for key_id in self.key_index
            ])
        self.model.sum_whether_key=pyomo.Expression(rule=sum_whether_key_expr)

        def sum_sentence_key_expr(model, sentence_id):
            return sum([
                model.sentence_key[sentence_id, key_id]
                for key_id in self.key_index
            ])
        self.model.sum_sentence_key=pyomo.Expression(self.sentence_index, rule=sum_sentence_key_expr)
        def sum_sentence_key_con_expr(model, sentence_id):
            return model.sum_sentence_key[sentence_id]==1
        self.model.sum_sentence_key_con=pyomo.Constraint(self.sentence_index, rule=sum_sentence_key_con_expr)
        
        def sum_green_key_expr(model, sentence_id, key_id):
            return sum([
                model.key_token_color[key_id, token_id]
                for token_id in self.dataset[sentence_id]
                if self.token_dict[token_id]==1
            ])
        self.model.sum_green_key=pyomo.Expression(self.sentence_key_index, rule=sum_green_key_expr)

        def whether_sum_green_wm_expr(model, sentence_id, key_id):
            if oracle_flag:
                return model.sum_green_key[sentence_id, key_id]>=(
                    model.sen_green_bound[sentence_id]
                    +(
                        model.sentence_key[sentence_id, key_id]-1
                    )*self.sentence_len_list[sentence_id]*2
                )
            return model.sum_green_key[sentence_id, key_id]>=(
                model.sen_green_bound[sentence_id]+(
                    model.sentence_key[sentence_id, key_id]-1
                    + model.whether_sentence[sentence_id]-1
                )*self.sentence_len_list[sentence_id]*2
            )
        self.model.whether_sum_green_key_wm=pyomo.Constraint(
            self.wm_sentence_key_index, 
            rule=whether_sum_green_wm_expr
        )

        if len(key_list)>1:
            key_group=[
                (self.key_index[k1],self.key_index[k2%len(self.key_index)])
                for k1 in range(len(self.key_index))
                for k2 in range(k1+1,k1+len(self.key_index))
            ]
            
            def sum_token_key_expr(model, key_id):
                return sum([
                    model.key_token_color[key_id, token_id]
                    for token_id in self.token_index
                ])
            self.model.sum_token_key=pyomo.Expression(self.key_index, rule=sum_token_key_expr)
            def sum_token_key_con_expr(model, key_id1, key_id2):
                return (model.sum_token_key[key_id1]-model.sum_token_key[key_id2])<=100
            self.model.sum_token_key_con=pyomo.Constraint(key_group, rule=sum_token_key_con_expr)
            def sum_token_key_con2_expr(model, key_id):
                return model.sum_token_key[key_id]>=min_green_num
            self.model.sum_token_key_con2=pyomo.Constraint(self.key_index, rule=sum_token_key_con2_expr)
        
        if self.gamma_flag:
            def sen_green_bound_wm_con_expr(model, sentence_id):
                if oracle_flag:
                    return model.sen_green_bound[sentence_id]==self.green_num_list[sentence_id]
                return model.sen_green_bound[sentence_id]>=self.z_score_bound_list[sentence_id]*model.whether_sentence[sentence_id]
            self.model.sen_green_bound_wm_con=pyomo.Constraint(
                self.wm_sentence_index, 
                rule=sen_green_bound_wm_con_expr
            )

        def sum_sen_green_bound_wm_expr(model):
            return sum([
                model.sen_green_bound[sentence_id]
                for sentence_id in self.wm_sentence_index
            ])
        self.model.sum_sen_green_bound_wm=pyomo.Expression(rule=sum_sen_green_bound_wm_expr)

        def whether_sum_green_nl_expr(model, sentence_id, key_id):
            if oracle_flag:
                return model.sum_green_key[sentence_id, key_id]<=model.sen_green_bound[sentence_id]
            return model.sum_green_key[sentence_id, key_id]<=(
                model.sen_green_bound[sentence_id]
                +(
                    1-model.whether_sentence[sentence_id]
                )*self.sentence_len_list[sentence_id]*2
            )
        self.model.whether_sum_green_nl=pyomo.Constraint(
            self.nl_sentence_key_index, 
            rule=whether_sum_green_nl_expr
        )
        
        if self.gamma_flag:
            def sen_green_bound_nl_con_expr(model, sentence_id):
                if oracle_flag:
                    return model.sen_green_bound[sentence_id]==self.green_num_list[sentence_id]
                return model.sen_green_bound[sentence_id]<=(
                    self.z_score_bound_list[sentence_id]+
                    (1-model.whether_sentence[sentence_id])*self.sentence_len_list[sentence_id]*2
                )
            self.model.sen_green_bound_nl_con=pyomo.Constraint(
                self.nl_sentence_index, 
                rule=sen_green_bound_nl_con_expr
            )
        if gamma_flag:
            self.model.abs_sen_green_bound=pyomo.Var(self.nl_sentence_index, domain=pyomo.NonNegativeReals)
            def abs_sen_green_bound_con1_expr(model, sentence_id):
                return model.abs_sen_green_bound[sentence_id]>=(
                    model.sen_green_bound[sentence_id]-self.sentence_len_list[sentence_id]*self.gamma
                )
            self.model.abs_sen_green_bound_con1=pyomo.Constraint(self.nl_sentence_index, rule=abs_sen_green_bound_con1_expr)
            def abs_sen_green_bound_con2_expr(model, sentence_id):
                return model.abs_sen_green_bound[sentence_id]>=(
                    -model.sen_green_bound[sentence_id]+self.sentence_len_list[sentence_id]*self.gamma
                )
            self.model.abs_sen_green_bound_con2=pyomo.Constraint(self.nl_sentence_index, rule=abs_sen_green_bound_con2_expr)
        
        def sum_sen_green_bound_nl_expr(model):
            return sum([
                model.sen_green_bound[sentence_id]
                for sentence_id in self.nl_sentence_index
            ])
        self.model.sum_sen_green_bound_nl=pyomo.Expression(rule=sum_sen_green_bound_nl_expr)

        if gamma_flag:
            def sum_abs_sen_green_bound_nl_expr(model):
                return sum([
                    model.abs_sen_green_bound[sentence_id]
                    for sentence_id in self.nl_sentence_index
                ])
            self.model.sum_abs_sen_green_bound_nl=pyomo.Expression(rule=sum_abs_sen_green_bound_nl_expr)
        
        def whether_bound_con_expr(model, sentence_id):
            return model.whether_sentence[sentence_id]*self.sentence_len_list[sentence_id]>=model.sen_green_bound[sentence_id]
        self.model.whether_bound_con=pyomo.Constraint(
            self.sentence_index,
            rule=whether_bound_con_expr
        )

        if not oracle_flag and perb_rate>0:
            self.model.min_wm_sen_green_bound=pyomo.Var(domain=pyomo.NonNegativeReals)
            self.model.max_nl_sen_green_bound=pyomo.Var(domain=pyomo.NonNegativeReals)

            def min_wm_sen_green_bound_con_expr(model, sentence_id):
                return model.min_wm_sen_green_bound<=(
                    model.sen_green_bound[sentence_id]/self.sentence_len_list[sentence_id]
                    +1-model.whether_sentence[sentence_id]
                )
            self.model.min_wm_sen_green_bound_con=pyomo.Constraint(
                self.wm_sentence_index, 
                rule=min_wm_sen_green_bound_con_expr
            )

            def max_nl_sen_green_bound_con_expr(model, sentence_id):
                return model.max_nl_sen_green_bound>=(
                    model.sen_green_bound[sentence_id]/self.sentence_len_list[sentence_id]
                    +model.whether_sentence[sentence_id]-1
                )
            self.model.max_nl_sen_green_bound_con=pyomo.Constraint(
                self.nl_sentence_index, 
                rule=max_nl_sen_green_bound_con_expr
            )
            
            if perb_rate>0:
                self.model.min_max_sen_green_bound_con2=pyomo.Constraint(
                    expr=(self.model.min_wm_sen_green_bound-self.model.max_nl_sen_green_bound)>=min_max_green_bound
                )

        def sum_whether_sentence_expr(model):
            return sum([
                model.whether_sentence[sentence_id]
                for sentence_id in self.nl_sentence_index
            ])+sum([
                model.whether_sentence[sentence_id]
                for sentence_id in self.wm_sentence_index
            ])
        self.model.sum_whether_sentence=pyomo.Expression(rule=sum_whether_sentence_expr)

        def sum_whether_wm_sentence_expr(model):
            return sum([
                model.whether_sentence[sentence_id]
                for sentence_id in self.wm_sentence_index
            ])
        self.model.sum_whether_wm_sentence=pyomo.Expression(rule=sum_whether_wm_sentence_expr)
        
        def sum_whether_wm_sentence_up_con_expr(model):
            return model.sum_whether_wm_sentence<=sentence_up_num*self.wm_size
        self.model.sum_whether_wm_sentence_up_con=pyomo.Constraint(rule=sum_whether_wm_sentence_up_con_expr)
        
        def sum_whether_wm_sentence_down_con_expr(model):
            return model.sum_whether_wm_sentence>=sentence_down_num*self.wm_size
        self.model.sum_whether_wm_sentence_down_con=pyomo.Constraint(rule=sum_whether_wm_sentence_down_con_expr)

        def sum_whether_nl_sentence_expr(model):
            return sum([
                model.whether_sentence[sentence_id]
                for sentence_id in self.nl_sentence_index
            ])
        self.model.sum_whether_nl_sentence=pyomo.Expression(rule=sum_whether_nl_sentence_expr)
        
        def sum_whether_nl_sentence_up_con_expr(model):
            if perb_rate>0:
                return model.sum_whether_nl_sentence<=sentence_up_num*self.nl_size
            else:
                return model.sum_whether_nl_sentence<=max((sentence_up_num+perb_rate),0.9)*self.nl_size
        self.model.sum_whether_nl_sentence_up_con=pyomo.Constraint(rule=sum_whether_nl_sentence_up_con_expr)
        
        def sum_whether_nl_sentence_down_con_expr(model):
            if perb_rate>0:
                return model.sum_whether_nl_sentence>=sentence_down_num*self.nl_size
            else:
                return model.sum_whether_nl_sentence>=max((sentence_down_num+perb_rate),0.8)*self.nl_size
        self.model.sum_whether_nl_sentence_down_con=pyomo.Constraint(rule=sum_whether_nl_sentence_down_con_expr)

        def sum_green_token_expr(model):
            return sum([
                model.key_token_color[key_id, token_id]*self.token_weight[token_id]#
                for key_id in self.key_index
                for token_id in self.token_index
                if self.token_dict[token_id]==1
            ])
        self.model.sum_green_token=pyomo.Expression(rule=sum_green_token_expr)
        
        def sum_green_token_each_expr(model, key_id):
            return sum([
                model.key_token_color[key_id, token_id]
                for token_id in self.token_index
                if self.token_dict[token_id]==1
            ])
        self.model.sum_green_token_each=pyomo.Expression(self.key_index,rule=sum_green_token_each_expr)
        
        def sum_green_token_each_con_expr(model, key_id):
            if gamma_flag:
                return model.sum_green_token_each[key_id]<=self.vocab_size*self.gamma*expect_green_size
            else:
                return model.sum_green_token_each[key_id]<=self.vocab_size*expect_green_size
        self.model.sum_green_token_each_con=pyomo.Constraint(self.key_index,rule=sum_green_token_each_con_expr)
        
        print_info('log info: model init')
    
    def solve_wm_nl_sum_green(
        self,
        TimeLimit=None, MIPGap=None, MIPGapAbs=None,
        wm_bound=0.95, nl_bound=1.05, 
        lock_sentence=False, 
        bound_flag=True,
    ):
        self.log_info(to_string(('TimeLimit, MIPGap: ', TimeLimit, MIPGap)))
        self.log_info(to_string(('wm_bound, nl_bound: ', wm_bound, nl_bound)))
        self.log_info(to_string(('lock_sentence: ', lock_sentence)))
        if self.gamma_flag:
            self.model.wm_nl_sum_green_obj=pyomo.Objective(
                expr  = (
                    self.model.sum_sen_green_bound_wm/self.wm_size
                    -self.model.sum_abs_sen_green_bound_nl/self.nl_size
                ), 
                sense = pyomo.maximize
            )
        else:
            self.model.wm_nl_sum_green_obj=pyomo.Objective(
                expr  = (
                    self.model.sum_sen_green_bound_wm/self.wm_size
                    -self.model.sum_sen_green_bound_nl/self.nl_size
                ), 
                sense = pyomo.maximize
            )
        self.solve(TimeLimit=TimeLimit, MIPGap=MIPGap, MIPGapAbs=MIPGapAbs)
        self.model.wm_nl_sum_green_obj.deactivate()

        try:
            sum_sen_green_bound_wm=pyomo.value(self.model.sum_sen_green_bound_wm)
            sum_sen_green_bound_nl=pyomo.value(self.model.sum_sen_green_bound_nl)
            sum_whether_wm_sentence=pyomo.value(self.model.sum_whether_wm_sentence)
            sum_whether_nl_sentence=pyomo.value(self.model.sum_whether_nl_sentence)
        except:
            return False

        mean_g_o_i_wm=np.mean([self.green_num_list[idx] for idx in self.wm_sentence_index])
        mean_g_o_i_nl=np.mean([self.green_num_list[idx] for idx in self.nl_sentence_index])

        mean_g_i_wm=np.mean([self.z_score_bound_list[idx] for idx in self.wm_sentence_index])
        mean_g_i_nl=np.mean([self.z_score_bound_list[idx] for idx in self.nl_sentence_index])

        mean_bi_wm=np.mean([
            pyomo.value(self.model.sen_green_bound[idx]) 
            for idx in self.wm_sentence_index
            if pyomo.value(self.model.sen_green_bound[idx])>0
        ])
        mean_bi_nl=np.mean([
            pyomo.value(self.model.sen_green_bound[idx]) 
            for idx in self.nl_sentence_index
            if pyomo.value(self.model.sen_green_bound[idx])>0
        ])
        
        self.log_info(to_string(('mean_g_o_i_wm: ', np.round(mean_g_o_i_wm,2))))
        self.log_info(to_string(('mean_g_o_i_nl: ', np.round(mean_g_o_i_nl,2))))
        self.log_info(to_string(('mean_g_i_wm: ', np.round(mean_g_i_wm,2))))
        self.log_info(to_string(('mean_g_i_nl: ', np.round(mean_g_i_nl,2))))
        self.log_info(to_string(('mean_bi_wm: ', np.round(mean_bi_wm,2))))
        self.log_info(to_string(('mean_bi_nl: ', np.round(mean_bi_nl,2))))

        self.log_info(to_string(('sum_whether_nl_sentence: ', sum_whether_nl_sentence)))
        self.log_info(to_string(('sum_whether_wm_sentence: ', sum_whether_wm_sentence)))
        
        self.model.new_sum_sen_green_bound_wm_con=pyomo.Constraint(
            expr=self.model.sum_sen_green_bound_wm>=sum_sen_green_bound_wm*wm_bound
        )

        self.model.new_sum_sen_green_bound_nl_con=pyomo.Constraint(
            expr=self.model.sum_sen_green_bound_nl<=sum_sen_green_bound_nl*nl_bound
        )

        if lock_sentence:
            whether_sentence={
                sentence_id:pyomo.value(self.model.whether_sentence[sentence_id])
                for sentence_id in self.sentence_index
            }
            
            self.model.sum_whether_wm_sentence_obj_con=pyomo.Constraint(
                expr=self.model.sum_whether_wm_sentence>=sum_whether_wm_sentence 
            )
            self.model.sum_whether_nl_sentence_obj_con=pyomo.Constraint(
                expr=self.model.sum_whether_nl_sentence>=sum_whether_nl_sentence 
            )
        if len(self.key_index)>1:
            sentence_key={
                (sen_id, key_id): pyomo.value(self.model.sentence_key[(sen_id, key_id)])
                for (sen_id, key_id) in self.wm_sentence_key_index
            }

            true_key_rate=0
            sen_num=0
            for sen_id in self.wm_sentence_index:
                if whether_sentence[sen_id]==0:
                    continue
                sen_num+=1
                _, tmp_true_key=self.count_green(self.dataset[sen_id])
                tmp_list=[sentence_key[(sen_id, key_id)] for key_id in self.key_index]
                tmp_idx=np.argmax(tmp_list)
                if list(self.true_green.keys())[tmp_idx]==tmp_true_key:
                    true_key_rate+=1
            self.log_info(to_string(('true_key_rate: ', np.round(true_key_rate/sen_num,4))))
        return True

    def solve_min_max_bound(self, TimeLimit=None, MIPGap=None):
        self.model.min_max_bound_obj=pyomo.Objective(
            expr  = self.model.min_wm_sen_green_bound-self.model.max_nl_sen_green_bound, 
            sense = pyomo.maximize
        )
        self.solve(TimeLimit=TimeLimit, MIPGap=MIPGap)
        self.model.min_max_bound_obj.deactivate()
        
        min_wm_sen_green_bound=np.round(pyomo.value(self.model.min_wm_sen_green_bound),2)
        max_nl_sen_green_bound=np.round(pyomo.value(self.model.max_nl_sen_green_bound),2)

        self.log_info(to_string(('min_wm_sen_green_bound:', min_wm_sen_green_bound)))
        self.log_info(to_string(('max_nl_sen_green_bound:', max_nl_sen_green_bound)))

        self.model.min_max_bound_obj_con1=pyomo.Constraint(
            expr=self.model.min_wm_sen_green_bound>=min_wm_sen_green_bound
        )
        self.model.min_max_bound_obj_con2=pyomo.Constraint(
            expr=self.model.max_nl_sen_green_bound<=max_nl_sen_green_bound
        )

    def solve_sum_whether_sentence(self, TimeLimit=None, MIPGap=None, wm_bound=1, nl_bound=1):
        self.model.sum_whether_sentence_obj=pyomo.Objective(
            expr  = self.model.sum_whether_sentence,
            sense = pyomo.maximize
        )
        self.solve(TimeLimit=TimeLimit, MIPGap=MIPGap, warmstart=False)
        self.model.sum_whether_sentence_obj.deactivate()
        
        sum_whether_nl_sentence=np.round(pyomo.value(self.model.sum_whether_nl_sentence),2)
        sum_whether_wm_sentence=np.round(pyomo.value(self.model.sum_whether_wm_sentence),2)
        self.log_info(to_string(('sum_whether_nl_sentence: ',sum_whether_nl_sentence)))
        self.log_info(to_string(('sum_whether_wm_sentence: ',sum_whether_wm_sentence )))

        sum_whether_sentence=pyomo.value(self.model.sum_whether_sentence)
        if sum_whether_sentence>0:
            whether_sentence={
                sentence_id:pyomo.value(self.model.whether_sentence[sentence_id])
                for sentence_id in self.sentence_index
            }
            def sum_whether_sentence_obj_con_expr(model, sentence_id):
                return self.model.whether_sentence[sentence_id]==whether_sentence[sentence_id]
            self.model.sum_whether_sentence_obj_con=pyomo.Constraint(
                self.sentence_index,
                rule=sum_whether_sentence_obj_con_expr
            )
    
    def solve_min_key_num(self, TimeLimit=None):
        self.model.sum_whether_key_obj=pyomo.Objective(
            expr  = self.model.sum_whether_key, 
            sense = pyomo.minimize
        )
        self.solve(TimeLimit=TimeLimit)
        sum_whether_key=pyomo.value(self.model.sum_whether_key)
        
        self.log_info(to_string(('least_key_num:', sum_whether_key)))
        
        self.model.sum_whether_key_obj.deactivate()
        self.model.sum_whether_key_obj_con=pyomo.Constraint(
            expr=self.model.sum_whether_key<=sum_whether_key
        )
        return
    
    def solve_green_num(self, max_min=False, TimeLimit=None, MIPGap=None, wm_bound=0.95, nl_bound=1.05, MIPGapAbs=None,):
        
        if max_min:
            self.model.sum_green_token_obj=pyomo.Objective(
                expr  = self.model.sum_green_token, 
                sense = pyomo.maximize
            )
        else:
            self.model.sum_green_token_obj=pyomo.Objective(
                expr  = self.model.sum_green_token, 
                sense = pyomo.minimize
            )
        self.solve(TimeLimit=TimeLimit, MIPGap=MIPGap, MIPGapAbs=MIPGapAbs)
        self.model.sum_green_token_obj.deactivate()
        sum_green_token=pyomo.value(self.model.sum_green_token)
        
        if max_min:
            self.model.sum_green_token_obj_con=pyomo.Constraint(
                expr=self.model.sum_green_token>=sum_green_token*wm_bound
            )
        else:
            self.model.sum_green_token_obj_con=pyomo.Constraint(
                expr=self.model.sum_green_token<=sum_green_token*nl_bound
            )
        self.log_info(to_string(('sum_green_token:', sum_green_token)))
        return
    
    def solve_min_key_sentence_num(self, TimeLimit=None, MIPGap=None):
        self.model.min_key_sentence_num_obj=pyomo.Objective(
            expr  = self.model.min_key_sentence_num, 
            sense = pyomo.maximize
        )
        self.solve(TimeLimit=TimeLimit, MIPGap=MIPGap)
        self.model.min_key_sentence_num_obj.deactivate()
        min_key_sentence_num=pyomo.value(self.model.min_key_sentence_num)
        self.log_info(to_string(('min_key_sentence_num', min_key_sentence_num)))
        self.model.min_key_sentence_num_obj_con=pyomo.Constraint(
            expr=self.model.min_key_sentence_num>=min_key_sentence_num
        )
        return      

    def count_token_green(self, sentence, token_color):
        key=0
        green_num=[]
        for tmp_key in token_color:
            tmp_green_num=0
            for token_id in sentence:
                if token_id not in token_color[tmp_key]:
                    continue
                tmp_green_num+=token_color[tmp_key][token_id][1]
            green_num.append(tmp_green_num)

        green_num=np.array(green_num)+1e-4
        key=np.argmax(green_num)
        return green_num, self.key_index[key]
    
    def fix_sentence_key(self, random_flag=False, file_path=''):
        
        for (sen_id, key_id) in self.wm_sentence_key_index:
            self.model.sentence_key[(sen_id, key_id)].unfix()

        fix_wm_sentence_key={
            (sentence_id, key_id): 0
            for sentence_id in self.wm_sentence_index
            for key_id in self.key_index
        }
        if random_flag:
            
            np.random.seed(123)
            for sen_id in self.wm_sentence_index:
                key_id=self.key_index[np.random.randint(len(self.key_index))]
                fix_wm_sentence_key[(sen_id, key_id)]=1
        else:
            token_color=self.token_color_list[-1]
            for sen_id in self.wm_sentence_index:
                green_num, key_id=self.count_token_green(self.dataset[sen_id], token_color)
                fix_wm_sentence_key[(sen_id, key_id)]=1
        
        for (sen_id, key_id) in self.wm_sentence_key_index:
            self.model.sentence_key[(sen_id, key_id)].fix(fix_wm_sentence_key[(sen_id, key_id)])
        self.log_info('fix done')
        if len(file_path)>0:
            tmp_fix_wm_sentence_key={
                str(sen_id)+'_'+str(key_id): fix_wm_sentence_key[(sen_id, key_id)]
                for (sen_id, key_id) in fix_wm_sentence_key
            }
            save_json(tmp_fix_wm_sentence_key, file_path)
        return
    
    def delete_comp(self):
        try:
            del self.model.wm_nl_sum_green_obj
        except:
            pass
        try:
            del self.model.new_sum_sen_green_bound_wm_con
        except:
            pass
        try:
            del self.model.new_sum_sen_green_bound_nl_con
        except:
            pass
        
        try:
            del self.model.sum_whether_wm_sentence_obj_con
        except:
            pass
        try:
            del self.model.sum_whether_nl_sentence_obj_con
        except:
            pass
        try:
            del self.model.sum_green_token_obj
        except:
            pass
        try:
            del self.model.sum_green_token_obj_con
        except:
            pass
        return