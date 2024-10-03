
import torch
import torch.nn.functional as F
from GensimModel import GensimModel
from get_greenlist import get_greenlist
import numpy as np
import os
from utli import load_json
from tqdm import tqdm

from utli import Logger,to_string, remove_repeat
from greenlist_inversion import count_green, count_z_score_bound
from Sentence_Embedder import Sentence_Embedder
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer, util
from transformers import OPTForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM

class WM_Wiper:

    def __init__(
        self, 
        tokenizer_tag, 
        device='cuda:0', 
        token_len_flag='False',
    ):

        self.gensimi=None
        if 'opt' in tokenizer_tag:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_tag, token=False)
        elif 'llama' in tokenizer_tag:
            self.tokenizer = LlamaTokenizer.from_pretrained(
                tokenizer_tag
            )
        self.vocab_size=self.tokenizer.vocab_size
        self.sentence_embedder=None
        self.ppl_model=None
        if token_len_flag=='False':
            self.token_len_flag=False
        else:
            self.token_len_flag=True
        self.device=device
        
    
    def set_para(self, gamma=0.25, z_threshold=4):
        self.gamma=gamma
        self.z_threshold=z_threshold
        
    
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
        return
    
    def get_z_threshold_num(self, token_num):
        return count_z_score_bound(token_num, self.z_threshold, self.gamma)

    def load_token_color(self, dir_path, file_name):
        file_path=os.path.join(dir_path, file_name)
        self.token_color=load_json(file_path=file_path)
        self.token_color_tokens={
            key: {
                str(token):0
                for token in range(self.tokenizer.vocab_size)
            }
            for key in self.token_color
        }
        for key in self.token_color:
            for token in self.token_color[key]:
                self.token_color_tokens[key][token]=1


    def count_token_green(self, sentence):
        key=0
        green_num=0
        for tmp_key in self.token_color:
            tmp_green_num=0
            for token_id in sentence:
                if str(token_id) not in self.token_color[tmp_key]:
                    continue
                tmp_green_num+=self.token_color[tmp_key][str(token_id)][1]
            if tmp_green_num>green_num:
                green_num=tmp_green_num
                key=tmp_key
        return green_num, key

    def compare_token_color(self, sentence):
        token_ids=self.tokenizer.encode(sentence)
        token_ids=remove_repeat(token_ids)
        ture_green_num=self.count_true_green(token_ids)[0]
        z_score_bound=self.get_z_threshold_num(len(token_ids))
        green_num=self.count_token_green(token_ids)[0]

        return ture_green_num, z_score_bound, green_num

    def count_true_green_key(self, token_ids, key):
        green_num=0
        for token in token_ids:
            green_num+=self.true_green_dict[key][token]
        return green_num    

    def count_true_green(self, token_ids):
        green_num=0
        key=''
        for tmp_key in self.true_green:
            tmp_num=self.count_true_green_key(token_ids, tmp_key)
            if green_num<tmp_num:
                green_num=tmp_num
                key=tmp_key
        return green_num, key

    def count_green_key(self, token_ids, key):
        green_num=0
        for token in token_ids:
            if self.token_color_tokens[key][str(token)]==1:
                green_num+=self.token_color[key][str(token)][1]
        return green_num    

    def count_green(self, token_ids):
        green_num=0
        key=''
        for tmp_key in self.true_green:
            tmp_num=self.count_green_key(token_ids, tmp_key)
            if green_num<=tmp_num:
                green_num=tmp_num
                key=tmp_key
        return green_num, key
    
    def check_green(self, token_id, key):
        if self.token_color_tokens[key][str(token_id)]==1 and (self.token_color[key][str(token_id)][1]==1):
            return True
        return False
    
    def evaluate_green_token(self):
        green_num=[]
        true_green_num=[]
        true_green_acc=[]
        
        scores={}
        for key1 in self.token_color:
            for key2 in self.true_green_dict:
                tmp_green_num=0
                tmp_true_green_num=0
                for token_id in self.token_color[key1]:
                    tmp_green_num+=self.token_color[key2][token_id][1]
                    if self.token_color[key1][token_id][1]==1 and self.true_green_dict[key2][int(token_id)]==1:
                        tmp_true_green_num+=1
                if tmp_green_num==0:
                    tmp_true_green_acc=0
                else:
                    tmp_true_green_acc=np.round(tmp_true_green_num/tmp_green_num,4)
                scores[(key1,key2)]=[tmp_green_num, tmp_true_green_num,tmp_true_green_acc]
        used_key=[]
        for key1 in self.token_color:
            max_key=''
            max_acc=0
            for key2 in self.true_green_dict:
                if key2 in used_key:
                    continue
                if scores[(key1,key2)][2]>max_acc:
                    max_key=key2
                    max_acc=scores[(key1,key2)][2]
                    tmp_green_num=scores[(key1,key2)][0]
                    tmp_true_green_num=scores[(key1,key2)][1]
                    tmp_true_green_acc=scores[(key1,key2)][2]
            
            green_num.append(tmp_green_num)
            true_green_num.append(tmp_true_green_num)
            true_green_acc.append(tmp_true_green_acc)
            used_key.append(max_key)
            
        return green_num, true_green_num, true_green_acc

    def substitute(self, token_id, key):
        if self.gensimi is None:
            self.gensimi=GensimModel()
        token=self.token_color[key][str(token_id)][0]
        space_flag=False
        if ' ' in token:
            space_flag=True
        token=token.replace(' ', '').lower()
        if self.token_len_flag:
            if len(token)<=2:
                return []
            if token in ['that', 'the', 'to', 'of', 'is', 'are','be','on','in','it','an','and','for']:
                return []
        simi_token_ids=[]
        simi_tokens=self.gensimi.find_simi_words(token, simi_num=self.simi_num_for_token)
        for simi_token in simi_tokens:
            if space_flag:
                simi_token=' '+simi_token
            simi_token_id=self.tokenizer.encode(simi_token)[1]
            if self.check_green(simi_token_id, key):
                continue
            else:
                simi_token_ids.append(simi_token_id)
        return simi_token_ids
    
    def find_red_simi(self, token_id, key):
        token=self.token_color[key][str(token_id)][0]
        space_flag=False
        if ' ' in token:
            space_flag=True
        token=token.replace(' ', '').lower()
        
        simi_tokens=self.gensimi.find_simi_words(token, simi_num=self.simi_num_for_token)
        red_simi=[]
        for simi_token in simi_tokens:
            if self.token_len_flag:
                if len(token)<=2:
                    continue
                if token in ['that', 'the', 'to', 'of', 'is', 'are','be','on','in','it','an','and','for']:
                    continue
            if space_flag:
                simi_token=' '+simi_token
            simi_token_id=self.tokenizer.encode(simi_token)[1]
            if self.check_green(simi_token_id, key):
                continue
            else:
                red_simi.append(simi_token_id) 
        return red_simi

    def wm_wipe_greedy(
        self, sentence, 
        simi_num_for_token=3,
        ppl_model_id=None,
        max_edit_rate=1,
    ):
        if self.ppl_model is None:
            if ppl_model_id is None:
                pass
            elif 'opt' in ppl_model_id:
                self.ppl_model=OPTForCausalLM.from_pretrained(ppl_model_id).to(self.device)
            elif 'llama' in ppl_model_id:
                self.ppl_model=LlamaForCausalLM.from_pretrained(
                    ppl_model_id, 
                    torch_dtype=torch.float16, 
                    # device_map='auto',
                ).to(self.device)

        self.simi_num_for_token=simi_num_for_token

        token_ids=self.tokenizer.encode(sentence)
        token_ids=remove_repeat(token_ids)
        sentence=self.tokenizer.decode(token_ids)
        token_num=len(token_ids)
        
        raw_true_green_num, raw_true_key=self.count_true_green(token_ids=token_ids)
        raw_green_num, raw_key=self.count_green(token_ids=token_ids)
        z_threshold_num=self.get_z_threshold_num(token_num)

        if raw_true_green_num<z_threshold_num:
            return (
                sentence, 
                -1, raw_true_green_num, z_threshold_num
            )
        
        if self.ppl_model is not None:
            with torch.no_grad():
                raw_ppl = self.get_ppl(
                    torch.tensor(token_ids).to(self.device),
                    max_length=20,
                    stride=20,
                )
                raw_ppl = raw_ppl.item()
        else:
            raw_ppl = 0

        edit_dist=0
        new_token_ids=[]
        max_edit_dist=max_edit_rate*len(token_ids)
        for token_id in token_ids:
            if self.check_green(token_id, raw_key) and edit_dist<max_edit_dist:
                tmp_subst=self.substitute(token_id, raw_key)
                if len(tmp_subst)>0:
                    new_token_ids.append(tmp_subst[0])
                    edit_dist+=1
                    continue
            new_token_ids.append(token_id)
        
        new_true_green_num, new_true_key=self.count_true_green(token_ids=new_token_ids)
        if new_true_green_num>=z_threshold_num:
            wipe_success=0
        else:
            wipe_success=1

        new_sentence=self.tokenizer.decode(new_token_ids)
        
        if self.ppl_model is not None:
            with torch.no_grad():
                new_ppl = self.get_ppl(
                    torch.tensor(new_token_ids).to(self.device),
                    max_length=20,
                    stride=20,
                )
                new_ppl = new_ppl.item()
        else:
            new_ppl = 0

        return (
            new_sentence, 
            wipe_success, raw_true_green_num, new_true_green_num, z_threshold_num, new_ppl,raw_ppl, raw_true_key
        )

    def wm_wipe_beam_search_simi(
        self, sentence, 
        simi_num_for_token=10, 
        beam_size=3, candi_num=3,
        embedder_name='hkunlp/instructor-large', 
    ):
        if self.sentence_embedder is None:
            self.sentence_embedder=Sentence_Embedder()
        self.simi_num_for_token=simi_num_for_token
        
        raw_embedding=self.sentence_embedder.get_embedding(sentence)
        token_ids=self.tokenizer.encode(sentence, max_length=500)
        token_ids=remove_repeat(token_ids)
        sentence=self.tokenizer.decode(token_ids)
        token_num=len(token_ids)
        
        raw_green_num, raw_key=self.count_true_green(token_ids=token_ids)
        z_threshold_num=self.get_z_threshold_num(token_num)

        if raw_green_num<z_threshold_num:
            return (
                -1, raw_green_num, z_threshold_num
            )
        
        new_token_ids=[[]for _ in range(beam_size)]
        for idx in range(len(token_ids)):
            token_id=token_ids[idx]
            if self.check_green(token_id, raw_key):
                tmp_subt=self.substitute(token_id, raw_key)[0:candi_num]
                if len(tmp_subt)>0:
                    tmp_sen_list=[]
                    for idz in range(len(tmp_subt)):
                        for idy in range(beam_size):
                            tmp_sen=new_token_ids[idy]+[tmp_subt[idz]]+token_ids[idx+1:]
                            if len(tmp_sen_list)==0:
                                tmp_sen_list.append(tmp_sen)
                            elif len(tmp_sen_list)>0 and tmp_sen!=tmp_sen_list[-1]:
                                tmp_sen_list.append(tmp_sen)
                    tmp_cos=[
                        (
                            tmp_sen, 
                            util.cos_sim(
                                raw_embedding, 
                                self.sentence_embedder.get_embedding(self.tokenizer.decode(tmp_sen))
                            )
                        )
                        for tmp_sen in tmp_sen_list
                    ]
                    new_token_ids=[
                        tmp_sen[0][0:idx+1]
                        for tmp_sen in sorted(tmp_cos, key=lambda x: x[1], reverse=True)[0:beam_size]
                    ]
                    continue
            for idy in range(beam_size):
                new_token_ids[idy].append(token_id)
        
        new_green_num, new_key=self.count_true_green(token_ids=new_token_ids[0])
        if new_green_num>=z_threshold_num:
            wipe_success=0
        else:
            wipe_success=1

        new_sentence=self.tokenizer.decode(new_token_ids[0])

        new_embedding=self.sentence_embedder.get_embedding(new_sentence)
        cos_simi=util.cos_sim(raw_embedding, new_embedding).item()

        return (
            wipe_success, raw_green_num, new_green_num, z_threshold_num, cos_simi
        )
    
    def get_ppl(self, input_ids, max_length=10, stride = 10):
        
        input_ids=input_ids.reshape((1,-1))
        seq_len = input_ids.size(1)

        prev_end_loc = 0
        iters=int(seq_len/stride)
        neg_log_likelihood=torch.tensor(0.0).to(self.device)
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  
            tmp_input_ids = input_ids[:,begin_loc:end_loc].to(self.device)
            target_ids = tmp_input_ids.clone()
            target_ids[:,0:-trg_len] = -100
            if (end_loc-begin_loc)<stride:
                break

            outputs = self.ppl_model(tmp_input_ids.long(), labels=target_ids.long())

            neg_log_likelihood += outputs.loss/iters


            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        return neg_log_likelihood

    def wm_wipe_gumbel(
        self, sentence, 
        simi_num_for_token=10, 
        iters=200,
        ppl_model_id='facebook/opt-1.3b',
        learning_rate=5
    ):
        if self.ppl_model is None:
            if 'opt' in ppl_model_id:
                self.ppl_model=OPTForCausalLM.from_pretrained(ppl_model_id).to(self.device)
            elif 'llama' in ppl_model_id:
                self.ppl_model=LlamaForCausalLM.from_pretrained(
                ppl_model_id, 
                torch_dtype=torch.float16, 
            ).to(self.device)
        self.simi_num_for_token=simi_num_for_token

        token_ids=self.tokenizer.encode(sentence)
        token_ids=remove_repeat(token_ids)
        sentence=self.tokenizer.decode(token_ids)
        token_num=len(token_ids)
        
        raw_green_num, raw_key=self.count_true_green(token_ids=token_ids)
        z_threshold_num=self.get_z_threshold_num(token_num)

        if raw_green_num<z_threshold_num:
            return (
                sentence, 
                -1, raw_green_num, z_threshold_num
            )
        
        if 'llama' in ppl_model_id:
            embedding_dim=self.ppl_model.model.embed_tokens.embedding_dim
        else:
            embedding_dim=self.ppl_model.model.decoder.embed_tokens.embedding_dim
        red_vec_dict={}
        red_simi_dict={}
        red_mask_dict={}
        red_simi_emb_dict={}
        for idx in range(len(token_ids)):
            token_id=token_ids[idx]
            if self.check_green(token_id, raw_key) and token_id not in red_vec_dict:
                red_simi=self.find_red_simi(token_id, raw_key)
                if len(red_simi)==0:
                    continue
                red_vec_dict[token_id]=torch.rand((1, len(red_simi))).to(self.device)
                red_simi_dict[token_id]=torch.tensor(red_simi).to(self.device)
                if 'llama' in ppl_model_id:
                    red_simi_emb_dict[token_id]=self.ppl_model.model.embed_tokens(red_simi_dict[token_id])
                else:
                    red_simi_emb_dict[token_id]=self.ppl_model.model.decoder.embed_tokens(red_simi_dict[token_id])
                red_mask_dict[token_id]=(
                    torch.ones((len(token_ids),embedding_dim)).to(self.device),
                    torch.zeros((len(token_ids),embedding_dim)).to(self.device)
                )
                red_mask_dict[token_id][0][idx]=0
                red_mask_dict[token_id][1][idx]=1
        
        raw_token_ids=torch.tensor(token_ids).reshape((1,-1))
        with torch.no_grad():
            raw_ppl = self.get_ppl(
                raw_token_ids,
                max_length=20,
                stride=20,
            )
        
        if 'llama' in ppl_model_id:
            raw_embedding=self.ppl_model.model.embed_tokens(raw_token_ids.to(self.device))
        else:
            raw_embedding=self.ppl_model.model.decoder.embed_tokens(raw_token_ids.to(self.device))
        new_embedding=raw_embedding.clone()
        new_embedding=new_embedding.to(self.device)
        self.ppl_model.eval()
        for iter in range(iters):
            self.ppl_model.zero_grad()
            
            new_embedding=new_embedding.clone().detach().to(self.device)
            raw_token_ids=raw_token_ids.clone().detach().to(self.device)
            for token_id in red_vec_dict:
                red_simi_emb_dict[token_id]=red_simi_emb_dict[token_id].clone().detach().to(self.device)
                red_vec_dict[token_id]=red_vec_dict[token_id].clone().detach().float().requires_grad_(True).to(self.device)
                
                g_label=F.gumbel_softmax(
                    red_vec_dict[token_id], tau=1, 
                )
                tmp_red_token=g_label.T*red_simi_emb_dict[token_id]
                tmp_emb=red_mask_dict[token_id][1]*torch.sum(tmp_red_token, dim=0)
                tmp_emb=tmp_emb.reshape((1,-1,embedding_dim))
                new_embedding=tmp_emb+red_mask_dict[token_id][0]*new_embedding
            if 'llama' in ppl_model_id:
                outputs = self.ppl_model.forward(inputs_embeds=new_embedding.half(), labels=raw_token_ids.to(self.device))
            else:
                outputs = self.ppl_model.forward(inputs_embeds=new_embedding, labels=raw_token_ids.to(self.device))
            
            outputs.loss.backward()
            for token_id in red_vec_dict:
                grad = red_vec_dict[token_id].grad.data
                red_vec_dict[token_id] = red_vec_dict[token_id]-grad*learning_rate
          
        new_token_ids_dict={}
        for token_id in red_vec_dict:
            tmp_idx=torch.argmax(red_vec_dict[token_id]).item()
            new_token_ids_dict[token_id]=red_simi_dict[token_id][tmp_idx].item()
            
        new_token_ids=[]
        for idx in range(len(token_ids)):
            token_id=token_ids[idx]
            if token_id in new_token_ids_dict:
                new_token_ids.append(new_token_ids_dict[token_id])
            else:
                new_token_ids.append(token_id)
        
        with torch.no_grad():
            new_ppl = self.get_ppl(
                torch.tensor(new_token_ids),
                max_length=20,
                stride=20,
            )
        new_green_num, new_key=self.count_true_green(token_ids=new_token_ids)
        if new_green_num>=z_threshold_num:
            wipe_success=0
        else:
            wipe_success=1

        new_sentence=self.tokenizer.decode(new_token_ids)

        return (
            new_sentence, 
            wipe_success, raw_green_num, new_green_num, z_threshold_num, new_ppl.item(),raw_ppl.item()
        )

    def wm_wipe_beam_search_ppl(
        self, sentence, 
        simi_num_for_token=10, 
        beam_size=3, candi_num=3,
        ppl_model_id='facebook/opt-1.3b', 
    ):
        if self.ppl_model is None:
            if 'opt' in ppl_model_id:
                self.ppl_model=OPTForCausalLM.from_pretrained(ppl_model_id).to(self.device)
            elif 'llama' in ppl_model_id:
                self.ppl_model=LlamaForCausalLM.from_pretrained(
                ppl_model_id, 
                torch_dtype=torch.float16, 
                # device_map='auto',
            ).to(self.device)
        
        self.simi_num_for_token=simi_num_for_token

        token_ids=self.tokenizer.encode(sentence)
        token_ids=remove_repeat(token_ids)
        sentence=self.tokenizer.decode(token_ids)
        token_num=len(token_ids)
        
        raw_green_num, raw_key=self.count_true_green(token_ids=token_ids)
        z_threshold_num=self.get_z_threshold_num(token_num)
        raw_token_ids=torch.tensor(token_ids).reshape((1,-1))

        if raw_green_num<z_threshold_num:
            return (
                sentence, 
                -1, raw_green_num, z_threshold_num
            )
        
        with torch.no_grad():
            raw_ppl = self.get_ppl(
                raw_token_ids,
                max_length=20,
                stride=20,
            )
        
        new_token_ids=[[] for _ in range(beam_size)]
        for idx in range(len(token_ids)):
            token_id=token_ids[idx]
            if self.check_green(token_id, raw_key):
                tmp_subt=self.substitute(token_id, raw_key)[0:candi_num]
                if len(tmp_subt)>0:
                    tmp_sen_list=[]
                    for idz in range(len(tmp_subt)):
                        for idy in range(beam_size):
                            tmp_sen=new_token_ids[idy]+[tmp_subt[idz]]+token_ids[idx+1:]
                            if len(tmp_sen_list)==0:
                                tmp_sen_list.append(tmp_sen)
                            elif len(tmp_sen_list)>0 and tmp_sen!=tmp_sen_list[-1]:
                                tmp_sen_list.append(tmp_sen)
                    tmp_cos=[
                        (
                            tmp_sen, 
                            self.get_ppl(
                                torch.tensor(tmp_sen).reshape((1,-1)),
                                max_length=20,
                                stride=20,
                            ).item()
                        )
                        for tmp_sen in tmp_sen_list
                    ]
                    new_token_ids=[
                        tmp_sen[0][0:idx+1]
                        for tmp_sen in sorted(tmp_cos, key=lambda x: x[1])[0:beam_size]
                    ]
                    new_token_ids=new_token_ids+[[]]*(beam_size-len(new_token_ids))
                    continue
            for idy in range(beam_size):
                new_token_ids[idy].append(token_id)
        
        new_green_num, new_key=self.count_true_green(token_ids=new_token_ids[0])
        if new_green_num>=z_threshold_num:
            wipe_success=0
        else:
            wipe_success=1

        with torch.no_grad():
            new_ppl = self.get_ppl(
                torch.tensor(new_token_ids[0]),
                max_length=20,
                stride=20,
            )

        new_sentence=self.tokenizer.decode(new_token_ids[0])

        return (
            new_sentence, 
            wipe_success, raw_green_num, new_green_num, z_threshold_num, new_ppl.item(),raw_ppl.item()
        )
