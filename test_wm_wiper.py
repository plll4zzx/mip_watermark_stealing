
# import torch
# import torch.nn.functional as F
# from GensimModel import GensimModel
# from get_greenlist import get_greenlist
import numpy as np
# from greenlist_inversion import count_green, count_z_score_bound
# from copy import deepcopy
# from Sentence_Embedder import Sentence_Embedder
# from InstructorEmbedding import INSTRUCTOR
# from sentence_transformers import SentenceTransformer, util
# from tokenizations import get_alignments
# import Levenshtein
# from transformers import GPT2LMHeadModel, GPT2TokenizerFast, OPTForCausalLM, AutoTokenizer
# import copy
import os
from utli import load_json, save_json, check_dir
from tqdm import tqdm
from utli import Logger,to_string
import datetime
from get_greenlist import get_greenlist_filename
import random
from wm_wiper import WM_Wiper
import argparse
from model_inversion_config import config, config_sen, config_pro, config_sta_max_num

if __name__=='__main__':
    
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    parser = argparse.ArgumentParser(description='test WM_Wiper')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='facebook/opt-1.3b')
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--delta', type=int, default=2)
    # parser.add_argument('--data_num', type=int)
    # parser.add_argument('--wm_bound', type=float, default=0.95)
    # parser.add_argument('--nl_bound', type=float, default=0.95)

    parser.add_argument('--wm_seed', type=int, default=123)
    parser.add_argument('--nl_seed', type=int, default=456)
    parser.add_argument('--rand_num', type=int, default=0)
    
    parser.add_argument('--query_flag', type=str, default='True')
    parser.add_argument('--gamma_flag', type=str, default='True')
    parser.add_argument('--oracle_flag', type=str, default='False')
    parser.add_argument('--naive_flag', type=str, default='True')

    parser.add_argument('--expect_green_size', type=int, default=1)
    parser.add_argument('--z_threshold', type=int, default=4)
    # parser.add_argument('--sentence_up_num', type=float, default=0.99)
    # parser.add_argument('--sentence_down_num', type=float, default=0.98)

    parser.add_argument('--wp_mode', type=str, default='greedy')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--candi_num', type=int, default=3)

    parser.add_argument('--attack_type', type=str, default='op')
    parser.add_argument('--token_len', type=str, default='False')
    parser.add_argument('--max_edit_rate', type=float, default=1)

    args = parser.parse_args()
    device = 'cuda:'+str(args.device)
    model_name = args.model_name
    gamma = args.gamma
    delta = args.delta
    # wm_bound = args.wm_bound
    # nl_bound = args.nl_bound

    expect_green_size = args.expect_green_size
    z_threshold = args.z_threshold
    # sentence_up_num = args.sentence_up_num
    # sentence_down_num = args.sentence_down_num

    wm_seed = args.wm_seed
    nl_seed = args.nl_seed
    rand_num = args.rand_num
    
    query_flag = args.query_flag
    gamma_flag = args.gamma_flag
    oracle_flag = args.oracle_flag
    naive_flag = args.naive_flag

    wp_mode=args.wp_mode
    beam_size=args.beam_size
    candi_num=args.candi_num

    attack_type=args.attack_type
    token_len=args.token_len

    max_edit_rate=np.round(args.max_edit_rate,2)

    if query_flag=='True' or naive_flag=='True' or oracle_flag=='True':
        sentence_up_num, sentence_down_num=1,1
    
    check_dir('log')
    log=Logger(
        'log/WM_Wiper_'+'_'.join((
            model_name.replace('/', '_'), str(gamma), str(delta),
            attack_type, wp_mode, str(beam_size), str(candi_num),
            device, 
            # str(wm_bound), str(nl_bound),
            str(datetime.datetime.now().date()),
        ))+'.log',
        level='debug', screen=False
    )

    dir_path='saved_data'
    save_path='saved_res'
    check_dir(dir_path)
    check_dir(save_path)
    input_type='.json'
    # time.sleep(1440)
    wm_type='o'
    dataset_name='c4_realnewslike'
    # model_name_list= ['facebook/opt-1.3b']#, 'facebook/opt-2.7b'
    wm_level_list = ['model']#'model','sentence',  'token', 'sentence_fi'
    dataset_num=str(30000)
    key_token_list=[123]
    key_num=1
    wm_level = wm_level_list[0]
    # gamma=0.25
    # delta=2
    random_seed=789#456

    with_watermark_list_len=1000
    
    def log_print(tmp_str):
        print(tmp_str)
        log.logger.info(tmp_str)
    perb_rate=0
    for data_num in [
        # 3000, 
        # 6000
        2000, 
        5000, 
        10000, 
        20000
    ]:
    # for (data_num, perb_rate) in [ 
    #     # (5000,0.1),(5000,0.3),
    #     (10000,0.5),
    #     (10000,0.7),
    # ]:
        
        if query_flag=='True' and gamma_flag=='True' and oracle_flag=='False' and naive_flag=='False' and key_num==1:
            (wm_bound, nl_bound, sentence_up_num, sentence_down_num) = config_pro[model_name][(gamma, delta)][data_num]
        # elif perb_rate>0:
        #     (wm_bound, nl_bound, sentence_up_num, sentence_down_num) = config_conf[model_name][(gamma, delta)][data_num]
        elif key_num>1:
            (sentence_up_num, sentence_down_num, wm_bound, nl_bound, min_max_green_bound, expect_green_size, min_green_num) = config_sen[model_name][(gamma, delta, data_num)]
        else:
            (wm_bound, nl_bound, sentence_up_num, sentence_down_num) = config[model_name][(gamma, delta)][data_num]
            if model_name=='facebook/opt-1.3b' and gamma==0.25 and query_flag=='True' and gamma_flag=='True'and oracle_flag=='True' and naive_flag=='False':
                nl_bound=1
            elif model_name=='facebook/opt-1.3b' and gamma==0.5 and query_flag=='True' and gamma_flag=='True'and oracle_flag=='True' and naive_flag=='False':
                wm_bound=0.95
            elif model_name=='facebook/opt-1.3b' and gamma==0.5 and query_flag=='True' and gamma_flag=='True'and oracle_flag=='False' and naive_flag=='True':
                wm_bound=0.95
            elif model_name=='../model/llama-2-7b' and gamma==0.25 and query_flag=='True' and gamma_flag=='True'and oracle_flag=='True' and naive_flag=='False':
                nl_bound=1
            # elif model_name=='../model/llama-2-7b' and gamma==0.25 and query_flag=='True' and gamma_flag=='True'and oracle_flag=='False' and naive_flag=='True':
            #     nl_bound=1.0
        # sentence_up_num, sentence_down_num=0.9,0.8
        if query_flag=='True' or naive_flag=='True' or oracle_flag=='True':
            sentence_up_num, sentence_down_num=1,1
        if naive_flag=='True' or oracle_flag=='True':
            wm_bound, nl_bound=1,1
        if perb_rate>0:
            data_num+=perb_rate
            sentence_up_num=np.round(sentence_up_num-perb_rate-0.05,2)
            if gamma==0.5:
                perb_rate+=0.2
            else:
                perb_rate+=0.1
            sentence_down_num=np.round(sentence_down_num-perb_rate,2)
            # wm_bound, nl_bound=1,1

        if attack_type =='sta':
            token_color_file_name='_'.join([
                'sta',wm_level, wm_type, dataset_name, model_name.replace('/','_'), dataset_num,
                str(key_token_list), str(key_num), 
                str(wm_seed), str(nl_seed), str(rand_num), 
                str(data_num), str(data_num),
                str(gamma), str(delta),
                str(expect_green_size),
                str(z_threshold), 
                str(query_flag), 
                # str(gamma_flag), 
                'token_color.json'
            ])
        else:
            token_color_file_name='_'.join([#str(15)+'_'+
                wm_level, wm_type, dataset_name, model_name.replace('/','_'), dataset_num,
                str(key_token_list), str(key_num), 
                str(wm_seed), str(nl_seed), str(rand_num), 
                str(data_num), str(data_num),
                str(gamma), str(delta),
                str(expect_green_size),
                str(z_threshold), 
                str(query_flag), str(gamma_flag), str(oracle_flag),
                str(sentence_up_num), str(sentence_down_num),
                str(wm_bound), str(nl_bound), 
                str(naive_flag),
                'token_color.json'
            ])

        
        # if os.path.isfile(os.path.join(save_path,token_color_file_name))==False:
        #     print('********')
        #     print(token_color_file_name)
        #     print('********')
        #     continue
        # else:
        #     continue

        output_file_path=os.path.join(
            'saved_data/res_wp',
            '_'.join([
                'wp', attack_type, wp_mode, 
                str(beam_size), str(candi_num),str(with_watermark_list_len),
                token_len,
                # str(max_edit_rate),
                token_color_file_name,
            ])
        )
        # output_file_path2=os.path.join(
        #     dir_path,
        #     '_'.join(['wp_result', str(with_watermark_list_len)])+'_'+token_color_file_name
        # )
        # output_file_path3=os.path.join(
        #     dir_path,
        #     '_'.join(['wm', attack_type, wp_mode, str(beam_size), str(candi_num)])+'_'+str(with_watermark_list_len)+'_'+token_color_file_name
        # )
        
        # if os.path.isfile(output_file_path):
        #     log_print(to_string((model_name,gamma, delta,data_num,wm_bound, nl_bound, sentence_up_num, sentence_down_num)))
        #     continue

        log_print('\n')
        log_print(output_file_path)
        log_print(to_string((model_name, wm_level, data_num, random_seed)))
        log_print(to_string(('attack_type',attack_type)))
        log_print(to_string(('wp_mode',wp_mode)))
        log_print(to_string(('beam_size',beam_size)))
        log_print(to_string(('candi_num',candi_num)))
        log_print(to_string(('gamma',gamma)))
        log_print(to_string(('delta',delta)))
        log_print(to_string(('data_num',data_num)))
        log_print(to_string(('wm_bound',wm_bound)))
        log_print(to_string(('nl_bound',nl_bound)))
        
        file_name='_'.join((
            dataset_name, 
            model_name.replace('/','_'),
            wm_level,wm_type,
            dataset_num,
            str(key_num),
            str(gamma), str(delta)
        ))
        
        input_file_path=os.path.join(dir_path,file_name+input_type)
        data_json=load_json(input_file_path)
        
        with_watermark_list=[]
        without_watermark_list=[]
        re_sentence_list=[]

        for data in data_json:
            if ('redecoded_input' in data) and ('output_with_watermark' in data) and ('output_without_watermark' in data):
                with_watermark_list.append(data['output_with_watermark'])
                without_watermark_list.append(data['output_without_watermark'])
                re_sentence_list.append(data['redecoded_input'])
        
        # tmp_wm_data=self.tokenizer.batch_encode_plus(with_watermark_list).data['input_ids']
        random.seed(random_seed)
        random.shuffle(with_watermark_list)
        with_watermark_list=with_watermark_list[0:with_watermark_list_len]
        
        wm_wiper=WM_Wiper(tokenizer_tag=model_name, device=device, token_len_flag=token_len)#, log=log
        wm_wiper.set_para(gamma=gamma, z_threshold=4)
        if wm_level=='model':
            green_list_name= get_greenlist_filename(key_token_list, gamma, model_name)
        else:
            green_list_name= get_greenlist_filename(key_token_list, gamma+delta, model_name+'_'+wm_level)
        wm_wiper.load_true_green(os.path.join(dir_path, green_list_name))

        wm_wiper.load_token_color(
            save_path, 
            token_color_file_name
        )

        result_dict=[]
        sen_num=len(with_watermark_list)
        success_num, raw_green_num, new_green_num=0,0,0
        cos_simi=0
        edit_dist=0
        for sentence in tqdm(with_watermark_list, ncols=100, desc='wiper', leave=True):
            
            if wp_mode == 'gumbel':
                result=wm_wiper.wm_wipe_gumbel(
                    sentence, 
                    ppl_model_id=model_name
                )
            elif wp_mode == 'beam':
                result=wm_wiper.wm_wipe_beam_search_ppl(
                    sentence, 
                    ppl_model_id=model_name,
                    beam_size=beam_size, candi_num=candi_num,
                )
            elif wp_mode == 'greedy':
                result=wm_wiper.wm_wipe_greedy(
                    sentence, 
                    simi_num_for_token=10,
                    # ppl_model_id=model_name,
                    max_edit_rate=max_edit_rate,
                )

            # (new_sentence, wipe_success, raw_green_num, new_green_num, z_threshold_num, new_ppl.item(),raw_ppl.item())
            raw_green_num+=result[2] 
            new_green_num+=result[3] 
            if result[1]==1:
                success_num+=1
                cos_simi+=result[5]
                edit_dist+=result[6]
            if result[1]==-1:
                sen_num-=1
            if len(result)<6:
                continue
            result_dict.append({
                'raw_sentence':sentence,
                'new_sentence':result[0],
                'wipe_success':result[1], 
                'raw_green_num':result[2], 
                'new_green_num':result[3], 
                'z_threshold_num':result[4], 
                'new_ppl':result[5],
                'raw_ppl':result[6],
                'raw_true_key':result[7],
            })
        # print(sen_num, success_num, success_num/sen_num, cos_simi/sen_num, edit_dist/sen_num)
        log_print(to_string(('sen_num', sen_num)))
        log_print(to_string(('success_num', success_num)))
        log_print(to_string(('success_rate', success_num/sen_num)))
        log_print(to_string(('new_ppl', cos_simi/sen_num)))
        log_print(to_string(('raw_ppl', edit_dist/sen_num)))
        log_print(to_string(('raw_green_num', raw_green_num/sen_num)))
        log_print(to_string(('new_green_num', new_green_num/sen_num)))
        
        save_json(
            result_dict,
            output_file_path
        )


