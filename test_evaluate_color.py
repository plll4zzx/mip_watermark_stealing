
import os
from utli import load_json, save_json, check_dir
from tqdm import tqdm
from utli import Logger,to_string, str2bool
import datetime
from get_greenlist import get_greenlist_filename
import random
from wm_wiper import WM_Wiper
import argparse
from model_inversion_config import config, config_pro, config_sta_max_num, config_conf, config_sen
import numpy as np

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='test_evaluate_color')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='../model/llama-2-7b')
    parser.add_argument('--dir_path', type=str, default='saved_data')
    parser.add_argument('--save_path', type=str, default='saved_res')
    parser.add_argument('--dataset_name', type=str, default='c4_realnewslike')
    
    parser.add_argument('--z_threshold', type=int, default='4')
    parser.add_argument('--query_flag', type=str, default='True')
    parser.add_argument('--gamma_flag', type=str, default='True')
    parser.add_argument('--oracle_flag', type=str, default='False')
    parser.add_argument('--naive_flag', type=str, default='False')
    parser.add_argument('--attack_type', type=str, default='op', help='op OR sta')
    
    args = parser.parse_args()
    
    dir_path=args.dir_path
    save_path=args.save_path
    check_dir(dir_path)
    check_dir(save_path)

    input_type='.json'
    wm_type='o'
    wm_level = 'model'
    dataset_num=str(30000)
    key_token_list=[123]
    key_num=1
    wm_seed, nl_seed=123, 456
    rand_num=0
    expect_green_size=1
    perb_rate=0

    dataset_name=args.dataset_name
    model_name= args.model_name
    z_threshold=args.z_threshold
    attack_type=args.attack_type
    device='cuda:'+str(args.device)

    query_flag=str2bool(args.query_flag) 
    gamma_flag=str2bool(args.gamma_flag) 
    oracle_flag=str2bool(args.oracle_flag) 
    naive_flag=str2bool(args.naive_flag )
    
    print(model_name, attack_type, query_flag, gamma_flag, oracle_flag, naive_flag)
    print('gamma','delta','data_num', end=' ')
    for idx in range(key_num):
        print('$N_g$', '$N_t$',	'Precision', end=' ')
    print()
    
    wm_wiper=WM_Wiper(tokenizer_tag=model_name, device=device)
    for (gamma, delta) in [
        (0.25, 2),
        (0.25, 4),
        (0.5, 2),
        (0.5, 4),
    ]:
        for data_num in [2000,5000,10000,20000]:
            if query_flag==True and gamma_flag==True and oracle_flag==False and naive_flag==False and key_num==1:
                (wm_bound, nl_bound, sentence_up_num, sentence_down_num) = config_pro[model_name][(gamma, delta)][data_num]
            elif key_num>1:
                (sentence_up_num, sentence_down_num, wm_bound, nl_bound, min_max_green_bound, expect_green_size, min_green_num) = config_sen[model_name][(gamma, delta, data_num)]
            else:
                (wm_bound, nl_bound, sentence_up_num, sentence_down_num) = config[model_name][(gamma, delta)][data_num]

            if query_flag or naive_flag or oracle_flag:
                sentence_up_num, sentence_down_num=1,1
            if naive_flag or oracle_flag:
                wm_bound, nl_bound=1,1

            if perb_rate>0:
                data_num+=perb_rate
                (sentence_up_num, sentence_down_num, wm_bound, nl_bound, min_max_green_bound, expect_green_size) = config_conf[model_name][(gamma, delta, perb_rate)]

            wm_wiper.set_para(gamma=gamma, z_threshold=4)
            if wm_level=='model':
                green_list_name= get_greenlist_filename(key_token_list, gamma, model_name)
            else:
                green_list_name= get_greenlist_filename(key_token_list, gamma+delta, model_name+'_'+wm_level)

            wm_wiper.load_true_green(os.path.join(dir_path, green_list_name))

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
                    'token_color.json'
                ])
            else:
                token_color_file_name='_'.join([
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
            
            if key_num>1 and attack_type =='op':
                max_true_green_acc=0
                for idx in range(20):
                    try:
                        wm_wiper.load_token_color(
                            save_path, 
                            str(idx)+'_'+token_color_file_name
                        )
                        tmp_green_num, tmp_true_green_num, tmp_true_green_acc=wm_wiper.evaluate_green_token()
                    except:
                        tmp_green_num, tmp_true_green_num, tmp_true_green_acc=[0]*key_num,[0]*key_num,[0]*key_num
                        # continue
                    if sum(tmp_true_green_acc)>=max_true_green_acc:
                        max_true_green_acc=sum(tmp_true_green_acc)
                        green_num=tmp_green_num
                        true_green_num=tmp_true_green_num
                        true_green_acc=tmp_true_green_acc
            else:
                
                wm_wiper.load_token_color(
                    save_path, 
                    token_color_file_name
                )
                green_num, true_green_num, true_green_acc=wm_wiper.evaluate_green_token()
            
            print(gamma, delta, data_num, end=' ')
            for idx in range(key_num):
                print(green_num[idx], true_green_num[idx], true_green_acc[idx], end=' ')
            print()
