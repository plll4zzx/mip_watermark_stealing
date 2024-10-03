from utli import load_json, check_dir, str2bool
from model_inversion_config import config, config_sen, config_pro, config_sta_max_num
import os
import numpy as np
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='test WM_Wiper')
    parser.add_argument('--device', type=int, default=0)
    
    parser.add_argument('--model_name', type=str, default='../model/llama-2-7b', help='../model/llama-2-7b OR facebook/opt-1.3b')#../model/llama-2-7b
    parser.add_argument('--dir_path', type=str, default='saved_data/res_wp')
    parser.add_argument('--save_path', type=str, default='saved_res')
    parser.add_argument('--dataset_name', type=str, default='c4_realnewslike')
    parser.add_argument('--wm_level', type=str, default='model', help='model OR sentence_fi')

    parser.add_argument('--z_threshold', type=int, default='4')
    parser.add_argument('--query_flag', type=str, default='True')
    parser.add_argument('--gamma_flag', type=str, default='True')
    parser.add_argument('--oracle_flag', type=str, default='False')
    parser.add_argument('--naive_flag', type=str, default='True')

    parser.add_argument('--attack_type', type=str, default='op', help='op OR sta')
    parser.add_argument('--wp_mode', type=str, default='greedy', help='greedy OR gumbel')
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--candi_num', type=int, default=1)
    args = parser.parse_args()
    
    device = 'cuda:'+str(args.device)
    model_name = args.model_name
    z_threshold = args.z_threshold
    dataset_name=args.dataset_name
    
    query_flag=str2bool(args.query_flag) 
    gamma_flag=str2bool(args.gamma_flag) 
    oracle_flag=str2bool(args.oracle_flag) 
    naive_flag=str2bool(args.naive_flag )

    wp_mode=args.wp_mode
    attack_type=args.attack_type

    beam_size=args.beam_size
    candi_num=args.candi_num

    dir_path=args.dir_path
    save_path=args.save_path
    check_dir(dir_path)
    check_dir(save_path)
    wm_level = args.wm_level

    input_type='.json'
    wm_type='o'
    dataset_num=str(30000)
    random_seed=456
    if wm_level=='model':
        key_token_list=[123]
        key_num=1
        gamma_delta=[(0.25, 2),(0.25, 4),(0.5, 2),(0.5, 4)]
        data_num_list=[2000, 5000, 10000, 20000]
    else:
        key_token_list=[1,2,3]
        key_num=3
        gamma_delta=[(0.25, 2),(0.5, 2),]
        data_num_list=[3000, 6000]


    with_watermark_list_len=1000
    max_edit_rate=1
    perb_rate=0

    expect_green_size=1
    wm_seed, nl_seed=123, 456
    rand_num = 0
    token_len_flag='False'

    
    print(model_name, wp_mode, attack_type, 'token_len_flag=', token_len_flag)
    print('gamma','delta','data_num',end=' ')
    for key in range(key_num):
        print('wipe_success','raw_green_num','new_green_num','new_ppl','raw_ppl', end=' ')
    print()
    
    for (gamma, delta) in gamma_delta:
        for data_num in data_num_list:
            if query_flag==True and gamma_flag==True and oracle_flag==False and naive_flag==False and key_num==1:
                (wm_bound, nl_bound, sentence_up_num, sentence_down_num) = config_pro[model_name][(gamma, delta)][data_num]
            elif key_num>1:
                (sentence_up_num, sentence_down_num, wm_bound, nl_bound, min_max_green_bound, expect_green_size, min_green_num) = config_sen[model_name][(gamma, delta, data_num)]
            else:
                (wm_bound, nl_bound, sentence_up_num, sentence_down_num) = config[model_name][(gamma, delta)][data_num]
                if model_name=='facebook/opt-1.3b' and gamma==0.25 and query_flag==True and gamma_flag==True and oracle_flag==True and naive_flag==False:
                    nl_bound=1
                elif model_name=='facebook/opt-1.3b' and gamma==0.5 and query_flag==True and gamma_flag==True and oracle_flag==True and naive_flag==False:
                    wm_bound=0.95
                elif model_name=='facebook/opt-1.3b' and gamma==0.5 and query_flag==True and gamma_flag==True and oracle_flag==False and naive_flag==True:
                    wm_bound=0.95
            if query_flag or naive_flag or oracle_flag:
                sentence_up_num, sentence_down_num=1,1
            if naive_flag or oracle_flag:
                wm_bound, nl_bound=1,1
            if perb_rate>0:
                data_num+=perb_rate
                sentence_up_num=np.round(sentence_up_num-perb_rate,2)
                if gamma==0.25 and data_num>5001:
                    perb_rate+=0.1
                elif gamma==0.5 and data_num>5001:
                    perb_rate+=0.2
                sentence_down_num=np.round(sentence_down_num-perb_rate,2)
                if data_num>5001:
                    wm_bound, nl_bound=1,1

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
                token_color_file_name='_'.join([#
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
                if wm_level=='sentence_fi':
                    token_color_file_name=str(15)+'_'+token_color_file_name
                
            output_file_path=os.path.join(
                dir_path,
                '_'.join([
                    'wp', attack_type, wp_mode, 
                    str(beam_size), str(candi_num),str(with_watermark_list_len),
                    token_len_flag,
                    token_color_file_name,
                ])
            )
            
            if os.path.isfile(output_file_path)==False:
                continue
            
            wp_res_list=load_json(output_file_path)

            key_list=[]
            for wp_res in wp_res_list:
                if 'raw_true_key' not in wp_res:
                    wp_res['raw_true_key']=key_token_list[0]
                if wp_res['raw_true_key'] in key_list:
                    continue
                key_list.append(wp_res['raw_true_key'])

            wipe_success={
                key:np.mean([
                    wp_res['wipe_success']
                    for wp_res in wp_res_list
                    if wp_res['raw_true_key']==key
                ])
                for key in key_list
            }
            raw_green_num={
                key:np.mean([
                    wp_res['raw_green_num']
                    for wp_res in wp_res_list
                    if wp_res['raw_true_key']==key
                ])
                for key in key_list
            }
            new_green_num={
                key:np.mean([
                    wp_res['new_green_num']
                    for wp_res in wp_res_list
                    if wp_res['raw_true_key']==key
                ])
                for key in key_list
            }
            raw_ppl={
                key:np.mean([
                    wp_res['raw_ppl']
                    for wp_res in wp_res_list
                    if wp_res['raw_true_key']==key
                ])
                for key in key_list
            }
            new_ppl={
                key:np.mean([
                    wp_res['new_ppl']
                    for wp_res in wp_res_list
                    if wp_res['raw_true_key']==key
                ])
                for key in key_list
            }

            
            print(gamma, delta, data_num, end=' ')
            for key in key_list:
                print(wipe_success[key], raw_green_num[key], new_green_num[key], new_ppl[key], raw_ppl[key], end=' ')
            print()
            # print(gamma, delta, data_num, wipe_success, raw_green_num, new_green_num, new_ppl, raw_ppl)