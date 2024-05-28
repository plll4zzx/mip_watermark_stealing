from utli import load_json
from model_inversion_config import config, config_sen, config_pro, config_sta_max_num
import os
import numpy as np

if __name__=='__main__':
    device = 'cuda:0'
    model_name = 'facebook/opt-1.3b'#'../model/llama-2-7b'#

    expect_green_size=1
    z_threshold=4

    wm_seed, nl_seed=123, 456
    rand_num = 0
    
    query_flag, gamma_flag, oracle_flag, naive_flag=True, True, False, True #vanilla
    # query_flag, gamma_flag, oracle_flag, naive_flag=True, True, True, False #oracle
    # query_flag, gamma_flag, oracle_flag, naive_flag=False, False, False, False 
    # query_flag, gamma_flag, oracle_flag, naive_flag=True, True, False, False#pro

    wp_mode='greedy'
    attack_type='op'#'sta'#
    token_len_flag='False'#'True'#

    beam_size=3
    candi_num=3

    dir_path='saved_data/res_wp'
    input_type='.json'
    save_path='saved_res'
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
    random_seed=456

    with_watermark_list_len=1000
    max_edit_rate=1
    perb_rate=0

    
    print(model_name, wp_mode, attack_type, 'token_len_flag=', token_len_flag)
    print('gamma','delta','data_num',end=' ')
    for key in range(key_num):
        print('wipe_success','raw_green_num','new_green_num','new_ppl','raw_ppl', end=' ')
    print()
    
    for (gamma, delta) in [
        (0.25, 2),
        (0.25, 4),
        (0.5, 2),
        (0.5, 4),
    ]:
        # for data_num in [3000, 6000]:#100,500,]:#
        for data_num in [2000, 5000, 10000, 20000]:#100,500,]:#
        # for (data_num, perb_rate) in [ 
        #     (5000,0.1),(5000,0.3),
        #     (10000,0.5),(10000,0.7),
        # ]:
            
            if query_flag==True and gamma_flag==True and oracle_flag==False and naive_flag==False and key_num==1:
                (wm_bound, nl_bound, sentence_up_num, sentence_down_num) = config_pro[model_name][(gamma, delta)][data_num]
            # elif perb_rate>0:
            #     (wm_bound, nl_bound, sentence_up_num, sentence_down_num) = config_conf[model_name][(gamma, delta)][data_num]
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
                
            output_file_path=os.path.join(
                dir_path,
                '_'.join([
                    'wp', attack_type, wp_mode, 
                    str(beam_size), str(candi_num),str(with_watermark_list_len),
                    token_len_flag,
                    # str(max_edit_rate),
                    token_color_file_name,
                ])
            )
            
            if os.path.isfile(output_file_path)==False:
                continue
            
            wp_res_list=load_json(output_file_path)

            key_list=[]
            for wp_res in wp_res_list:
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