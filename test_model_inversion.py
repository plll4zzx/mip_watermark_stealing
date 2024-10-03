# import pyomo.environ as pyomo
import os
from utli import load_json, save_json
# import numpy as np
from get_greenlist import get_greenlist,get_greenlist_filename
from utli import to_string, print_info, check_dir, str2bool
from greenlist_inversion_plus import GreenlistInversorPlus
from utli import Logger
import datetime
from model_inversion_config import config, config_pro, config_conf
import numpy as np
import argparse



if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='test multi key')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='../model/llama-2-7b')
    parser.add_argument('--dir_path', type=str, default='saved_data')
    parser.add_argument('--save_path', type=str, default='saved_res')
    parser.add_argument('--dataset_name', type=str, default='c4_realnewslike')
    
    parser.add_argument('--z_threshold', type=int, default=4)
    parser.add_argument('--query_flag', type=str, default='False')
    parser.add_argument('--gamma_flag', type=str, default='False')
    parser.add_argument('--oracle_flag', type=str, default='False')
    parser.add_argument('--naive_flag', type=str, default='False')
    
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--delta', type=int, default=2)
    parser.add_argument('--wm_data_num', type=int, default=5000)
    parser.add_argument('--nl_data_num', type=int, default=5000)
    parser.add_argument('--TimeLimit', type=int, default=500)
    args = parser.parse_args()

    dir_path=args.dir_path
    save_path=args.save_path
    check_dir(dir_path)
    check_dir(save_path)
    dataset_name=args.dataset_name
    model_name= args.model_name
    z_threshold=args.z_threshold

    query_flag=str2bool(args.query_flag) 
    gamma_flag=str2bool(args.gamma_flag) 
    oracle_flag=str2bool(args.oracle_flag) 
    naive_flag=str2bool(args.naive_flag )
    
    # query_flag, gamma_flag, oracle_flag, naive_flag=True, True, False, True #vanilla-as1
    # query_flag, gamma_flag, oracle_flag, naive_flag=True, True, True, False #oracle-as1
    # query_flag, gamma_flag, oracle_flag, naive_flag=True, True, False, False#pro-as1
    # query_flag, gamma_flag, oracle_flag, naive_flag=False, False, False, False #pro-as2
    
    wm_level = 'model'
    input_type='.json'
    wm_type='o'
    dataset_num=str(30000)
    key_token_list=[123]
    key_num=1
    wm_seed, nl_seed=123, 456
    
    rand_num=0
    expect_green_size=1
    perb_rate=0
    TimeLimit=500
    min_max_green_bound=-1
    
    check_dir('log')
    logger=Logger(
        'log/GreenlistInversor-'+'-'.join([
            model_name.replace('/','_'),
            str(query_flag), str(gamma_flag), str(oracle_flag),str(naive_flag),
        ])+'-'+str(datetime.datetime.now().date())+'.log',
        level='debug', 
        screen=False
    )
    
    gamma = args.gamma
    delta = args.delta
    wm_data_num = args.wm_data_num
    nl_data_num = args.nl_data_num
    TimeLimit = args.TimeLimit
            
    if query_flag or naive_flag or oracle_flag:
        sentence_up_num, sentence_down_num=1,1
    else:
        (_, _, sentence_up_num, sentence_down_num) = config_pro[model_name][(gamma, delta)][wm_data_num]

    if naive_flag or oracle_flag:
        wm_bound, nl_bound=1,1
    else:
        (wm_bound, nl_bound, _, _) = config_pro[model_name][(gamma, delta)][wm_data_num]
    if perb_rate>0:
        if query_flag or naive_flag or oracle_flag:
            sentence_up_num, sentence_down_num=1,0.8
        wm_data_num+=perb_rate
        nl_data_num+=perb_rate

        (sentence_up_num, sentence_down_num, wm_bound, nl_bound, min_max_green_bound, expect_green_size) = config_conf[model_name][(gamma, delta, perb_rate)]
        print(sentence_up_num, sentence_down_num, wm_bound, nl_bound, min_max_green_bound, expect_green_size)

    output_file_name='_'.join([
        wm_level, wm_type, dataset_name, model_name.replace('/','_'), dataset_num,
        str(key_token_list), str(key_num), 
        str(wm_seed), str(nl_seed), str(rand_num), 
        str(wm_data_num), str(nl_data_num),
        str(gamma), str(delta),
        str(expect_green_size),
        str(z_threshold), 
        str(query_flag), str(gamma_flag), str(oracle_flag),
        str(sentence_up_num), str(sentence_down_num),
        str(wm_bound), str(nl_bound), 
        str(naive_flag),
        'token_color.json'
    ])
    output_file_path=os.path.join(save_path, output_file_name)
    print()

    green_inversor = GreenlistInversorPlus(tokenizer_tag=model_name, logger=logger)
    
    green_inversor.log_info(model_name)
    green_inversor.log_info(to_string((wm_level, rand_num, gamma, delta)))
    green_inversor.log_info(to_string(('native_flag', naive_flag)))
    
    green_list_name= get_greenlist_filename(key_token_list, gamma, model_name)#+'_'+wm_level +delta
    green_inversor.load_true_green(os.path.join(dir_path, green_list_name))

    file_name='_'.join((
        dataset_name, 
        model_name.replace('/','_'),
        wm_level,wm_type,
        dataset_num,
        str(key_num),
        str(gamma), str(delta)
    ))
    
    green_inversor.set_dataset(
        dir_path=dir_path, file_name=file_name, input_type=input_type,
        wm_data_num=wm_data_num, nl_data_num=nl_data_num,
        gamma=gamma,
        z_threshold=z_threshold,
        key_num=key_num, rand_num=rand_num,
        wm_seed=wm_seed, nl_seed=nl_seed,
        true_key_list=key_token_list,
        query_flag=query_flag
    )

    green_inversor.set_model(
        sentence_up_num=sentence_up_num, sentence_down_num=sentence_down_num,
        perb_rate=perb_rate,
        expect_green_size=expect_green_size,
        key_list=key_token_list,
        token_len=2, 
        gamma_flag=gamma_flag, oracle_flag=oracle_flag,
        reals_flag=True,
        min_max_green_bound=min_max_green_bound,
    )
    green_inversor.set_solver()

    flag=True
    if naive_flag==False and oracle_flag==False:
        lock_sentence=True
        if perb_rate>=0.5:
            lock_sentence=False
            
            green_inversor.solve_sum_whether_sentence(
                TimeLimit=TimeLimit, MIPGap=1e-4,
                wm_bound=wm_bound, nl_bound=nl_bound,
            )
            green_inversor.save_solution(save_path, output_file_name)
        if perb_rate<0.5:
            flag=green_inversor.solve_wm_nl_sum_green(
                TimeLimit=TimeLimit, MIPGap=1e-4, 
                wm_bound=wm_bound, nl_bound=nl_bound,
                lock_sentence=lock_sentence, 
            )
            if flag==False:
                green_inversor.log_info(to_string((model_name, gamma, delta, wm_data_num,)))
            else:
                green_inversor.save_solution(save_path, output_file_name)
    if flag:
        green_inversor.solve_green_num(max_min=False, TimeLimit=TimeLimit, MIPGap=1e-4, MIPGapAbs=2000,)
        green_inversor.save_solution(save_path, output_file_name, )
