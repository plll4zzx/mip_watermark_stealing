from transformers import OPTForCausalLM

# model=OPTForCausalLM.from_pretrained("")
# model.generate()

from model_with_watermark import ModelWithWatermark
# from model_with_esswm import ModelWithEswm
from read_data import c4
from tqdm import tqdm
import os
from utli import save_json
# import json
# from Sentence_Embedder import Sentence_Embedder
# from sentence_transformers import SentenceTransformer, util
# import numpy as np

def get_watermark_data(
    c4_dataset, dir_path,
    dataset_name, file_num, file_data_num,
    wm_level, wm_type, model_name, output_dir,
    finit_key_num,
    gamma=0.25, delta=2,
):

    data_num=str(int(file_num*file_data_num))
    print('Dataset imported')
    
    load_fp16=False
    if '1.3' not in model_name:
        load_fp16=True
    
    mww=ModelWithWatermark(model_name, load_fp16=load_fp16)
    mww.set_watermark(
        wm_level=wm_level, 
        finit_key_num=finit_key_num,
        gamma=gamma, 
        delta=delta,
        max_new_tokens=200,
    )

    # counter=0
    # sentence_embedder=Sentence_Embedder()#embedder_name='all-mpnet-base-v2'
    # simi_list=[]
    # simi_list1=[]
    # simi_list2=[]
    data_list=[]

    for idx in tqdm(range(len(c4_dataset.data)), ncols=100, desc='watermark', leave=True):

        mww.generation_seed=123
        
        output1 = mww.generate(c4_dataset.data[idx]['text'])#

        # e_c=sentence_embedder.get_embedding(output1[2])
        # e_w=sentence_embedder.get_embedding(output1[3])
        # simi_score=util.cos_sim(e_c, e_w).item()
        # simi_list.append(simi_score)

        # sample_key2=456
        # mww.generation_seed=456
        # output2 = mww.generate(c4_dataset.data[idx]['text'])#, sample_key2
        if len(output1[2])<100 or len(output1[3])<100:
            continue

        # e0=sentence_embedder.get_embedding(output1[2])
        # e1=sentence_embedder.get_embedding(output2[2])
        # simi_score1=util.cos_sim(e0, e1).item()
        # simi_list1.append(simi_score1)

        # e3=sentence_embedder.get_embedding(output1[3])
        # e4=sentence_embedder.get_embedding(output2[3])
        # simi_score2=util.cos_sim(e3, e4).item()
        # simi_list2.append(simi_score2)

        c4_dataset.data[idx]['redecoded_input']=output1[0]
        c4_dataset.data[idx]['truncation_warning']=output1[1]
        c4_dataset.data[idx]['output_without_watermark']=output1[2]
        c4_dataset.data[idx]['output_with_watermark']=output1[3]
        # c4_dataset.data[idx]['simi_score']=simi_score

        data_list.append(c4_dataset.data[idx])
        # print(idx)
    # print(np.mean(simi_list))
    # print(np.mean(simi_list2))
    print('data_list len', len(data_list))
    output_path=os.path.join(
        output_dir, 
        '_'.join((
            dataset_name, 
            model_name.replace('/','_'),
            wm_level,wm_type,
            data_num,
            str(finit_key_num),
            str(gamma), str(delta)
        ))+'.json'
    )
    save_json(data_list, output_path)

if __name__=='__main__':
    dir_path='../../dataset/c4/realnewslike'
    dataset_name='c4_realnewslike'
    file_num=int(3)
    file_data_num=1
    
    output_dir='saved_data'

    wm_type='o'

    c4_dataset=c4(dir_path=dir_path, file_num=file_num, file_data_num=file_data_num)
    c4_dataset.load_data()

    if wm_type=='es':
        wm_level_list=['sentence']
    else:
        wm_level_list=['model_simi']
        # wm_level_list=['sentence']
        # wm_level_list = ['model', 'sentence', 'token', 'sentence_fi']
    
    model_name_list= ['facebook/opt-1.3b','../model/llama-2-7b', ]#'openlm-research/open_llama_7b_v2', 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-1.3b', 'microsoft/phi-2', 
    finit_key_num=3
    # os.system('set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32')
    for model_name in model_name_list:
        for wm_level in wm_level_list:
            for gamma in [0.25, 0.5, ]:#5,7,9,11
                for delta in [2, 4, ]:#5,7,9,11
                    print(model_name, wm_level, wm_type, gamma, delta)
                    get_watermark_data(
                        c4_dataset,
                        dir_path=dir_path, 
                        dataset_name=dataset_name, 
                        file_num=file_num, 
                        file_data_num=file_data_num, 
                        wm_level=wm_level,
                        wm_type=wm_type, 
                        model_name=model_name, 
                        output_dir=output_dir,
                        finit_key_num=finit_key_num,
                        gamma=gamma, delta=delta,
                    )

    # get_watermark_data(
    #     c4_dataset,
    #     dir_path=dir_path, dataset_name=dataset_name, 
    #     file_num=file_num, file_data_num=file_data_num, 
    #     wm_level='token'+wm_type, model_name=model_name, output_dir=output_dir
    # )

    # get_watermark_data(
    #     c4_dataset,
    #     dir_path=dir_path, dataset_name=dataset_name, 
    #     file_num=file_num, file_data_num=file_data_num, 
    #     wm_level='sentence'+wm_type, model_name=model_name, output_dir=output_dir
    # )