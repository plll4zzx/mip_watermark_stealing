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
    mww,
    c4_dataset, dir_path,
    dataset_name, file_num, file_data_num,
    wm_level, wm_type, model_name, output_dir,
    finit_key_num,
    gamma=0.25, delta=2,
):

    data_num=str(int(file_num*file_data_num))
    print('Dataset imported')
    
    # load_fp16=False
    # if '1.3' not in model_name:
    #     load_fp16=True
    
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

        if len(output1[2])<100 or len(output1[3])<100:
            continue

        c4_dataset.data[idx]['redecoded_input']=output1[0]
        c4_dataset.data[idx]['truncation_warning']=output1[1]
        c4_dataset.data[idx]['output_without_watermark']=output1[2]
        c4_dataset.data[idx]['output_with_watermark']=output1[3]
        # c4_dataset.data[idx]['simi_score']=simi_score

        data_list.append(c4_dataset.data[idx])
        
        
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

    wm_level_list = ['model', 'sentence_fi']
    
    model_name_list= ['../model/llama-2-7b', 'facebook/opt-1.3b',]
    finit_key_num=3
    # os.system('set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32')
    for model_name in model_name_list:
        mww=ModelWithWatermark(model_name)
        for wm_level in wm_level_list:
            for gamma in [0.25, 0.5,]:
                for delta in [2, 4,]:
                    print(model_name, wm_level, wm_type, gamma, delta)
                    get_watermark_data(
                        mww,c4_dataset,
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
