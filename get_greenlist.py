
import torch
from utli import save_json
import os

def get_greenlist_filename(key_token_list, gamma, model_name):
    key_token_str_list=[
        str(key_token)
        for key_token in key_token_list
    ]
    return 'greenlist_'+model_name.replace('/','_')+'_'+'_'.join(key_token_str_list)+'_'+str(gamma)+'.json'

def get_greenlist(vocab_size, gamma, key_token = 123, secret_key = 15485863, device='cuda:0'):
    rng=torch.Generator(device=device)
    rng.manual_seed(secret_key * key_token)

    greenlist_size = int(vocab_size * gamma)
    vocab_permutation = torch.randperm(vocab_size, device=device, generator=rng)
    greenlist_ids = vocab_permutation[:greenlist_size]
    if device !='cpu':
        greenlist_ids=greenlist_ids.cpu()
    return greenlist_ids.tolist()

if __name__=='__main__':
    key_token_list=[123]
    gamma=0.25
    model_name='facebook/opt-1.3b'
    vocab_size=50265
    for (model_name, vocab_size) in[('facebook/opt-1.3b', 50265), ('../model/llama-2-7b',32000)]:
        for gamma in [0.25,0.5]:
            green_list_dict={
                key_token: get_greenlist(vocab_size=vocab_size, gamma=gamma, device='cuda:0', key_token=key_token)
                for key_token in key_token_list
            }
            filename=get_greenlist_filename(key_token_list, gamma, model_name)
            file_path=os.path.join('saved_data', filename)
            save_json(data=green_list_dict, file_path=file_path)
    
