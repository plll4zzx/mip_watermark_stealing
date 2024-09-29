# Stealing Watermarks of Large Language Models via Mixed Integer Programming

## File List for Code
* get_watermark_data.py: Grnerate watermark text;
* get_greenlist.py: Collect ground truth green list for each target model;
* test_model_inversion.py: Steal green list of unigram watermark;
* test_sen_inversion.py: Steal green list of multi-key watermark;
* greenlist_inversion_plus.py: Core code for mixed integer programming;
* greenlist_inversion_sta.py: Stealing green list via frequency-based method;
* model_inversion_config.py: Setting of hyperparameters for stealing;
* test_evaluate_color.py: Print the result of stealing;
* test_wm_wiper.py: Watermark removal;
* wm_wiper.py: Core code for watermark removal;
* test_wp_res.py: Print the result of watermark removal.

## Dataset
We randomly sample text from the C4 dataset as prompts to query the watermarked LLM (''model_with_watermark.py'') for generating watermarked text. C4 dataset can be downloaded from https://huggingface.co/datasets/allenai/c4, we recommend to download through git. Due to watermarked sentences being generated from LLM, this process is very slow. GPU and CUDA can be used to accelerate this process. 

## Experimental Settings
The Python version is 3.9; every required repository is listed in ''requirements.txt''. You can execute the following command to install these repositories.
```
    pip install -r requirements.txt
```

Mix integer programming solver Gurobi is essential for this project. Gurobi is the most powerful and widely used solver for mixed integer programming. Gurobi can be installed from its official website. It is worth noting that Gurobi is not an open-source software, but the academic version is available for free https://www.gurobi.com/academia/academic-program-and-licenses/. 

## Target Model
The LLMs used in the experiments are OPT-1.3B and LLaMA-2-7B. OPT and LLaMA can be directly downloaded from Huggingface. If you download LLaMA from its official site, then you need to convert the model weights into huggingface format by using this script https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py. 

## Guideline

### Step-1 Collecting natural and watermarked sentences:
If you would like to collect natural and watermarked sentences, you can run ''get_watermark_data.py''. 
There are 4 parameters in the script: 
(1) ''wm_level'' can be set as ''model'' or ''sentence_fi'', represents unigram-watermark and multi-key watermark, respectively; 
(2) ''model_name'' is used to assign target model; 
(3) ''gamma'' and ''delta'' is the hyperparameters for the watermark scheme; usually, the domain of ''gamma'' and ''delta'' is $[0.1, 0.5]$, $[1, 4]$ respectively. To reproduce the experiment in our paper, ''gamma'' should be set at $\{2, 4\}$, ''delta'' should be set at $\{0.25, 0.5\}$.

### Step-2 Stealing green list:
''test_model_inversion.py'' and ''test_sen_inversion.py'' are used to steal a green list of unigram-watermark and multi-key watermark, respectively. 
''greenlist_inversion_sta.py'' is the code for the frequency-based method. 
After stealing, ground truth green lists are needed to assess the performance of stealing. Ground truth green list can be exported by ''get_greenlist.py''

### Step-3 Removing watermark:
Commands in ''tmp_sh.sh'' are used to remove the watermarks based on the green lists stolen by the Step-2. While removing, the attacker locates which token in the sentence belongs to the stolen green list and then replaces it with a red candidate synonym. Synonyms are found by Gensim \cite{rehurek_lrec}, and synonyms belonging to the stolen green list should be excluded from the candidate set. For ''wm_wiper.py'', there 4 hyperparameters: 
(1) ''attack_type'' can be assigned to ''op'' or ''sta'', representing mixed integer programming based method and frequency-based method, respectively; 
(2) ''wp_mode'' can be assigned to ''greedy'' or ''gumbel''; 
(3) ''beam_size'' can be set to 1 for Greedy Search-based removal; 
(4) ''candi_num'' is the size of candidate synonym tokens.

For example, after setting up the base environment, to reproduce the multi-key results in Table.~7 (in accepted version of paper), you can set ''query_flag'', ''gamma_flag'', ''oracle_flag'', ''naive_flag'' to ''Flase'', ''gamma'' to 0.25, ''delta'' to 2, ''model_name'' to ''llama'', and then run the code in ''test_sen_inversion.py''. 

## License

This project includes code from open-source projects that are licensed under the Apache License 2.0. 

The following projects have been used:
- [lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking)

In accordance with the Apache License 2.0, you can find a copy of the license in the [LICENSE](./LICENSE-2.0.txt) file.

This project adheres to the Apache License 2.0 requirements. If you make modifications to the included code, please ensure that you comply with the license terms.