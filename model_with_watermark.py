import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    LogitsProcessorList
)
from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

from functools import partial
from transformers import LlamaTokenizer, LlamaForCausalLM

class ModelWithWatermark:

    def __init__(self, model_name_or_path, load_fp16=True, use_gpu=True, tokenizer_path=None):
        self.is_seq2seq_model = any([(model_type in model_name_or_path) for model_type in ["t5","T0"]])
        self.is_decoder_only_model = any([(model_type in model_name_or_path) for model_type in ["gpt","opt","bloom","phi",'llama']])

        self.model_name_or_path=model_name_or_path
        if 'llama' in model_name_or_path:
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_name_or_path
            )
            self.model=LlamaForCausalLM.from_pretrained(
                model_name_or_path, 
                torch_dtype=torch.float16, 
                # device_map='auto',
            )
        elif self.is_seq2seq_model:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True,
                # device_map="auto",
            )
        elif self.is_decoder_only_model:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            if load_fp16:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,torch_dtype=torch.float16, 
                    # device_map='auto', 
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, trust_remote_code=True,
                    # device_map="auto"
                )
        else:
            raise ValueError(f"Unknown model type: {model_name_or_path}")

        if use_gpu:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            # if load_fp16: 
            #     pass
            # else: 
            self.model = self.model.to(self.device)
        else:
            self.device = "cpu"
        self.model.eval()

        self.wm_level='token'
    
    def set_watermark(
        self, 
        wm_level='token',
        gamma=0.25, 
        delta=2.0, 
        seeding_scheme="simple_1", 
        select_green_tokens=True,
        use_sampling=True,
        sampling_temp=0.7,
        max_new_tokens=200,
        n_beams=1,
        prompt_max_length=None,
        generation_seed=123,
        seed_separately=True,
        finit_key_num=5,
    ):
        
        self.gamma=gamma
        self.delta=delta
        self.seeding_scheme=seeding_scheme
        self.select_green_tokens=select_green_tokens
        self.prompt_max_length=prompt_max_length
        self.generation_seed=generation_seed
        self.max_new_tokens=max_new_tokens
        self.seed_separately=seed_separately

        self.wm_level=wm_level

        if wm_level=='model_simi':
            tokenizer=self.tokenizer
        else:
            tokenizer=None
        self.watermark_processor = WatermarkLogitsProcessor(
            wm_level=wm_level,
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=self.gamma,
            delta=self.delta,
            seeding_scheme=self.seeding_scheme,
            select_green_tokens=self.select_green_tokens,
            finit_key_num=finit_key_num,
            tokenizer=tokenizer,
            model_name=self.model_name_or_path
        )

        gen_kwargs = dict(max_new_tokens=self.max_new_tokens)

        if use_sampling:
            gen_kwargs.update(dict(
                do_sample=True, 
                top_k=0,
                temperature=sampling_temp
            ))
        else:
            gen_kwargs.update(dict(
                num_beams=n_beams
            ))
        if 'llama' in self.model_name_or_path:
            gen_kwargs.update(dict(
                pad_token_id = self.tokenizer.eos_token_id
            ))

        self.generate_without_watermark = partial(
            self.model.generate,
            **gen_kwargs
        )
        self.generate_with_watermark = partial(
            self.model.generate,
            logits_processor=LogitsProcessorList([self.watermark_processor]), 
            **gen_kwargs
        )

    def generate(self, prompt):
        """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
        and generate watermarked text by passing it to the generate method of the model
        as a logits processor. """
        
        # print(f"Generating with {args}")
        
        if self.prompt_max_length:
            pass
        elif hasattr(self.model.config,"max_position_embedding"):
            self.prompt_max_length = self.model.config.max_position_embeddings-self.max_new_tokens
        else:
            self.prompt_max_length = 2048-self.max_new_tokens

        tokd_input = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            add_special_tokens=True, 
            truncation=True, 
            max_length=self.prompt_max_length
        ).to(self.device)
        truncation_warning = True if tokd_input["input_ids"].shape[-1] == self.prompt_max_length else False
        redecoded_input = self.tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

        torch.manual_seed(self.generation_seed)
        output_without_watermark = self.generate_without_watermark(**tokd_input)

        # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
        if self.seed_separately: 
            torch.manual_seed(self.generation_seed)
        output_with_watermark = self.generate_with_watermark(**tokd_input)

        if self.is_decoder_only_model:
            # need to isolate the newly generated tokens
            output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
            output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]

        decoded_output_without_watermark = self.tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
        decoded_output_with_watermark = self.tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

        return (
            redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark, 
            decoded_output_with_watermark,
        ) 