import time
from typing import Dict
import transformers
from transformers import GenerationConfig
import torch

def main():
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"

    def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
    ):
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
            
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN), tokenizer=tokenizer, model=model)
    tokenizer.add_special_tokens(
        {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
        }
    )
    input_sents = 'I' # actually two tokens inputs with bos_token
    inputs = tokenizer(input_sents, return_tensors="pt", truncation=True, max_length=2048)
    print(f'Length of inputs: {inputs["input_ids"].shape}')
    print("\n")

    inputs["input_ids"] = inputs["input_ids"].repeat(BATCH_SIZE, 1)
    print(inputs["input_ids"].shape)
    

    print('warmup start')
    _ = model.generate(input_ids=inputs["input_ids"].cuda(),
                             generation_config=generation_config,
                             return_dict_in_generate=True,
                             output_scores=True
                             )
    print('warmup finish')

    start = time.time()
    for _ in range(TIMES):
        _ = model.generate(input_ids=inputs["input_ids"].cuda(),
                                generation_config=generation_config,
                                return_dict_in_generate=True,
                                output_scores=True
                                )
    end = time.time()
    cost = (end - start) / TIMES
    print("\nEnd-to-End Time: {}\tSpeed: {}\n".format(cost, MAX_NEW_TOKENS / cost))
    print('finist testing speed')

if __name__ == '__main__':
    TIMES = 2 #set generation repeat times for speed test
    BATCH_SIZE = 6 #batch size for generation
    MAX_NEW_TOKENS= 2048
    model_name_or_path = 'modeling/dense_gau_retnet_350m'
    inference_dtype = torch.float16

    generation_config = GenerationConfig(
        temperature=0.1,
        do_sample = False,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype = inference_dtype)
    print(model)
    model = model.half()
    model.cuda()
    model.eval()
    print('model to cuda')
    all_param = 0
    for k, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            print(f'parameter {k}: {num_params}')
            all_param += num_params
    print(f'total parameter : {all_param}')
    print(f"Total model size={all_param / 2 ** 20:.2f}M params")
    main()
