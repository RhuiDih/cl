import json,copy,sys,pickle
import pandas as pd

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import torch
from torch import nn
from torch.utils.data import Dataset

from datasets import load_dataset

import transformers
from transformers import Trainer,TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.training_args import OptimizerNames
from transformers.utils import is_sagemaker_mp_enabled

from peft import LoraConfig, TaskType, get_peft_model, PeftModelForCausalLM


DEFAULT_PAD_TOKEN = "[PAD]"

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_merged_model(model_id,peft_checkpoint_name):
    model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype='auto')

    tokenizer=AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
        model.resize_token_embeddings(len(tokenizer))
    
    model=PeftModelForCausalLM.from_pretrained(model,peft_checkpoint_name)
    return model.merge_and_unload()
        
def main():
    model_id='outputs/llama3-8b'

    models={}
    li_data_id=['cstance4','fomc4','meeting4','py150-4','science4','20minuten4','metamath4']
    for data_id in li_data_id:
        peft_checkpoint_name='outputs/llama3-8b-fl_{}_round1/checkpoint-{}/'.format(data_id,18516 if data_id=='metamath4' else 780)
        models[data_id]=get_merged_model(model_id,peft_checkpoint_name)
        
    model_merged=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype='auto')
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    model_merged.resize_token_embeddings(len(tokenizer))

    for j,data_id in enumerate(li_data_id):
        if j==0:
            continue
        dict_params=[]
        for i in range(j+1):
            dict_params.append( dict(models[li_data_id[i]].named_parameters()) )
        
        dict_param_merged = dict(model_merged.named_parameters())
        with torch.no_grad():
            for name in list(dict_param_merged.keys()):
                dict_param_merged[name].data.copy_(  torch.stack([d[name].data for d in dict_params]).mean(0) )
        model_merged.load_state_dict(dict_param_merged)

        output_dir='outputs/merged-llama3-8b-fl_round1_{}'.format( '+'.join([ li_data_id[i] for i in range(j+1) ]) )
        model_merged.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print('saved',output_dir)
        

if __name__=='__main__':
    main()
    
