import json,copy,sys,pickle,os,gzip,yaml
import numpy as np
import pandas as pd
from datetime import datetime

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import torch
from torch import nn
from torch.utils.data import Dataset

from datasets import load_dataset

import transformers
from transformers import Trainer,TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithFlattening
from transformers.training_args import OptimizerNames
from transformers.utils import is_sagemaker_mp_enabled
from transformers.data.data_collator import PreTrainedTokenizerBase,PaddingStrategy

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

        
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        list_data_dict=dataset
        #list_data_dict = random.sample(dataset,  len(dataset))
        #list_data_dict = list_data_dict[:data_args.data_length]

        # logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # print(list_data_dict[0])
        if 'instruction' in list_data_dict[0]:
            pass
        else:
            def get_input(query):
                if query.find('\n') == -1:
                    return ''
                return '\n'.join(query.split('\n')[1:])
            list_data_dict = [{'instruction':data['query'].split('\n')[0], 'input':get_input(data['query']), 'output':data['response']} for data in list_data_dict]
        # import ipdb; ipdb.set_trace()
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])
    



class DuplicateDataset(Dataset):
    def __init__(self,dataset,multiplier):
        self.dataset=dataset
        self.multiplier=multiplier
        
    def __getitem__(self,index):
        return self.dataset[index%len(self.dataset)]
    
    def __len__(self):
        return self.multiplier*len(self.dataset)


class JointDataset(Dataset):
    def __init__(self,li_dataset):
        self.li_dataset=li_dataset
        self.lengths=np.cumsum([len(d) for d in li_dataset])

    def __getitem__(self,index):
        offset=0
        for i,n in enumerate(self.lengths):
            if index<n:
                return self.li_dataset[i][index-offset]
            offset=n
        
    def __len__(self):
        return self.lengths[-1]
    
    
class DataCollatorFlat2(DataCollatorWithFlattening):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer=tokenizer

    def __call__(self, instances):
        li_tok=[]
        for d in instances:
            tok1=self.tokenizer(d['input_ids'])['input_ids']
            tok2=self.tokenizer(d['labels'],add_special_tokens=False)['input_ids']
            tok_all=tok1+tok2
            if len(tok_all)>self.tokenizer.model_max_length:
                if len(tok2)>self.tokenizer.model_max_length//2:
                    n1=int(len(tok1)/len(tok_all)*self.tokenizer.model_max_length)
                    n2=self.tokenizer.model_max_length-n1
                    tok1=tok1[:n1//2]+tok1[-(n1-n1//2):]
                    tok2=tok2[:n2]
                else:
                    n1=self.tokenizer.model_max_length-len(tok2)
                    tok1=tok1[:n1//2]+tok1[-(n1-n1//2):]
                tok_all=tok1+tok2
            li_tok.append({'input_ids':tok_all,'labels':[-100]*len(tok1)+tok2})
        return super().__call__(li_tok)

    


class MyTrainer(Trainer):
    def __init__(self, olora_lambda,olora_targets, **kwargs):
        super().__init__(**kwargs)
        
        self.olora_lambda=olora_lambda
        self.olora_targets=olora_targets
            
        
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            raise 

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_xpu_available():
                torch.xpu.empty_cache()
            elif is_mlu_available():
                torch.mlu.empty_cache()
            elif is_npu_available():
                torch.npu.empty_cache()
            elif is_torch_version(">=", "2.0") and is_mps_available():
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            
            
        #### Orthogonal loss ####
        #print('loss.dtype',loss.dtype)
        if self.olora_lambda>0 and self.olora_targets is not None:
            orthogonal_loss = 0.
            for name, param in model.named_parameters():
                if "lora_A" in name:
                    olora_key=name.split('lora_A')[0]
                    if olora_key not in self.olora_targets: ## wrapped
                        olora_key=olora_key.split('.',1)[1]
                    param_ = self.olora_targets[olora_key].to(param.device)
                    orthogonal_loss += torch.abs(torch.mm(param, param_.T)).sum() # [r * dim] * [dim * r]
            loss = loss + orthogonal_loss * self.olora_lambda
        #### end Orthogonal loss ###

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps    


def trace_pretty(d,tokenizer,add_eos=True):
    return {'input_ids':d['prompt'],
            'labels': '{}{}'.format(d['answer'],tokenizer.eos_token if add_eos else '')}



def load_config(conf_fname):
    if not os.path.exists(conf_fname):
        raise
    with open(conf_fname,'r') as fp:
        config=yaml.safe_load(fp)
    return config

def main():
    config=load_config(sys.argv[1])
    dataset_name=config['dataset_name']
    model_id=config['model_id']
    output_id=config['output_id']

    print('Model',flush=True)

    ## Use Flash Attention
    model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype='auto',
                                               attn_implementation='flash_attention_2')

    model_max_length=config['model_max_length']
    
    print('Tokenizer',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(model_id,
                        model_max_length=model_max_length,
                        padding_side="right",
                        use_fast=False)
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    if len(config['lora_target'])<=0:
        print('Full finetuning',flush=True)
        lora_config = None
    else:
        print('LoRA',flush=True)
        lora_config = LoraConfig(
            r=config['lora_r'],
            target_modules=config['lora_target'],
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout']
        )

    olora_lambda=0
    olora_targets=None

    if lora_config is not None:
        print('Peft Model',flush=True)
        model=get_peft_model(model,lora_config)
    
        
    print('Dataset',flush=True)
    if dataset_name=='metamath4':
        dataset=load_dataset('meta-math/MetaMathQA')
        dataset = SupervisedDataset(tokenizer=tokenizer, dataset=dataset['train'])
        
        data_module={'train_dataset':dataset,'eval_dataset':None,
                     'data_collator':DataCollatorFlat2(tokenizer=tokenizer) }
        batch_size=4
        grad_step=4

    elif dataset_name=='trace6+metamath4':
        li_dataset=[]
        dataset=load_dataset('meta-math/MetaMathQA')
        li_dataset.append( SupervisedDataset(tokenizer=tokenizer, dataset=dataset['train']) )
        for name in ['ScienceQA','FOMC','C-STANCE','20Minuten','Py150','MeetingBank']:
            with gzip.open('{}_split1234.pickle.gz'.format(name),'rb') as fp:
                dataset=pickle.load(fp)
            dataset=[trace_pretty(d,tokenizer,add_eos=True) for d in dataset]
            li_dataset.append( DuplicateDataset(dataset,multiplier=79) )
        dataset=JointDataset(li_dataset)
        
        data_module={'train_dataset':dataset,'eval_dataset':None,
                     'data_collator':DataCollatorFlat2(tokenizer=tokenizer) }
        batch_size=2
        grad_step=8

    elif dataset_name=='trace6nat+metamath4':
        li_dataset=[]
        dataset=load_dataset('meta-math/MetaMathQA')
        li_dataset.append( SupervisedDataset(tokenizer=tokenizer, dataset=dataset['train']) )
        for name in ['ScienceQA','FOMC','C-STANCE','20Minuten','Py150','MeetingBank']:
            with gzip.open('{}_split1234.pickle.gz'.format(name),'rb') as fp:
                dataset=pickle.load(fp)
            dataset=[trace_pretty(d,tokenizer,add_eos=True) for d in dataset]
            li_dataset.append( dataset )
        dataset=JointDataset(li_dataset)
        
        data_module={'train_dataset':dataset,'eval_dataset':None,
                     'data_collator':DataCollatorFlat2(tokenizer=tokenizer) }
        batch_size=2
        grad_step=8

    elif dataset_name=='science4':
        with gzip.open('ScienceQA_split1234.pickle.gz','rb') as fp:
            dataset=pickle.load(fp)
        dataset=[trace_pretty(d,tokenizer,add_eos=True) for d in dataset]
        
        data_module={'train_dataset':dataset,'eval_dataset':None,
                     'data_collator':DataCollatorFlat2(tokenizer=tokenizer) }
        batch_size=4
        grad_step=4

    elif dataset_name=='fomc4':
        with gzip.open('FOMC_split1234.pickle.gz','rb') as fp:
            dataset=pickle.load(fp)
        dataset=[trace_pretty(d,tokenizer,add_eos=True) for d in dataset]
        
        data_module={'train_dataset':dataset,'eval_dataset':None,
                     'data_collator':DataCollatorFlat2(tokenizer=tokenizer) }
        batch_size=4
        grad_step=4

    elif dataset_name=='cstance4':
        with gzip.open('C-STANCE_split1234.pickle.gz','rb') as fp:
            dataset=pickle.load(fp)
        dataset=[trace_pretty(d,tokenizer,add_eos=True) for d in dataset]
        
        data_module={'train_dataset':dataset,'eval_dataset':None,
                     'data_collator':DataCollatorFlat2(tokenizer=tokenizer) }
        batch_size=4
        grad_step=4

    elif dataset_name=='20minuten4':
        with gzip.open('20Minuten_split1234.pickle.gz','rb') as fp:
            dataset=pickle.load(fp)
        dataset=[trace_pretty(d,tokenizer,add_eos=True) for d in dataset]
        
        data_module={'train_dataset':dataset,'eval_dataset':None,
                     'data_collator':DataCollatorFlat2(tokenizer=tokenizer) }
        batch_size=2
        grad_step=8

    elif dataset_name=='py150-4':
        with gzip.open('Py150_split1234.pickle.gz','rb') as fp:
            dataset=pickle.load(fp)
        dataset=[trace_pretty(d,tokenizer,add_eos=True) for d in dataset]
        
        data_module={'train_dataset':dataset,'eval_dataset':None,
                     'data_collator':DataCollatorFlat2(tokenizer=tokenizer) }
        batch_size=2
        grad_step=8

    elif dataset_name=='meeting4':
        with gzip.open('MeetingBank_split1234.pickle.gz','rb') as fp:
            dataset=pickle.load(fp)
        dataset=[trace_pretty(d,tokenizer,add_eos=True) for d in dataset]
        
        data_module={'train_dataset':dataset,'eval_dataset':None,
                     'data_collator':DataCollatorFlat2(tokenizer=tokenizer) }
        batch_size=2
        grad_step=8
        
    else:
        raise

    output_dir='outputs/{}_{}_round1'.format(output_id,dataset_name)
        
    print('Trainer',flush=True)
    save_strategy='steps' if 'save_steps' in config else 'epoch'
    save_steps=config.get('save_steps',1000)
    training_args = TrainingArguments(output_dir=output_dir,num_train_epochs=config['num_train_epochs'],
                                      per_device_train_batch_size=batch_size,gradient_accumulation_steps=grad_step,
                                      eval_strategy='no',save_strategy=save_strategy,save_steps=save_steps,
                                      learning_rate=2e-5,lr_scheduler_type='constant')

    trainer = MyTrainer(olora_lambda=olora_lambda,olora_targets=olora_targets,
                        model=model, tokenizer=tokenizer, args=training_args, **data_module)

    print('Accel',trainer.accelerator.num_processes,trainer.accelerator.local_process_index,
          trainer.accelerator.distributed_type,flush=True)
    local_idx=trainer.accelerator.local_process_index
    print('Train',flush=True)
    trainer.train()#resume_from_checkpoint=True)

    ### save merged model
    if trainer.accelerator.is_main_process and lora_config is not None:
        print('Merge and save')
        #model_unwrapped=trainer.accelerator.unwrap_model(trainer.model_wrapped)
        
        olora_targets={}
        olora_targets_B={}
        for n,p in model.named_parameters():
            if 'lora_A' in n:
                olora_targets[n.split('lora_A')[0]] = p.detach().cpu()
            if 'lora_B' in n:
                olora_targets_B[n.split('lora_B')[0]] = p.detach().cpu()
        model=model.merge_and_unload()
        output_dir=os.path.join(output_dir,'final-merged-e{}'.format(config['num_train_epochs']))
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        with open('{}/olora_targets.pickle'.format(output_dir),'wb') as fp:
            pickle.dump(olora_targets,fp)
            pickle.dump(olora_targets_B,fp)
        
                          
if __name__=='__main__':
    print('Main',flush=True)
    main()
    
    
