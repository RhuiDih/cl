import json,copy,sys,pickle,os,gzip,yaml
import pandas as pd
from datetime import datetime

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import numpy as np
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
    



class MixtureDataset(torch.utils.data.IterableDataset):
    def __init__(self,li_dataset,li_weights):
        self.li_dataset=li_dataset
        self.li_weights=li_weights
        self.lengths=[round(w*len(d)) for d,w in zip(li_dataset,li_weights)]
        self.total_len=sum(self.lengths)
        self.seed=sum([(i+1)*L for i,L in enumerate(self.lengths)])
        
    def __len__(self):
        return self.total_len

    def __iter__(self):
        randgen=np.random.RandomState(self.seed)
        li=[]
        for L,d in zip(self.lengths,self.li_dataset):
            n=len(d)
            n_copy=L//n+(L%n>0)
            r_idx=randgen.permutation(n_copy*n)
            li.extend([d[i%n] for i in r_idx[:L]])
        r_idx=randgen.permutation(len(li))
        return iter([li[i] for i in r_idx])
            
    def set_epoch(self,epoch):
        self.seed=sum([(i+1)*L for i,L in enumerate(self.lengths)])+epoch

    

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
    checkpoint_name=config['checkpoint_name']
    output_id=config['output_id']

    model=AutoModelForCausalLM.from_pretrained(checkpoint_name,torch_dtype='auto',
                                               attn_implementation='flash_attention_2')

    model_max_length=config['model_max_length']
    
    tokenizer=AutoTokenizer.from_pretrained(checkpoint_name,
                        model_max_length=model_max_length,
                        padding_side="right",
                        use_fast=False)
    ##Assume tokenizer alread has pad token!
    
    if len(config['lora_target'])<=0:
        lora_config = None
        olora_lambda=0.0
        olora_targets=None
    else:
        lora_config = LoraConfig(
            r=config['lora_r'],
            target_modules=config['lora_target'],
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout']
        )

        olora_lambda=config['olora_lambda']
        
        with open('{}/olora_targets.pickle'.format(checkpoint_name),'rb') as fp:
            olora_targets=pickle.load(fp)
            olora_targets_B=pickle.load(fp)
    
        model=get_peft_model(model,lora_config)

        
    if dataset_name=='metamath4':
        dataset=load_dataset('meta-math/MetaMathQA')
        dataset = SupervisedDataset(tokenizer=tokenizer, dataset=dataset['train'])
        
        data_module={'train_dataset':dataset,'eval_dataset':None,
                     'data_collator':DataCollatorFlat2(tokenizer=tokenizer) }
        batch_size=4
        grad_step=4
        
    elif dataset_name.startswith('replay') and dataset_name.endswith('trace4'):
        n_round=int(dataset_name.split('-')[1])
        w_hist=float(dataset_name.split('-')[2])
        li_name=['C-STANCE','FOMC','MeetingBank','Py150','ScienceQA','20Minuten','metamath']
        li_dataset=[]
        for name in li_name[:n_round]:
            if name=='metamath':
                dataset=load_dataset('meta-math/MetaMathQA')
                dataset = SupervisedDataset(tokenizer=tokenizer, dataset=dataset['train'])
            else:
                with gzip.open('{}_split1234.pickle.gz'.format(name),'rb') as fp:
                    dataset=pickle.load(fp)
                dataset=[trace_pretty(d,tokenizer,add_eos=True) for d in dataset]
            li_dataset.append( dataset )
        li_weights=[w_hist*len(li_dataset[-1])/len(d) for d in li_dataset[:-1]]+[1]
        dataset=MixtureDataset(li_dataset,li_weights)
        
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

    output_dir='{}__{}_{}_round2_{}'.format(os.path.dirname(checkpoint_name),output_id,dataset_name,olora_lambda)
        
    training_args = TrainingArguments(output_dir=output_dir,num_train_epochs=config['num_train_epochs'],
                                      per_device_train_batch_size=batch_size,gradient_accumulation_steps=grad_step,
                                      eval_strategy='no',save_strategy='epoch',#'steps',save_steps=1000,
                                      dispatch_batches=False,
                                      learning_rate=2e-5,lr_scheduler_type='constant')

    trainer = MyTrainer(olora_lambda=olora_lambda,olora_targets=olora_targets,
                        model=model, tokenizer=tokenizer, args=training_args, **data_module)
    
    trainer.train()#resume_from_checkpoint=True)

    
    ### save merged model
    if trainer.accelerator.is_main_process and lora_config is not None:
        print('Merge and save')
        #model_unwrapped=trainer.accelerator.unwrap_model(trainer.model_wrapped)
        
        new_olora_targets={}
        new_olora_targets_B={}
        for n,p in model.named_parameters():
            if 'lora_A' in n:
                new_olora_targets[n.split('lora_A')[0]] = torch.cat([ olora_targets[n.split('lora_A')[0]], p.detach().cpu() ])
            if 'lora_B' in n:
                new_olora_targets_B[n.split('lora_B')[0]] = torch.cat([ olora_targets_B[n.split('lora_B')[0]], p.detach().cpu() ], 1)
        model=model.merge_and_unload()
        output_dir=os.path.join(output_dir,'final-merged-e{}'.format(config['num_train_epochs']))
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        with open('{}/olora_targets.pickle'.format(output_dir),'wb') as fp:
            pickle.dump(new_olora_targets,fp)
            pickle.dump(new_olora_targets_B,fp)

    
if __name__=='__main__':
    main()
    
    
