import os,json

def extract_metric(d,use_acc_norm=True):
    if use_acc_norm:
        return d.get('acc_norm,none',d.get('acc,none',d.get('exact_match,flexible-extract',d.get('exact_match,none'))))
    else:
        return d.get('acc,none',d.get('exact_match,flexible-extract',d.get('exact_match,none')))
    
def get_res_eval(filter_txt,use_acc_norm=True):
    res={}
    for name in  os.listdir(LMEVAL_PATH):
        if not all([txt in name for txt in filter_txt]):
            continue
        parts=name.split('__')
        assert parts[0]=='outputs'
        if parts[-1].startswith('checkpoint-'):
            checkpoint_idx=int(parts[-1].split('-')[1])
            config_name='__'.join(parts[1:-1])

            pathname=os.path.join(LMEVAL_PATH,name)
            for fname in os.listdir(pathname):
                with open(os.path.join(pathname,fname),'r') as fp:
                    d_res=json.load(fp)
                for task,d in d_res['results'].items():
                    res.setdefault(config_name,{}).setdefault(task,[]).append( (checkpoint_idx,extract_metric(d,use_acc_norm)) )
        else:
            if parts[1]=='llama3-8b':
                config_name='base1'
            else:
                config_name='__'.join(parts[1:])
            checkpoint_idx=0
            pathname=os.path.join(LMEVAL_PATH,name)
            for fname in os.listdir(pathname):
                with open(os.path.join(pathname,fname),'r') as fp:
                    d_res=json.load(fp)
                for task,d in d_res['results'].items():
                    res.setdefault(config_name,{}).setdefault(task,[]).append( (checkpoint_idx,extract_metric(d,use_acc_norm)) )
    return res

LMEVAL_PATH='/data2/shonglim/hf/cos/outputs_lmeval/'

##cl-lora-replay results

res=get_res_eval(['replay','fl'])
for x in res:
    parts=x.split('_')
    li_route=[]
    s_lambda=set()
    for i,part in enumerate(parts):
        if part.startswith('round'):
            li_route.append(parts[i-1])
        if part.startswith('round2'):
            s_lambda.add(float(parts[i+1]))
    route_name='_'.join(li_route)    
    li=[]
    for dataset in ['mmlu','arc_challenge', 'commonsense_qa', 'hellaswag', 'openbookqa', 'piqa', 'winogrande','gsm8k','gsm8k_cot']:
        assert len(res[x][dataset])==1
        li.append( res[x][dataset][0][1] )
    print(route_name,','.join(['{:.4f}'.format(x) for x in li]))    
