import os,gzip,pickle
import re
import numpy as np
from fuzzywuzzy import fuzz
from datasets import load_metric
from rouge import Rouge

def score_rouge(str1, str2):
    rouge = Rouge(metrics=["rouge-l"])
    scores = rouge.get_scores(str1, str2, avg=True)
    rouge_l = scores['rouge-l']['f']
    return rouge_l

def caculate_rouge(results, data):
    rouges = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id] 
        if prediction == "" or target == "":
            continue
        try:
            rouge = score_rouge(target, prediction)
            rouges.append(rouge)
        except:
            print('skipped',output_id)
    avg_rouge = sum(rouges) / len(results)
    return avg_rouge

def calculate_sari(inputs, results, data):
    sari = load_metric("sari")
    translation_result = sari.compute(sources=inputs, predictions=results, references=[[label] for label in data]),
    return translation_result

def caculate_fuzz(results, data):
    scores = 0
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id] 
        if prediction == "" or target == "":
            continue
        scores += fuzz.ratio(prediction, target)
    avg_score = scores / len(results) 
    return avg_score

def postprocess(code):
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code


def compute_edim(predicted_sequences, ground_truths):
    outputs = []
    for output in predicted_sequences:
        outputs.append(postprocess(output))
    gts = []
    for gt in ground_truths:
        gts.append(postprocess(gt))

    fuzz = caculate_fuzz(outputs, gts)
    evaluation_result = {"similarity": fuzz}
    
    return evaluation_result

def compute_score(pathname,dataset):
    fname=os.path.join(pathname,'{}.pickle.gz'.format(dataset))
    if not os.path.exists(fname):
        return None
    with gzip.open(fname,'rb') as fp:
        predicted_sequences=pickle.load(fp)
        ground_truths=pickle.load(fp)
        sources_sequences=pickle.load(fp)
    if dataset=='20minuten':
        score=calculate_sari(sources_sequences,predicted_sequences,ground_truths)[0]['sari']/100
    elif dataset=='py150':
        score=compute_edim(predicted_sequences, ground_truths)['similarity']/100
    elif dataset=='meeting':
        score=caculate_rouge(predicted_sequences,ground_truths)
    else:
        li_pred=[x.strip() for x in predicted_sequences]
        li_pred=[x[0] if len(x)>1 else x for x in li_pred]
        li_gt=[x[0] for x in ground_truths]
        score=np.mean([x==y for x,y in zip(li_pred,li_gt)])
    return score

def get_route_name(x):
    parts=x.split('_')
    li_route=[]
    s_lambda=set()
    for i,part in enumerate(parts):
        if part.startswith('round'):
            li_route.append(parts[i-1])
        if part.startswith('round2'):
            s_lambda.add(float(parts[i+1]))
    return '_'.join(li_route)



TRACE_EVAL_PATH='/data2/shonglim/hf/cos/outputs_trace/'

##cl-lora-replay results

res={}
for x in sorted(os.listdir(TRACE_EVAL_PATH)):
    if 'replay' in x and 'fl' in x:
        for dataset in ['cstance','fomc','meeting','py150','science','20minuten']:
            score=compute_score( os.path.join(TRACE_EVAL_PATH,x),dataset)
            res.setdefault(get_route_name(x), {})[dataset]=score
for x in sorted(res.keys()):
    scores=','.join(['{:.4f}'.format(res[x][d]) for d in ['cstance','fomc','meeting','py150','science','20minuten']])
    print(x,scores)
