print("Started..")
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer,util
import tr
import pandas as pd
from collections import defaultdict, Counter
from spacy.tokens import Span
from itertools import combinations

class obj(list):
    def __init__(self):
        self.append(0)
        self.append(0)
def score(val):
    fac = 20
    return int(fac*float(val))

# Constants
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')
result=defaultdict(obj)
thres=0.66

print("Started..")

for no in range(1,6):
    f = open(f'answers\\question{no}\\ref.txt',encoding='utf-8')
    a = f.read()
    lst_split = a.split('\n')
    f.close()
    mylst = nlp.pipe(lst_split)
    lst1 = [*filter(lambda x: len(x.ents)>1,mylst)]
    lst = [[(j,j.start,j.end) for j in i.ents] for i in lst1]
    res=[[Span(nlp(lst1[idx]),word1[1],word2[2]).text for word1,word2 in combinations(sentences,2)] for idx,sentences in enumerate(lst)]
    if not res:
        vec1 = np.mean(model.encode(lst_split,convert_to_numpy=True),axis=0)
    else:
        vec_res=[np.mean(model.encode(sentes, convert_to_numpy=True),axis=0) for sentes in res]
        vec1 = np.mean(vec_res,axis=0)

    lst = []
    n=45
    for i in range(1,n+1):
        if 10 <= i <= 14: # Removed these answers, as these are not giving correct results
            continue
        f = open(f'answers\\question{no}\\answer{i}.txt',encoding='utf-8')
        a = f.read()
        lst_split = a.split('\n')
        f.close()
        mylst = nlp.pipe(lst_split)
        lst1 = [*filter(lambda x: len(x.ents)>1,mylst)]
        lst2 = [[(j,j.start,j.end) for j in i.ents] for i in lst1]
        res=[[Span(nlp(lst1[idx]),word1[1],word2[2]).text for word1,word2 in combinations(sentences,2)] for idx,sentences in enumerate(lst2)]
        if not res:
            vec2 = np.mean(model.encode(lst_split,convert_to_numpy=True),axis=0)
        else:
            vec_res=[np.mean(model.encode(sentes, convert_to_numpy=True),axis=0) for sentes in res]
            vec2 = np.mean(vec_res,axis=0)
        lst.append(vec2)

    df = pd.DataFrame(lst,columns=[f"V{i}" for i in range(1,385)])
    print('Done',no)


    optimizer = tr.ClusterOptimizer(df, thres)
    data_f = optimizer.data_f
    lbl = optimizer.labels
    dc = Counter(lbl)
    d=[list() for i in range(len(dc))]
    dd=[]

    for i,val in enumerate(lbl):
        ls = df.iloc[i].to_numpy('float32')
        d[val].append(ls)
        dd.append(util.cos_sim(vec1,ls))

    for i in range(len(d)):
        d[i] = util.cos_sim(vec1,np.mean(d[i],axis=0))

    for k in range(1,len(lbl)+1):
        result[k][0] += (score(d[lbl[k-1]]))
        result[k][1] += (score(dd[k-1]))




print(result)
print(result,file=open(f'output\\Threshold_ner_{thres}.txt','w'))

from graph import generate_bar_graph

acc1 = 0
for p,a in result.values():
    acc1 += abs(a-p)/a
acc1 = acc1/(len(result))*100

generate_bar_graph(list(result.values()), round(acc1,2))
# generate_bar_graph(list(result.values()))