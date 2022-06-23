import pickle
import MeCab
import unicodedata
with open('jarujaru3.txt', "r", encoding="utf-8") as f:
      content = f.read()
      c = content.split('\n')
nouns = []
title = []
d = {}
def morpheme(input,speech=False):
    wakati=MeCab.Tagger("-Owakati")
    if speech:
        speech_list=[]
        sentence = wakati.parse(input).split()
        node = wakati.parseToNode(input)
        while node:
            if node.feature.split(",")[0]!="BOS/EOS":
                speech_list.append(node.feature.split(",")[0])    
            node=node.next
        return sentence,speech_list
    else:
        sentence=wakati.parse(input).split()
    return sentence
k = []
for _c in c:
    _c=unicodedata.normalize("NFKC",_c)
    sentence,speech = morpheme(_c,True)
    for s,sp in zip(sentence,speech):
        if sp == '名詞' and s not in k:
            d[_c] = s
            k.append(s)
            break
        else:
            continue
print(len(d.keys()))
with open('jarujaru.pkl','wb') as f:
    pickle.dump(d,f)
    
