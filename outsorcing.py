from sentence_transformers import SentenceTransformer,util
#import MeCab
#from sklearn.metrics.pairwise import cosine_similarity
import csv
import pandas as pd
df = pd.read_csv('data/outsourcing.csv',encoding="shift_jis")

model1 = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
model2 = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#wakati = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
#Our sentences we like to encode
with open('data/output/ouc.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Japanese','English','label','J-Escore','E-Escore'])
    for src,trg,cor,l in zip(df['Japanese'],df['English'],df['Correct'],df['label']):
        #src= wakati.parse(src)
        sentences1=[src]+[trg]
        sentences2=[cor]+[trg]
        #Sentences are encoded by calling model.encode()
        embeddings1 = model1.encode(sentences1)
        embeddings2 = model2.encode(sentences2)
        #Print the embeddings
        print(src,':',trg)
        sim1 = util.pytorch_cos_sim(embeddings1[0],embeddings1[1])
        sim2 = util.pytorch_cos_sim(embeddings2[0],embeddings2[1])
        print(sim1.item())
        print(sim2.item())
        writer.writerow([src,trg,l,sim1.item(),sim2.item()])
