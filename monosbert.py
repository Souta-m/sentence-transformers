from sentence_transformers import SentenceTransformer,util
#import MeCab
#from sklearn.metrics.pairwise import cosine_similarity
import csv
import pandas as pd
df = pd.read_csv('data/outsourcing.csv',encoding="shift_jis")

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#wakati = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
#Our sentences we like to encode
with open('data/output/EEouc.csv','w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['English1','English2','label','Score'])
    for src,trg,l in zip(df['Correct'],df['English'],df['label']):
        #src= wakati.parse(src)
        sentences=[src]+[trg]
        #Sentences are encoded by calling model.encode()
        embeddings = model.encode(sentences)

        #Print the embeddings
        print(src,':',trg)
        sim = util.pytorch_cos_sim(embeddings[0],embeddings[1])
        print(sim.item())
        writer.writerow([src,trg,l,sim.item()])
