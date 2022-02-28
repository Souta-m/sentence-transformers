from sentence_transformers import SentenceTransformer,util
#import MeCab
#from sklearn.metrics.pairwise import cosine_similarity
import csv
import pandas as pd
df = pd.read_csv('data/outsourcing.csv',encoding="shift_jis")

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
#wakati = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
#Our sentences we like to encode
with open('data/output/JEouc.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Japanese','English','label','J-Escore'])
    for src,trg,l in zip(df['Japanese'],df['English'],df['label']):
        #src= wakati.parse(src)
        sentences=[src]+[trg]
        #Sentences are encoded by calling model.encode()
        embeddings = model.encode(sentences)

        #Print the embeddings
        print(src,':',trg)
        sim = util.pytorch_cos_sim(embeddings[0],embeddings[1])
        print(sim.item())
        writer.writerow([src,trg,l,sim.item()])
