from sentence_transformers import SentenceTransformer,util
#import MeCab
#from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
#wakati = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
#Our sentences we like to encode
src='たいていの日本人は、大部分の時間を生きるために働いで過ごす。'
trg='Most Japens people spend most of their time working to live.'
sentences=[src]+[trg]
#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
print(src,':',trg)
sim = util.pytorch_cos_sim(embeddings[0],embeddings[1])
print(sim.item())
