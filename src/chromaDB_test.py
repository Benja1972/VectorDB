import networkx as nx
# ~ from karateclub import Diff2Vec, Node2Vec
import chromadb
from chromadb.config import Settings
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
from stellargraph import StellarGraph
from sentence_transformers import util 
import numpy as np


import sys
sys.path.append("/home/sergei/ALPHA10X/RD_Projects/KG_KnowledgeGraphs/ALPHA_GENOME/Taxo_Sem_drift/src/")
from taxo2vec import Taxo
import plotly.express as px



def scoring(sc,u=0.7):
    v = 1 - u
    scc  = u*sc["text_score"] + v*sc["node_score"]
    
    idx = np.argsort(scc)[0][::-1]

    return {sc["ids"][i]:scc[0][i] for i in idx}



fin = f"../data/AIB_SD.graphml"
G = nx.read_graphml(fin)



# Create collection
CREATE = True

if CREATE:
    # ~ TxG = Taxo(G)
    # ~ TxG.abstr2corpus()
    sG = StellarGraph.from_networkx(G)
    docs = nx.get_node_attributes(G,"name")
    docs = {k:v +". "+G.nodes[k]["abstract"] if "abstract" in G.nodes[k] else v for k,v in docs.items()}



    chroma_client = chromadb.Client(Settings(
                            chroma_db_impl="duckdb+parquet",
                            persist_directory=".chromaDDB"))
    
    collectTE = chroma_client.create_collection(name="taxo_graph")
    collectTE.add(
        # ~ documents=TxG.docs,
        # ~ ids=TxG.docs_ids
        documents=list(docs.values()),
        ids=list(docs.keys())
    )
    
    
    # Node embedding
    rw = BiasedRandomWalk(sG)

    walks = rw.run(
                nodes=list(sG.nodes()),  # root nodes
                length=3,  # maximum length of a random walk
                n=50,  # number of random walks per root node
                p=0.7,  # Defines (unormalised) probability, 1/p, of returning to source node
                q=0.2,  # Defines (unormalised) probability, 1/q, for moving away from source node
            )


    str_walks = [[str(n) for n in walk] for walk in walks]
    model = Word2Vec(str_walks, window=5, min_count=0, sg=1, workers=2)
    ne = model.wv.get_normed_vectors()


    collectNE = chroma_client.create_collection(name="node2vec_taxo_graph")
    collectNE.add(
        embeddings=ne.tolist(),
        ids=list(sG.nodes())
    )
    
    chroma_client.persist()

# ~ @st.cache_resource
def load_DB():
    chroma_client = chromadb.Client(Settings(
                            chroma_db_impl="duckdb+parquet",
                            persist_directory=".chromaDB"))
    
    collectTE = chroma_client.get_collection(name="taxo_graph")
    collectNE = chroma_client.get_collection(name="node2vec_taxo_graph")
    return collectTE, collectNE

# Use collection
USE = not CREATE
if USE:
    collectTE, collectNE = load_DB()

 

def search_concept(collectTE,txt, topN=5):
    top = collectTE.query(query_texts = [txt], n_results=topN)
    out = {k: top["distances"][0][i] for i,k in enumerate(top["ids"][0])}
    return out
    


# GET prototype 
iD = 'dbr:Constitution_type'
topN = 20

# ~ v = 0.7
# ~ u = 1 - v

def get_top_scores(collectTE,collectNE,iD, topN):

    emTE = np.array(collectTE.get(ids= [iD], include = ["embeddings"])["embeddings"][0])
    emNE = np.array(collectNE.get(ids= [iD], include = ["embeddings"])["embeddings"][0])

    top_emTE = collectTE.query(query_embeddings= [emTE], n_results=topN)["ids"][0]
    top_emNE = collectNE.query(query_embeddings= [emNE], n_results=topN)["ids"][0]

    unn = set(top_emTE).union(set(top_emNE))
    unn.discard(iD)
    unn = list(unn)
    
    unnTE = collectTE.get(ids= unn, include = ["embeddings"])
    unnTE = {k:unnTE["embeddings"][i] for i,k in enumerate(unnTE["ids"])}
    
    unnNE = collectNE.get(ids= unn, include = ["embeddings"])
    unnNE = {k:unnNE["embeddings"][i] for i,k in enumerate(unnNE["ids"])}

    unnE =  {k:{"text_emb":unnTE[k],"node_emb":unnNE[k]} for k in unn}

    uT = np.vstack( [unnE[k]["text_emb"] for k in unn])
    uN = np.vstack( [unnE[k]["node_emb"] for k in unn])

    scT = util.pytorch_cos_sim(emTE, uT).numpy()
    scN = util.pytorch_cos_sim(emNE, uN).numpy()
    return {'ids':unn, "text_score":scT, "node_score":scN} 

sc = get_top_scores(collectTE,collectNE,iD, topN)
cs =  scoring(sc,u=0.7)


# ~ topShow=15
# ~ x = list(cs.values())[topShow::-1]
# ~ y = text = list(cs.keys())[topShow::-1]
# ~ fig = px.bar(x =x,y=y , orientation='h', text =y)
# ~ fig.show()



# ~ G =  nx.convert_node_labels_to_integers(G,label_attribute="ID")


# ~ res = collectTE.query(
    # ~ query_texts=["Gene"],
    # ~ n_results=2)
    
# ~ res = collectNE.get(ids= ["dbr:Gene"], include = ["embeddings"])





# ~ G =  nx.convert_node_labels_to_integers(G,label_attribute="ID")
