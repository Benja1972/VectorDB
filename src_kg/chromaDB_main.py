

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


# ~ from sentence_transformers import util 
import numpy as np
from py2neo import Graph as neoGraph


## == Connect ===========
from secrets import *




### Query List ==================


LIST = """
MATCH (c:organization)
WHERE c.embedding is NOT NULL
RETURN Id(c) AS ID, 
       c.embedding AS embedding 
       SKIP $skp LIMIT $topn
"""


CNT = """
MATCH (c:organization)
RETURN count(c) AS cnt
"""

GETIDS = """
MATCH (c:organization)
WHERE Id(c) in $idlist
RETURN Id(c) AS ID, c.name AS name, c.full_description AS description
"""

### ==========================


graph = neoGraph(NEO4J_LINK, auth=NEO4J_AUTH, name=NEO4J_DATABASE)
model = 'all-mpnet-base-v2'

st_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)









CREATE = False

if CREATE:

    chroma_client = chromadb.Client(Settings(
                            chroma_db_impl="duckdb+parquet",
                            persist_directory=".chromaDB"))

    collectTE = chroma_client.create_collection(name="match_graph", embedding_function=st_model)
    num_comp =  graph.run(CNT).data()[0]["cnt"]
    topN = 100000

    for sk in range(0,num_comp,topN):
        res = graph.run(LIST, topn=topN, skp = sk).to_data_frame()
        collectTE.add(
                embeddings=list(res["embedding"]),
                ids=list(res["ID"].astype("str")))
        print(f"Embeded {len(res)} companies starting from {sk}  \n")
        del res

    chroma_client.persist()





# ~ @st.cache_resource
def load_DB():
    chroma_client = chromadb.Client(Settings(
                            chroma_db_impl="duckdb+parquet",
                            persist_directory=".chromaDB"))
    
    collectTE = chroma_client.get_collection(name="match_graph", embedding_function=st_model)
    collectME = chroma_client.get_collection(name="team_graph", embedding_function=st_model)
    return collectTE, collectME

# Use collection
USE = not CREATE
if USE:
    collectTE, collectME = load_DB()

 

def form_filter(a,b):
    if len(a) == 1:
        fa = {"country_code":a[0]}
    else:
        fa = {"$or": [{"country_code":c} for c in a]}
    
    if len(b) == 1:
        fb = {"stage":b[0]}
    else:
        fb = {"$or": [{"stage":c} for c in b]}

    return {"$and":[fa,fb]}

def search_company(collectTE,txt, topN=5):
    top = collectTE.query(query_texts = [txt], n_results=topN)
    out = {int(k): top["distances"][0][i] for i,k in enumerate(top["ids"][0])}
    
    names = graph.run(GETIDS, idlist=list(out.keys())).to_data_frame()
    names["distance"] = names["ID"].map(out)
    
    return names
    


def get_similar(collectTE,iD, topN=5, where = None):
    em = np.array(collectTE.get(ids= [str(iD)], include = ["embeddings"])["embeddings"][0])
    top = collectTE.query(query_embeddings= [em], n_results=topN, where = where)
    out = {int(k): top["distances"][0][i] for i,k in enumerate(top["ids"][0])}
    
    ls = list(out.keys())
    if iD in ls:
        ls.remove(iD)
    
    names = graph.run(GETIDS, idlist=ls).to_data_frame()
    names["distance"] = names["ID"].map(out)
    return names




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




codes = ["FRA","USA"]

STAGE =["Early", "Growth","Late"]
stage = STAGE[:1]

filt = form_filter(codes,stage)

iD = 12


sc = get_similar(collectTE,iD, topN=100, where = filt)

# ~ sc = get_top_scores(collectTE,collectNE,iD, topN)
# ~ cs =  scoring(sc,u=0.7)














# ~ def scoring(sc,u=0.7):
    # ~ v = 1 - u
    # ~ scc  = u*sc["text_score"] + v*sc["node_score"]
    
    # ~ idx = np.argsort(scc)[0][::-1]

    # ~ return {sc["ids"][i]:scc[0][i] for i in idx}





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
