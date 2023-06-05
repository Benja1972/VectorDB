
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from py2neo import Graph as neoGraph

from sentence_transformers import util 
import numpy as np 
import streamlit as st
import pandas

from  country import *

# ~ import plotly.express as px

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




@st.cache_resource
def load_model(model):
    st_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)
    return st_model


@st.cache_resource
def load_DB():
    chroma_client = chromadb.Client(Settings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=".chromaDB"))

    collectTE = chroma_client.get_collection(name="match_graph", embedding_function=st_model)
    return chroma_client, collectTE

@st.cache_resource
def load_TMDB():

    collectTM = chroma_client.get_collection(name="team_graph")
    return collectTM




@st.cache_resource
def search_company(txt="ALPHA10X", topN=5):
    top = collectTE.query(query_texts = [txt], n_results=topN)
    out = {int(k): top["distances"][0][i] for i,k in enumerate(top["ids"][0])}
    
    names = graph.run(GETIDS, idlist=list(out.keys())).to_data_frame()
    names["distance"] = names["ID"].map(out)
    
    return names

@st.cache_resource
def get_similar(iD=1, topN=5, where=None):
    em = np.array(collectTE.get(ids= [str(iD)], include = ["embeddings"])["embeddings"][0])
    top = collectTE.query(query_embeddings= [em],
                          where = where,
                          n_results=topN)
    out = {int(k): top["distances"][0][i] for i,k in enumerate(top["ids"][0])}
    
    ls = list(out.keys())
    ls.remove(iD)
    
    names = graph.run(GETIDS, idlist=ls).to_data_frame()
    names["distance"] = names["ID"].map(out)
    return names



@st.cache_resource
def get_similar_team(iD=1, topN=5, where=None):
    em = np.array(collectTM.get(ids= [str(iD)], include = ["embeddings"])["embeddings"][0])
    top = collectTM.query(query_embeddings= [em], 
                           where = where,
                           n_results=topN)
    out = {int(k): top["distances"][0][i] for i,k in enumerate(top["ids"][0])}
    
    ls = list(out.keys())
    print(iD, ls)
    if iD in ls:
        ls.remove(iD)
    
    names = graph.run(GETIDS, idlist=ls).to_data_frame()
    names["distance"] = names["ID"].map(out)
    return names

@st.cache_resource
def get_top_scores(iD=1, topN=5, where=None):
    mult = 20

    emTE = np.array(collectTE.get(ids= [str(iD)], include = ["embeddings"])["embeddings"][0])
    emNE = np.array(collectTM.get(ids= [str(iD)], include = ["embeddings"])["embeddings"][0])

    top_emTE = collectTE.query(query_embeddings= [emTE], n_results=mult*topN, where = where)["ids"][0]
    # ~ top_emNE = collectTM.query(query_embeddings= [emNE], n_results=topN, where = where)["ids"][0]
    top_emNE = []
    
    unn = set(top_emTE).union(set(top_emNE))
    inter = set(top_emTE).intersection(set(top_emNE))
    print(f"Common companies - {len(inter)}")
    unn = list(unn)
    if str(iD) in unn:
        unn.remove(str(iD))
    
    
    unnTE = collectTE.get(ids= unn, include = ["embeddings"])
    unnTE = {k:unnTE["embeddings"][i] for i,k in enumerate(unnTE["ids"])}
    
    unnNE = collectTM.get(ids= unn, include = ["embeddings"])
    unnNE = {k:unnNE["embeddings"][i] for i,k in enumerate(unnNE["ids"])}

    unnE =  {k:{"text_emb":unnTE[k],"node_emb":unnNE[k]} for k in unn}

    uT = np.vstack( [unnE[k]["text_emb"] for k in unn])
    uN = np.vstack( [unnE[k]["node_emb"] for k in unn])

    scT = util.pytorch_cos_sim(emTE, uT).numpy()
    scN = util.pytorch_cos_sim(emNE, uN).numpy()
    return {'ids':unn, "text_score":scT, "node_score":scN}

def scoring(sc,u=0.7, topK = 5):
    v = 1 - u
    scc  = u*sc["text_score"] + v*sc["node_score"]
    
    # ~ print(scc)
    idx = np.argsort(scc)[0][::-1]

    return {int(sc["ids"][i]):scc[0][i] for i in idx[:topK]}



model = 'all-mpnet-base-v2'
st_model = load_model(model)

chroma_client, collectTE = load_DB()
collectTM = load_TMDB()
# ~ collectTM = load_TMDB()


STAGE =["Early", "Growth","Late"]




with st.sidebar:
    st.subheader("Filters")
    country = st.multiselect('Select countries',
                            COUNTRY_NAMES,
                            COUNTRY_NAMES[10])
    codes = [COUNTRY[c] for c in country]
    stages =  st.multiselect('Select stage',STAGE, STAGE[0])
    filter = form_filter(codes,stages)
    # ~ st.write(filter)
    st.subheader("Embedding Equalizer")
    ui = st.slider("Text embedding", min_value=0.000, max_value=1., value=0.6, step=0.01)
    vi = st.slider("Team embedding", min_value=0.000, max_value=1., value=0.6, step=0.01)


    
st.header("Semantic Search")
cl = st.columns(2)
with cl[0]:
    
    txt =  st.text_input(label='Search',value="Biotechnology", label_visibility='hidden')
    sr = search_company(txt, topN=40)
    # ~ st.write(sr)
    
with cl[1]:

    # ~ st.header("Similar")
    term = st.selectbox('Find similar company',list(enumerate(sr["name"])))


topNsim = st.slider("Top similar results", min_value=3, max_value=30, value=3, step=2)
iid = sr.iloc[term[0]]["ID"]

# ~ st.write(iid)

sc = get_top_scores(iD=iid, topN=topNsim,where=filter)

u = abs(ui/(ui+vi))

#  == Score
cs =  scoring(sc,u, topNsim)



st.subheader(f"{term[1]}")
with st.expander(""):
    st.write(sr.loc[sr["ID"]==iid]["description"].tolist()[0])
# ~ print(cs)


names = graph.run(GETIDS, idlist=list(cs.keys())).to_data_frame()
names["distance"] = names["ID"].map(cs)

st.write(names)
