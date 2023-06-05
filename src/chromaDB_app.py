# ~ import networkx as nx
# ~ from karateclub import Diff2Vec, Node2Vec
import chromadb
from chromadb.config import Settings
# ~ from stellargraph.data import BiasedRandomWalk
# ~ from gensim.models import Word2Vec
# ~ from stellargraph import StellarGraph
from sentence_transformers import util 
import numpy as np 
import streamlit as st
import pandas


# ~ import sys
# ~ sys.path.append("/home/sergei/ALPHA10X/RD_Projects/KG_KnowledgeGraphs/ALPHA_GENOME/Taxo_Sem_drift/src/")
# ~ from taxo2vec import Taxo
import plotly.express as px

@st.cache_resource
def load_DB():
    chroma_client = chromadb.Client(Settings(
                            chroma_db_impl="duckdb+parquet",
                            persist_directory=".chromaDDB"))
    
    collectTE = chroma_client.get_collection(name="taxo_graph")
    collectNE = chroma_client.get_collection(name="node2vec_taxo_graph")
    return collectTE, collectNE



@st.cache_resource
def search_concept(txt="gene", topN=5):
    top = collectTE.query(query_texts = [txt], n_results=topN)
    out = {k: top["distances"][0][i] for i,k in enumerate(top["ids"][0])}
    return out

@st.cache_resource
def get_top_scores(iD="dbr:Gene", topN=5):

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


def scoring(sc,u=0.7, topK = 5):
    v = 1 - u
    scc  = u*sc["text_score"] + v*sc["node_score"]
    
    idx = np.argsort(scc)[0][::-1]

    return {sc["ids"][i]:scc[0][i] for i in idx[:topK]}



# ~ fin = f"../data/AIB_SD.graphml"
# ~ G = nx.read_graphml(fin)





collectTE, collectNE = load_DB()

cl = st.columns(2)
with cl[0]:
    txt =  st.text_input(label='Search term',value="Biotechnology")
    sr = search_concept(txt, topN=15)
with cl[1]:
    term = st.selectbox('Term',list(sr.keys()))

# ~ st.write(sr)

# GET prototype 
# ~ iD = "dbr:Gene"
topN = st.slider("Top n results", min_value=3, max_value=30, value=3, step=2)

with st.sidebar:
    st.subheader("Embedding Equalizer")
    ui = st.slider("Text embedding", min_value=0.001, max_value=1., value=0.6, step=0.01)
    vi = st.slider("Graph embedding", min_value=0.001, max_value=1., value=0.6, step=0.01)


 
# == Retrieve 
sc = get_top_scores(term, topN)


u = abs(ui/(ui+vi))

#  == Score
cs =  scoring(sc,u, topN)


ttl = term.split(":")[1]

st.subheader(ttl.replace("_", " "))
st.info("Similar concepts")
st.write(cs)
topShow=15
x = list(cs.values())[topShow::-1]
y = text = list(cs.keys())[topShow::-1]
fig = px.bar(x =x,y=y , orientation='h', text =y)
# ~ fig.update_yaxes(tickfont_family="Inconsolata Black")
# ~ fig.update_xaxes(tickfont_family="Inconsolata Black")
fig.update_layout(font=dict(size=44, family="Inconsolata Black"))
st.write(fig)



# ~ G =  nx.convert_node_labels_to_integers(G,label_attribute="ID")


# ~ res = collectTE.query(
    # ~ query_texts=["Gene"],
    # ~ n_results=2)
    
# ~ res = collectNE.get(ids= ["dbr:Gene"], include = ["embeddings"])





# ~ G =  nx.convert_node_labels_to_integers(G,label_attribute="ID")
