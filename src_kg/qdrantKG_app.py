# ~ from qdrant_client import QdrantClient
# ~ from qdrant_client import models
import numpy as np
import pandas as pd
import json
from py2neo import Graph as neoGraph
import streamlit as st
from  country import *
from sentence_transformers import  util


from neural_searcher import Neural, Similarity, Retrivier


## == Connect ===========
from secrets import *





graph = neoGraph(NEO4J_LINK, auth=NEO4J_AUTH, name=NEO4J_DATABASE)



@st.cache_resource
def search_company(txt="", topN=5, filters = None):
    
    return neural_searcher.search(text=txt, filter_ = filters,topn = topN )
@st.cache_resource
def similar_company(ids=[], topN=5, filters = None):
    
    return similar_searcher.search(ids=ids, filter_ = filters,topn = topN )




# === Query =================
DESCR = """
MATCH (c:organization)
RETURN Id(c) AS ID, 
        c.full_description          AS   full_description
       SKIP $skp LIMIT $topn
"""


GETIDS = """
MATCH (c:organization)
WHERE Id(c) in $idlist
RETURN Id(c) AS ID, c.name AS name, c.full_description AS description
"""
# ===========================


# ==== Functions ===========
@st.cache_resource
def get_top_scores(iD=1, topN=5, filters = None):
    retriever = Retrivier(collection_name=COLLECTION_NAME)
    # ~ similar_searcher = Similarity(collection_name=COLLECTION_NAME)
    
    res = retriever.get_similar(ids=[iD], filter_ = filters, topn = topN)
    ret = retriever.get(ids=[iD])


    emTE = np.array(ret[0].vector["text"])
    emME = np.array(ret[0].vector["team"])
    
    top_emTE = np.array([hit.vector["text"] for hit in res])
    top_emME = np.array([hit.vector["team"] for hit in res])
    
    
    ids = [hit.id for hit in res]
    comp = pd.DataFrame([hit.payload for hit in res])

    scT = util.pytorch_cos_sim(emTE, top_emTE).numpy()
    scM = util.pytorch_cos_sim(emME, top_emME).numpy()
    comp["text_score"] = scT[0]
    comp["team_score"] = scM[0]
    
    return  comp#, scT, scM #{'ids':ids, "text_score":scT, "team_score":scM}

@st.cache_resource
def scoring(df,u=0.7, topK = 5):
    v = 1 - u
    df["score"]  = u*df["text_score"] + v*df["team_score"]
    
    df.sort_values(by=['score'],ascending=False, inplace=True)
    df = df.head(topK)

    return df






# ======================
COLLECTION_NAME = "matchingkg"
# ~ app = FastAPI()

neural_searcher = Neural(collection_name=COLLECTION_NAME)
similar_searcher = Similarity(collection_name=COLLECTION_NAME)
STAGE =["Early", "Growth","Late"]

with st.sidebar:
    st.subheader("Filters")
    country = st.multiselect('Select countries',
                            COUNTRY_NAMES,
                            COUNTRY_NAMES[10])
    codes = [COUNTRY[c] for c in country]
    stages =  st.multiselect('Select stage',STAGE, STAGE[0])
    
    
    st.divider()
    st.subheader("Embedding Equalizer")
    ui = st.slider("Text embedding", min_value=0.000, max_value=1., value=0.6, step=0.01)
    vi = st.slider("Team embedding", min_value=0.000, max_value=1., value=0.6, step=0.01)


st.header("Semantic Search")




# ~ cl = st.columns(2)
# ~ with cl[0]:
    
txt =  st.text_input(label='Search',value="Biotechnology", label_visibility='hidden')
topN = st.slider("Top results", min_value=5, max_value=400, value=10, step=5)




filters = {'must': [{'key': 'country_code', 'match': {'any': codes}}, {'key': 'stage', 'match': {'any': stages}}]} if len(codes)+len(stages)>0  else None



res = search_company(txt=txt, filters = filters,topN = topN )

comp = [hit.payload for hit in res]
df = pd.DataFrame(comp)
st.write(df)


ls = [(hit.id,hit.payload["name"]) for  hit in res]
company = st.selectbox('Find similar company',ls)

st.subheader("Similar company")
# ~ res_sim = similar_company(ids=[company[0]], filters = filters,topN = topN )
res_sim = get_top_scores(iD=company[0], filters = filters,topN = 10*topN )

u = abs(ui/(ui+vi))
comps = scoring(res_sim,u=u, topK = topN)
# ~ dfs = pd.DataFrame(comps)
st.write(comps)

# ~ print(ls)


# ~ sr = search_company(txt, topN=40)
    # ~ st.write(sr)


# ~ client = QdrantClient(host="localhost", port=6333)

