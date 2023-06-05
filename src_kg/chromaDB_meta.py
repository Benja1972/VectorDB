

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


# ~ from sentence_transformers import util 
import numpy as np
from py2neo import Graph as neoGraph

def norm(x):
    return np.log1p(x)/np.log1p(1)

## == Connect ===========

## == Connect ===========
from secrets import *



### Query List ==================


emmpl_range ={
    "unknown":0,
    "1-10":1,
    "11-50":2, 
    "51-100":3, 
    "51-200":4,
    "101-250":5,
    "251-500":6, 
    "501-1000":7,  
    "1001-5000":8,
    "5001-10000":9,
    "10000+":10
}



def map_range(x):
    return emmpl_range[x]

TEAM = """
MATCH (c:organization)
RETURN Id(c) AS ID, 
        c.board_members_exits_count AS   board_members_exits_count,
        c.deg_people                AS   deg_people,
        c.employee_count            AS   employee_count,
        c.employee_count_range      AS   employee_count_range,
        c.founder_count             AS   founder_count,
        c.founders_exits_count      AS   founders_exits_count,
        c.masters_ratio             AS   masters_ratio,
        c.phd_ratio                 AS   phd_ratio,
        c.team_score                AS   team_score,
        c.team_size_yoy             AS   team_size_yoy
       SKIP $skp LIMIT $topn
"""



GETMAX = """
MATCH (c:organization)
RETURN max(c.ATTRIBUTE) AS  max 
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

attrb = [
    "board_members_exits_count",
    "deg_people",
    "employee_count",
    "founder_count",
    "founders_exits_count",
    "masters_ratio",
    "phd_ratio",
    "team_score",
    "team_size_yoy"
]

# ~ max_values = dict()
# ~ for att in attrb:
    # ~ qr = GETMAX.replace("ATTRIBUTE",att)
    # ~ att_max =  graph.run(qr).data()
    # ~ print(f"Max value {att}: {att_max}")
    # ~ max_values[att] = float(att_max[0]["max"])

max_values = {'board_members_exits_count': 198.0,
             'deg_people': 128920.0,
             'employee_count': 4998.0,
             'founder_count': 1088.0,
             'founders_exits_count': 72.0,
             'masters_ratio': 1.0,
             'phd_ratio': 1.0,
             'team_score': 721.8240662984739,
             'team_size_yoy': 104.0}

graph = neoGraph(NEO4J_LINK, auth=NEO4J_AUTH, name=NEO4J_DATABASE)





CREATE = False

if CREATE:
    chroma_client = chromadb.Client(Settings(
                             chroma_db_impl="duckdb+parquet",
                             persist_directory=".chromaDB"))

    collectTE = chroma_client.create_collection(name="team_graph")


    num_comp =  graph.run(CNT).data()[0]["cnt"]
    topN = 100000

    for sk in range(0,num_comp,topN):
        team =  graph.run(TEAM, skp=sk,topn=topN ).to_data_frame()

        team["employee_count_range"] = team["employee_count_range"].fillna("unknown")
        team["employee_count_range"] = team["employee_count_range"].apply(map_range)

        team = team.fillna(0)
        for att in max_values.keys():
            v = max_values[att]
            team[att] = team[att]/v
            team[att] = team[att].apply(norm)


        iDs = list(team["ID"].astype("str"))
        team.set_index("ID", inplace=True)
        embd = team.to_numpy()
        embd = embd.tolist()
        
        collectTE.add(
            embeddings=embd,
            ids=iDs
        )
        
        print(f"Embeded {len(team)} companies starting from {sk}  \n")


    chroma_client.persist()











# ~ @st.cache_resource
def load_DB():
    chroma_client = chromadb.Client(Settings(
                            chroma_db_impl="duckdb+parquet",
                            persist_directory=".chromaDB"))
    
    collectTE = chroma_client.get_collection(name="team_graph")
    return collectTE

# Use collection
USE = not CREATE
if USE:
    collectTE = load_DB()

 

def search_company(collectTE,txt, topN=5):
    top = collectTE.query(query_texts = [txt], n_results=topN)
    out = {int(k): top["distances"][0][i] for i,k in enumerate(top["ids"][0])}
    
    names = graph.run(GETIDS, idlist=list(out.keys())).to_data_frame()
    names["distance"] = names["ID"].map(out)
    
    return names
    


def get_similar(collectTE,iD, topN=5):
    em = np.array(collectTE.get(ids= [str(iD)], include = ["embeddings"])["embeddings"][0])
    top = collectTE.query(query_embeddings= [em], n_results=topN)
    out = {int(k): top["distances"][0][i] for i,k in enumerate(top["ids"][0])}
    
    ls = list(out.keys())
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

