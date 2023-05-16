

import chromadb
from chromadb.config import Settings
# ~ from chromadb.utils import embedding_functions


# ~ from sentence_transformers import util 
import numpy as np
from py2neo import Graph as neoGraph

def norm(x):
    return np.log1p(x)/np.log1p(1)

## == Connect ===========

DEV_MTCH = "52.236.176.143"
NEO4J_LINK = f"neo4j://{DEV_MTCH}:7687"

## Admin
NEO4J_AUTH = ("neo4j", "nqLK496LCS6Ec6Rk")
NEO4J_DATABASE = "neo4jdev"




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


META = """
MATCH (c:organization)
RETURN Id(c) AS ID, 
        c.country_code_a3           AS   country_code,
        c.stage                     AS   stage
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








#### Functions ===================

# ~ @st.cache_resource
def load_DB():
    chroma_client = chromadb.Client(Settings(
                            chroma_db_impl="duckdb+parquet",
                            persist_directory=".chromaDB"))
    
    collectTE = chroma_client.get_collection(name="team_graph")
    return collectTE



 

def search_company(collectTE,txt, topN=5):
    top = collectTE.query(query_texts = [txt], n_results=topN)
    out = {int(k): top["distances"][0][i] for i,k in enumerate(top["ids"][0])}
    
    names = graph.run(GETIDS, idlist=list(out.keys())).to_data_frame()
    names["distance"] = names["ID"].map(out)
    
    return names
    


def get_similar(collectTE,iD, topN=5, where=None):
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



#### ==============================



graph = neoGraph(NEO4J_LINK, auth=NEO4J_AUTH, name=NEO4J_DATABASE)





UPDATE = False

if UPDATE:
    chroma_client = chromadb.Client(Settings(
                             chroma_db_impl="duckdb+parquet",
                             persist_directory=".chromaDB"))

    # ~ collectTE = chroma_client.get_collection(name="match_graph")
    collectTE = chroma_client.get_collection(name="team_graph")


    num_comp =  graph.run(CNT).data()[0]["cnt"]
    topN = 10000

    for sk in range(0,num_comp,topN):
        team =  graph.run(META, skp=sk,topn=topN ).data()
        iDs = [t.pop("ID") for t in team]
        iDs = [str(i) for i in iDs]
        
        team = [{k:str(v) for k,v in dt.items()} for dt in team]

        collectTE.update(
                        ids=iDs,
                        metadatas= team,
                    )


        
        print(f"Embeded {len(team)} companies starting from {sk}  \n")


    chroma_client.persist()












# Use collection
USE = not UPDATE
if USE:
    collectTE = load_DB()



codes = ["FRA","USA","NLD"]
stages = ["Early","Growth"]



def form_filter(a,b):
    fa = {"$or": [{"country_code":c} for c in a]}
    fb = {"$or": [{"stage":c} for c in b]}

    return {"$and":[fa,fb]}



filter = form_filter(codes,stages)

# ~ filter = {
    # ~ "$and": [
        # ~ {
            # ~ "country_code": code
        # ~ },
        # ~ {
            # ~ "stage": stage
        # ~ }
    # ~ ]
# ~ }


sm =  get_similar(collectTE,2165595, topN=20, where=filter)
