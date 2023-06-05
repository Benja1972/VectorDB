from qdrant_client import QdrantClient
from qdrant_client import models
import numpy as np
import json
from py2neo import Graph as neoGraph


## == Connect ===========
from secrets import *

graph = neoGraph(NEO4J_LINK, auth=NEO4J_AUTH, name=NEO4J_DATABASE)



# === Query =================
TEAM = """
MATCH (c:organization)
RETURN Id(c) AS ID,
        c.full_description          AS   full_description,
        c.embedding                 AS   embedding, 
        c.name                      AS   name,
        c.country_code_a3           AS   country_code,
        c.stage                     AS   stage,
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


DESCR = """
MATCH (c:organization)
RETURN Id(c) AS ID, 
        c.full_description          AS   full_description
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
# ===========================


# ==== Functions ===========
def norm(x):
    return np.log1p(x)/np.log1p(1)

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


team_attrb = [
    "board_members_exits_count",
    "deg_people",
    "employee_count",
    "employee_count_range",
    "founder_count",
    "founders_exits_count",
    "masters_ratio",
    "phd_ratio",
    "team_score",
    "team_size_yoy"
]



meta_attrb = [
    "name",
    "full_description",
    "country_code",
    "stage"
]

max_values = {'board_members_exits_count': 198.0,
             'deg_people': 128920.0,
             'employee_count': 4998.0,
             'founder_count': 1088.0,
             'founders_exits_count': 72.0,
             'masters_ratio': 1.0,
             'phd_ratio': 1.0,
             'team_score': 721.8240662984739,
             'team_size_yoy': 104.0}

# ==========================



client = QdrantClient(host="localhost", port=6333)

INGEST = False

INDEX = True


if INGEST:
    client.recreate_collection(
      collection_name='matchingkg',
      vectors_config={
            "text": models.VectorParams(size=768, distance=models.Distance.COSINE),
            "team": models.VectorParams(size=10, distance=models.Distance.EUCLID),
        }
    )

    num_comp =  graph.run(CNT).data()[0]["cnt"]
    topN = 1000

    for sk in range(0,num_comp,topN):
    # ~ for sk in range(0,20,topN):
        data =  graph.run(TEAM, skp=sk,topn=topN ).to_data_frame()
        
        
        embeddings=list(data["embedding"])
        ids=list(data["ID"])
        team = data[["ID"]+team_attrb]
        meta = data[meta_attrb]
        meta = meta.to_dict('records')
        
        team["employee_count_range"] = team["employee_count_range"].fillna("unknown")
        team["employee_count_range"] = team["employee_count_range"].apply(map_range)

        team = team.fillna(0)
        for att in max_values.keys():
            v = max_values[att]
            team[att] = team[att]/v
            team[att] = team[att].apply(norm)


        team.set_index("ID", inplace=True)
        embd_team = team.to_numpy()
        embd_team = embd_team.tolist()


        # And the final step - data uploading
        print(f"Ingesting {sk} out of {num_comp}")
        client.upsert(
            collection_name='matchingkg',
            points=models.Batch(
                ids=ids,
                payloads=meta,
                vectors={
                    "text":embeddings,
                    "team":embd_team
                },
            ),
            batch_size=200
        )





if INDEX:
    client.create_payload_index(
        collection_name="matchingkg",
        field_name="full_description",
        field_schema=models.TextIndexParams(
            type="text",
            tokenizer=models.TokenizerType.WORD,
            min_token_len=1,
            max_token_len=5,
            lowercase=True,
        )
    )

