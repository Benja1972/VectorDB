from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http.models.models import Filter
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "matchingkg"

class NeuralSearcher:

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cpu')
        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def search(self, text: str, filter_: dict = None) -> List[dict]:
        vector = self.model.encode(text).tolist()
        hits = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=Filter(**filter_) if filter_ else None,
            top=5
        )
        return [hit.payload for hit in hits]

class Neural:

    def __init__(self, collection_name: str, model_name: str = 'all-mpnet-base-v2'):
        self.collection_name = collection_name
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, device='cpu')
        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def search(self, text: str, filter_: dict = None, topn: int = 5) -> List[dict]:
        vector = self.model.encode(text).tolist()
        hits = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=("text",vector),
            query_filter=Filter(**filter_) if filter_ else None,
            limit=topn
        )
        return hits

class Similarity:

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def search(self, ids: List[int], filter_: dict = None, topn: int = 5) -> List[dict]:
        hits = self.qdrant_client.recommend(
                            collection_name=self.collection_name,
                            query_filter=Filter(**filter_) if filter_ else None,
                            # ~ negative=[718],
                            positive=ids,
                            using =  "text",
                            with_vectors = True,
                            limit=topn,
        )
        return hits


class Retrivier:

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def get(self, ids: List[int]) -> List[dict]:
        hits = self.qdrant_client.retrieve(
                            collection_name=self.collection_name,
                            ids=ids,
                            with_vectors = True,
        )
        return hits
    
    def get_similar(self, ids: List[int], filter_: dict = None, topn: int = 5) -> List[dict]:
        hits = self.qdrant_client.recommend(
                            collection_name=self.collection_name,
                            query_filter=Filter(**filter_) if filter_ else None,
                            # ~ negative=[718],
                            positive=ids,
                            using =  "text",
                            with_vectors = True,
                            limit=topn,
        )
        return hits




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


def scoring(df,u=0.7, topK = 5):
    v = 1 - u
    df["score"]  = u*df["text_score"] + v*df["team_score"]
    
    df.sort_values(by=['score'],ascending=False, inplace=True)
    df = df.head(topK)

    return df



if __name__ == '__main__':
    neural_searcher = Neural(collection_name=COLLECTION_NAME)
    similar_searcher = Similarity(collection_name=COLLECTION_NAME)
    retriever = Retrivier(collection_name=COLLECTION_NAME)

    country = "FRA"
    filters = {'must': [{'key': 'country_code', 'match': {'value': country}}]}
    
    # ~ res = similar_searcher.search(ids=[2165595], filter_ = filters)
    # ~ ret = retriever.get(ids=[2165595])
    
    
    res = get_top_scores(iD=2165595, topN=100, filters = filters)

