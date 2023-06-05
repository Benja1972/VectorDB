from fastapi import FastAPI

from neural_searcher import Neural
# ~ from qdrant_demo.text_searcher import TextSearcher

COLLECTION_NAME = "matchingkg"
app = FastAPI()

neural_searcher = Neural(collection_name=COLLECTION_NAME)
# ~ text_searcher = TextSearcher()


@app.get("/api/search")
async def read_item(q: str, topn: int = 5, country: str = None):
    if country:
        filters = {'must': [{'key': 'country_code', 'match': {'value': country}}]}

    return {
        "result": neural_searcher.search(text=q, topn = topn, filter_=filters)
        # ~ if neural else text_searcher.search(query=q)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
