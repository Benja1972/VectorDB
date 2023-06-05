# Vector Databases experiments

## Overview 

[List of vectorstores](https://python.langchain.com/en/latest/modules/indexes/vectorstores.html)

We will experiment with two databases
### Chroma 

[Chroma](https://docs.trychroma.com/) is the open-source embedding database. Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs. Chroma gives you the tools to:

-   store embeddings and their metadata
-   embed documents and queries
-   search embeddings

Chroma prioritizes:

-   simplicity and developer productivity
-   analysis on top of search
-   it also happens to be very quick


### Qdrant 

[Qdrant](https://qdrant.tech/documentation/) is powering the next generation of AI applications with advanced and high-performant vector similarity search technology. 

Qdrant is a vector database & vector similarity search engine. It deploys as an API service providing search for the nearest high-dimensional vectors. With Qdrant, embeddings or neural network encoders can be turned into full-fledged applications for matching, searching, recommending, and much more!

 - Easy to Use API. Provides the OpenAPI v3 specification to generate a client library in almost any programming language. Alternatively utilize ready-made client for Python or other programming languages with additional functionality.
 - Fast and Accurate. Implement a unique custom modification of the HNSW algorithm for Approximate Nearest Neighbor Search. Search with a State-of-the-Art speed and apply search filters without compromising on results.
 - Filtrable. Support additional payload associated with vectors. Not only stores payload but also allows filter results based on payload values. Unlike Elasticsearch post-filtering, Qdrant guarantees all relevant vectors are retrieved.

 ## Use-cases 
  - [Chroma DB for taxonomy embedding](Chroma_taxonomy.md)
  - [Chroma DB for Matching KG](Chroma_kg.md)
  - [Qdrant DB for Matching KG](Qdrant_kg.md)


