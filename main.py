import random

from datasets import load_dataset

from opensearchpy import OpenSearch

host = ''
port = 443
auth = ('admin', '')  # For testing only. Don't store credentials in code.

client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=False
)


def create_index(index_name, field_name, dimensions):
    client.indices.delete(index=index_name)
    client.indices.create(
        index_name,
        body={
            "settings": {
                "index.knn": True,
                "index.number_of_shards": 1,
                "index.number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text"
                    },
                    "views": {
                        "type": "float"
                    },
                    "wiki_id": {
                        "type": "long"
                    },
                    "paragraph_id": {
                        "type": "long"
                    },
                    "title": {
                        "type": "text",
                    },
                    field_name: {
                        "type": "knn_vector",
                        "dimension": dimensions
                    },
                }
            }
        }
    )


def insert_doc(doc_id, index_name, document):
    response = client.index(
        index=index_name,
        body=document,
        id=doc_id,
        refresh=False
    )
    print(response)


def index_documents(limit, index_name, field):
    counter = 0
    docs = load_dataset(f"Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True)
    for doc in docs:
        if counter == limit:
            break
        document = {}
        doc_id = doc['id']
        document['title'] = doc['title']
        document['text'] = doc['text']
        document[field] = doc['emb']
        document['views'] = doc['views']
        document['paragraph_id'] = doc['paragraph_id']
        document['wiki_id'] = doc['wiki_id']
        insert_doc(doc_id, index_name, document)
        counter += 1


def hybrid_query(index, field_name, param, dimensions):
    vec = []
    for j in range(dimensions):
        vec.append(round(random.uniform(0, 1), 2))

    search_query = {
        "_source": {
            "exclude": [
                "target_field"
            ]
        },
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "match": {
                            "text": {
                                "query": param
                            }
                        }
                    },
                    {
                        "knn": {
                            field_name: {
                                "k": 10,
                                "vector": vec,
                            }
                        }
                    }
                ]
            }
        }
    }

    results = client.search(index=index, body=search_query, params={
        "search_pipeline": "nlp-search-pipeline"
    })
    print(results['took'])
    for hit in results["hits"]["hits"]:
        print(hit)


index = "target_index_768"
field_name = "target_field"
create_index(index, field_name, 768)
index_documents(100000, index, field_name)
hybrid_query(index, field_name, "Testament", 768)
