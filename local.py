import json
import random
from timeit import default_timer as timer

from datasets import load_dataset

from opensearchpy import OpenSearch, TransportError

host = 'localhost'
port = 9200
auth = ('admin', '')  # For testing only. Don't store credentials in code.

client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=auth,
    use_ssl=False,
    verify_certs=False
)


def create_index(index_name, field_name, dimensions):
    try:
        client.indices.delete(index=index_name)
    except TransportError as e:
        pass
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
                        "dimension": dimensions,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "lucene",
                            "parameters": {
                                "ef_construction": 100,
                                "m": 16
                            }
                        }
                    },
                }
            }
        }
    )


def warmup(index_name):
    method = "GET"
    warmup_url = "/_plugins/_knn/warmup/{}".format(index_name)
    response = client.transport.perform_request(method, warmup_url)
    print(response)

def force_merge(index_name):
    method = "POST"
    warmup_url = "/{}/_forcemerge".format(index_name)
    params = {"max_num_segments": 1}
    response = client.transport.perform_request(method, warmup_url, params)
    print(response)

def refresh(index_name):
    response = client.indices.refresh(index_name)
    print(response)


def insert_doc(doc_id, index_name, document):
    response = client.index(
        index=index_name,
        body=document,
        id=doc_id,
        refresh=False
    )


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


def hybrid_query(index, field_name, param, vec, max):


    search_query = {
        "size": 100,
        "_source": False,
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "knn": {
                            field_name: {
                                "k": 100,
                                "vector": vec,
                            }
                        }
                    },
                    {
                        "match": {
                            "text": {
                                "query": param
                            }
                        }
                    },
                    {
                        "range": {
                            "views": {
                                "gte": 1000,
                                "lte": max,
                            }
                        }
                    },
                    {
                        "range": {
                            "wiki_id": {
                                "gte": 1000,
                                "lte": max,
                            }
                        }
                    },
                ]
            }
        }
    }
    client_start = timer()
    results = client.search(index=index, body=search_query, params={
        "search_pipeline": "nlp-search-pipeline", "timeout": 1000,
    })
    print(results)
    client_end = timer()
    client_time = client_end - client_start
    # print("hybrid: took", results['took'])
    # print("hybrid: client", )
    return client_time
    # for hit in results["hits"]["hits"]:
    #     print(hit)



def term_query(index, param):
    search_query = {
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "match": {
                            "text": {
                                "query": param
                            }
                        }
                    }
                ]
            }
        }
    }
    client_start = timer()
    results = client.search(index=index, body=search_query, params={
        "search_pipeline": "nlp-search-pipeline", "timeout": 1000,
    })
    client_end = timer()
    print("hybrid: took", results['took'])
    print("hybrid: client", client_end - client_start)
    for hit in results["hits"]["hits"]:
        print(hit)

def load_test():
    with open('names.json', 'rb') as fp:
        n_list = json.load(fp)
        return n_list

def boolean_query(index, field_name, param, vec, max):
 
    search_query = {
        "size" : 100,
        "_source": False,
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "text": {
                                "query": param
                            }
                        }
                    },
                    {
                        "range": {
                            "views": {
                                "gte": 1000,
                                "lte": max,
                            }
                        }
                    },
                    {
                        "knn": {
                            field_name: {
                                "k": 100,
                                "vector": vec,
                            }
                        }
                    }
                ]
            }
        }
    }
    client_start = timer()
    results = client.search(index=index, body=search_query, params={
        "search_pipeline": "nlp-search-pipeline", "timeout": 1000,
    })
    client_end = timer()
    client_time = client_end - client_start
    # print("hybrid: took", results['took'])
    # print("hybrid: client", )
    return client_time
    # for hit in results["hits"]["hits"]:
    #     print(hit)


index = "target_index_768"
field_name = "target_field"
# create_index(index, field_name, 768)
# index_documents(50000, index, field_name)
# print("indexing is completed")
# refresh(index)
# force_merge(index)
# refresh(index)
# # warmup(index)
# #term_query(index, "Testament")
vec = []
for k in range(1000):
    vec=[]
    for j in range(768):
        vec.append(round(random.uniform(0, 1), 2))
    hybrid_query(index, field_name, "Day", vec, 2500)
test = load_test()
latency = []
print("warmup is completed")

for row in test:
    client_time = hybrid_query(index, field_name, "Testament", row, 3000)
    latency.append(client_time * 1000)
    # for reading also binary mode is important

#boolean_query(index, field_name, "Testament", vec)

import numpy as np
a = np.array(latency)
print("P50", np.percentile(a, 50))
print("P90", np.percentile(a, 90))
print("P99", np.percentile(a, 99))
print("P100", np.percentile(a, 100))
