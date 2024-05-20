import json

from datasets import load_dataset


def index_documents(limit):
    counter = 0
    docs = load_dataset(f"Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True)
    test = []
    for doc in docs:
        if counter < limit:
            counter += 1
            continue
        test.append(doc['emb'])
        if len(test) % 10 == 0:
            print(len(test))
        if len(test) == 10000:
            with open("names.json", "w") as fp:
                json.dump(test, fp)
                print("Done writing JSON data into .json file")
                break


index_documents(10002)
