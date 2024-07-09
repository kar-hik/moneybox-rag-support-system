import cohere
from pinecone import Pinecone, ServerlessSpec
import numpy as np

co = cohere.Client("feqm6L1rQdncmas9yquTodYg04XgLhJ5iVTFeAxU")

with open('scraped.txt', 'r', encoding='utf-8') as file:
    content = file.read().splitlines()

embeds = co.embed(
    texts=content,
    model='embed-english-v3.0',
    input_type='search_document',
    truncate='END'
).embeddings

shape = np.array(embeds).shape
print(f"Embeddings shape: {shape}")

pc = Pinecone(api_key='69ed45f3-116d-4e7a-a399-566d0dc3bcbe')

index_name = 'cohere-pinecone-scraped-data'

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=shape[1],
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)
batch_size = 128

ids = [str(i) for i in range(shape[0])]
meta = [{'text': text} for text in content]

to_upsert = list(zip(ids, embeds, meta))

for i in range(0, shape[0], batch_size):
    i_end = min(i+batch_size, shape[0])
    index.upsert(vectors=to_upsert[i:i_end])

print(index.describe_index_stats())

# Query example
query = "What protection does the Financial Services Compensation Scheme (FSCS) provide for Moneybox Junior ISA holders?"

xq = co.embed(
    texts=[query],
    model='embed-english-v3.0',
    input_type='search_query',
    truncate='END'
).embeddings

print(f"Query embedding shape: {np.array(xq).shape}")

# Query the index and print top matches
res = index.query(vector=xq, top_k=5, include_metadata=True)
for match in res['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")