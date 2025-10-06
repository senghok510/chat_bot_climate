import chromadb

print("=== In-memory Client Example ===")
client = chromadb.Client()   # in-memory only

# Step 1: check existing collections
print("Existing collections:", client.list_collections())

# Step 2: create a collection and add some data
collection = client.create_collection("demo")
collection.add(
    documents=["This will disappear after restart!"],
    ids=["1"]
)
print("Added 1 document to 'demo'")

# Step 3: query the collection
results = collection.query(query_texts=["disappear"], n_results=1)
print("Query results:", results)
