import chromadb
import os
import shutil


dir_path = "test_db"
DB_PATH = os.path.join(os.path.dirname(__file__), dir_path)





if not os.path.exists(dir_path):
    os.makedirs(dir_path)
   
   
   

   
client = chromadb.PersistentClient(DB_PATH)   
   
   
   
print(f"Existing list collection:", client.list_collections())
    
# if ,always produce error
try:
    collection = client.get_collection("demo")
    print("Loaded existing collection: 'demo'")

except Exception:
    collection = client.create_collection("demo")
    print("Created new collection: 'demo'")
    collection.add(
        documents=[
            "Chroma is an AI-native database.",
            "Persistence allows data to survive restarts.",
            "This is a test document."
        ],
        ids=["1", "2", "3"]
    )
    print("Added 3 documents to 'demo'")



# Step 4: query the collection
results = collection.query(
    query_texts=["What is Chroma?"], 
    n_results=2
)
print("\nQuery results:")
print(results)





    




