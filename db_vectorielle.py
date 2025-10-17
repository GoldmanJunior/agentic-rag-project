import requests
import os
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class RemoteVectorDB:
    def __init__(self, base_url, collection_name):
        self.base_url = base_url
        self.collection_name = collection_name

    def similarity_search(self, query, k=5):
        endpoint = self.base_url
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        print("Modèle utilisé :", embeddings.model)
        self.query = embeddings.embed_query(query)
        payload = {
            "collectionName": self.collection_name,
            "data": [self.query],
            "limit": k,
        }

        #print(f"Querying vector DB with: {query}")
        #print(f"Payload: {payload}")

        try:
            response = requests.post(endpoint, json=payload, timeout=10)
            print("Réponse complète de l'API:", response.json())
            response.raise_for_status()
            results = response.json().get("results", [])
            print(f"Number of results: {len(results)}")
            return [r["content"] for r in results]
        except Exception as e:
            print(f"Error querying remote vector DB: {e}")
            return []
