import os
import warnings

import pandas as pd
from dotenv import load_dotenv
#from openai import OpenAI
from ollama import Client
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

# Suppress warnings
warnings.filterwarnings('ignore')


def build_rag():
    data = (
        pd
        .read_csv('../../data/top_rated_wines.csv')
        .query('variety.notna()')
        .reset_index(drop=True)
        .to_dict('records')
    )
    print(data[:2])

    # create the vector database client
    qdrant = QdrantClient(":memory:")  # Create in-memory Qdrant instance

    # Create the embedding encoder
    encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Model to create embeddings

    collection_name = "top_wines"
    print(encoder.get_sentence_embedding_dimension())
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
            distance=models.Distance.COSINE
        )
    )

    qdrant.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=idx,
                vector=encoder.encode(doc["notes"]).tolist(),
                payload=doc
            ) for idx, doc in enumerate(data)  # data is the variable holding all the wines
        ]
    )
    print(qdrant.get_collection(collection_name=collection_name))

    user_prompt = "Suggest me an amazing Malbec wine from Argentina"

    query_vector = encoder.encode(user_prompt).tolist()

    hits = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3
    )

    print(hits)

    #  OpenAI account setup and API key generation: https://developers.openai.com/api/docs/quickstart/
    load_dotenv()

    #    client = OpenAI()
    # completion = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system",
    #          "content": "You are chatbot, a wine specialist. "
    #                     "Your top priority is to help guide users into selecting "
    #                     "amazing wine and guide them with their requests."},
    #         {"role": "user", "content": user_prompt},
    #         {"role": "assistant", "content": "Here is my wine recommendation:"}
    #     ]
    # )
    # print(completion.choices[0].message.content)

    # Ollama should be running at http://localhost:11434/ if you need a local version
    # Set you OLLAMA_API_KEY
    client = Client(
        host="https://ollama.com",
        headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
    )

    messages = [
                 {"role": "system",
                  "content": "You are chatbot, a wine specialist. "
                             "Your top priority is to help guide users into selecting "
                             "amazing wine and guide them with their requests."},
                 {"role": "user", "content": user_prompt},
                 {"role": "assistant", "content": "Here is my wine recommendation:"}
    ]
    chat = client.chat(model="qwen3.5:397b-cloud", messages=messages)
    print("No RAG:")
    print(chat.message.content)
    print("=================================")
    print("With RAG:")
    search_results = [hit.payload for hit in hits.points]
    print(f"search results ares: {str(search_results)}")

    chat2 = client.chat(
        model="qwen3.5:397b-cloud",
        messages=[
            {"role": "system",
             "content": "You are chatbot, a wine specialist. "
                        "Your top priority is to help guide users into selecting "
                        "amazing wine and guide them with their requests."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": str(search_results)}
        ]
    )
    print(chat2.message.content)
    print("The end!")

    return


if __name__ == '__main__':
    build_rag()
