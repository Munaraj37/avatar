
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import openai
import json

# Load JSON file
with open('./interactive/api/local.settings.json', 'r') as file:
    data = json.load(file)

print(data)

# Environment variables

# Azure Cognitive Search
cognitive_search_endpoint = data["Values"]["AZURE_SEARCH_ENDPOINT"]
cognitive_search_api_key = data["Values"]["AZURE_SEARCH_API_KEY"]
cognitive_search_index_name = data["Values"]["AZURE_SEARCH_INDEX"]

# Azure OpenAI
azure_openai_api_key= data["Values"]["AZURE_OPENAI_API_KEY"]
azure_openai_endpoint = data["Values"]["AZURE_OPENAI_ENDPOINT"]
azure_openai_embed_deployment_name = data["Values"]["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]
azure_openai_deployment_name = data["Values"]["AZURE_OPENAI_CHAT_DEPLOYMENT"]



# OpenAI resource (required for chat scenario)
#azure_openai_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT') # e.g. https://my-aoai.openai.azure.com/
#azure_openai_api_key = os.environ.get('AZURE_OPENAI_API_KEY')
#azure_openai_deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME') # e.g. my-gpt-35-turbo-deployment
#azure_openai_embed_deployment_name = os.environ.get('AZURE_OPENAI_EMBED_DEPLOYMENT_NAME') # e.g. my-gpt-35-turbo-deployment

# Cognitive search resource (optional, only required for 'on your data' scenario)
#cognitive_search_endpoint = os.environ.get('COGNITIVE_SEARCH_ENDPOINT') # e.g. https://my-cognitive-search.search.windows.net/
#cognitive_search_api_key = os.environ.get('COGNITIVE_SEARCH_API_KEY')
#cognitive_search_index_name = os.environ.get('COGNITIVE_SEARCH_INDEX_NAME') # e.g. my-search-index

def get_embeddings(text: str):
    import openai

    client = openai.AzureOpenAI(
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_api_key,
        api_version="2023-09-01-preview",
    )
    embedding = client.embeddings.create(input=[text], model=azure_openai_embed_deployment_name)
    return embedding.data[0].embedding


def get_hotel_index(query: str):
    
    search_client = SearchClient(cognitive_search_endpoint, cognitive_search_index_name, AzureKeyCredential(cognitive_search_api_key))   
    vector = VectorizedQuery(vector=get_embeddings(query),k_nearest_neighbors=2,fields="embedding")
    results = search_client.search(vector_queries=[vector],select=["content"])
    
    input_text=""
    for result in results:
        input_text = input_text + result['content'] + " "

    return input_text

if __name__ == "__main__":

    query = "What are the various account avaiable in Rabobank?"
    input_text = get_hotel_index(query)
#    print(input_text)
    client = openai.AzureOpenAI(azure_endpoint=azure_openai_endpoint,api_key=azure_openai_api_key,api_version="2024-02-01",)
#    completion = client.completions.create(
#        model = azure_openai_deployment_name,
#        prompt = f"Answer Input:{input_text}. Question:{query}",
        #max_tokens=10,
        #top_p=1,
        #frequency_penalty=0,
        #presence_penalty=0
#    )     

    #print(input_text)
    #print(query)

    completion = client.chat.completions.create(
    model=azure_openai_deployment_name,
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer Input:{input_text}. Question:{query}"}
    ]
    ) 

    print(completion.choices[0].message.content)

#    print(completion.choices[0].text)
