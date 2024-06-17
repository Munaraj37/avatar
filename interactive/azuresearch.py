
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
import json

# Load JSON file
with open('./interactive/api/local.settings.json', 'r') as file:
    data = json.load(file)

# Azure Cognitive Search
cognitive_search_endpoint = data["Values"]["AZURE_SEARCH_ENDPOINT"]
cognitive_search_api_key = data["Values"]["AZURE_SEARCH_API_KEY"]
cognitive_search_index_name = data["Values"]["AZURE_SEARCH_INDEX"]


# Azure OpenAI
#openai.api_type = "azure"
azure_openai_api_key= data["Values"]["AZURE_OPENAI_API_KEY"]
azure_openai_endpoint = data["Values"]["AZURE_OPENAI_ENDPOINT"]
#openai.api_version = data["Values"]["AZURE_OPENAI_API_VERSION"]
azure_openai_embed_deployment_name = data["Values"]["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]


# Environment variables
# Speech resource (required)
#speech_region = os.environ.get('SPEECH_REGION') # e.g. westus2
#speech_key = os.environ.get('SPEECH_KEY')
#speech_private_endpoint = os.environ.get('SPEECH_PRIVATE_ENDPOINT') # e.g. https://my-speech-service.cognitiveservices.azure.com/ (optional)

# OpenAI resource (required for chat scenario)
#azure_openai_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT') # e.g. https://my-aoai.openai.azure.com/
#azure_openai_api_key = os.environ.get('AZURE_OPENAI_API_KEY')
#azure_openai_deployment_name = os.environ.get('AZURE_OPENAI_CHAT_DEPLOYMENT') # e.g. my-gpt-35-turbo-deployment
#azure_openai_embed_deployment_name = os.environ.get('AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT') # e.g. my-gpt-35-turbo-deployment

# Cognitive search resource (optional, only required for 'on your data' scenario)
#cognitive_search_endpoint = os.environ.get('AZURE_SEARCH_ENDPOINT') # e.g. https://my-cognitive-search.search.windows.net/
#cognitive_search_api_key = os.environ.get('AZURE_SEARCH_API_KEY')
#cognitive_search_index_name = os.environ.get('AZURE_SEARCH_INDEX') # e.g. my-search-index

print(cognitive_search_endpoint)

def get_document_info():

    import json
    import uuid
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import CharacterTextSplitter

    loader = PyPDFLoader(r"./interactive/data/GenAIdata.pdf")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
    documents = text_splitter.split_documents(documents)

    docs = []
    for doc in documents:
        
        docs.append({"documentID":str(uuid.uuid4()),"content":doc.page_content,"embedding":get_embeddings(doc.page_content)})
        
    json_data=json.dumps(docs)
    
    with open(r"./interactive/data/HandbookContent.json","w") as f:
        f.write(json_data)

    with open(r"./interactive/data/HandbookContent.json","r") as f:
        document = json.load(f)

    return document


def get_embeddings(text: str):
    import openai

    client = openai.AzureOpenAI(
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_api_key,
        api_version="2024-02-01",
    )
    embedding = client.embeddings.create(input=[text], model=azure_openai_embed_deployment_name)
    return embedding.data[0].embedding


def get_index(name: str):
    from azure.search.documents.indexes.models import (
        SearchIndex,
        SearchField,
        SearchFieldDataType,
        SimpleField,
        SearchableField,
        VectorSearch,
        VectorSearchProfile,
        HnswAlgorithmConfiguration,
    )

    fields = [
        SimpleField(name="documentID", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="my-vector-config",
        )
    ]

    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="my-vector-config", algorithm_configuration_name="my-algorithms-config")],
        algorithms=[HnswAlgorithmConfiguration(name="my-algorithms-config")],
    )
   
    return SearchIndex(name=name, fields=fields, vector_search=vector_search)


if __name__ == "__main__":
    print("Started")
    index_client = SearchIndexClient(cognitive_search_endpoint, AzureKeyCredential(cognitive_search_api_key))
    index = get_index(cognitive_search_index_name)
    try:
        if index_client.get_index(cognitive_search_index_name):
            print('Deleting existing index...')
            index_client.delete_index(cognitive_search_index_name)

    except:
        print('Index does not exist. No need to delete it.')

    index_client.create_index(index)
    client = SearchClient(cognitive_search_endpoint, cognitive_search_index_name, AzureKeyCredential(cognitive_search_api_key))
    hotel_docs = get_document_info()
    print("Doc Loaded")
    client.upload_documents(documents=hotel_docs)
    print("Completed")
