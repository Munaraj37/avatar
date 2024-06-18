import logging
import os
import json
import requests
from datetime import datetime, timedelta
import pyodbc
import azure.functions as func

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_API_KEY") 
search_api_version = '2023-07-01-Preview'
search_index_name = os.getenv("AZURE_SEARCH_INDEX")

AOAI_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
AOAI_key = os.getenv("AZURE_OPENAI_API_KEY")
AOAI_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
embeddings_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

# font color adjustments
blue, end_blue = '\033[36m', '\033[0m'

place_orders = False

tools = [
        {
            "type": "function",
            "function": {
                "name": "open_checking_account",
                "description": "Open checking accounts based on the provided parameters",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the customer (i.e., muruga, parthi, etc.)",
                        },
                        "address": {
                            "type": "string",
                            "description": "Address of the customer (i.e., 3522 BC Rijnlaan, etc.)"
                        },
                        "nationality": {
                            "type": "string",
                            "description": "Nationality of the customer (i.e., Indian, Netherlands, etc.)"
                        },
                    },
                    "required": ["name","address","nationality"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_product_information",
                "description": "Find information about a product based on a user question. Any information related to rabobank product should call this function. Use only if the requested information if not already available in the conversation context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_question": {
                            "type": "string",
                            "description": "User question (i.e., Can you explain various product available in Rabobank?, What is the cost of the Current Account?, etc.)"
                        },
                    },
                    "required": ["user_question"],
                },
            },
        }
    ]

def generate_embeddings(text):
    """ Generate embeddings for an input string using embeddings API """

    url = f"{AOAI_endpoint}/openai/deployments/{embeddings_deployment}/embeddings?api-version={AOAI_api_version}"

    headers = {
        "Content-Type": "application/json",
        "api-key": AOAI_key,
    }

    data = {"input": text}

    response = requests.post(url, headers=headers, data=json.dumps(data)).json()
    return response['data'][0]['embedding']

def chat_complete(messages, tools, tool_choice='auto'):
    """  Return assistant chat response based on user query. Assumes existing list of messages """
    
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-03-01-preview"
    )

    response = client.chat.completions.create(
        model=chat_deployment,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    
    return response
 
def get_product_information(user_question, categories='*', top_k=1):
    """ Vectorize user query to search Cognitive Search vector search on index_name."""
     
    url = f"{search_endpoint}/indexes/{search_index_name}/docs/search?api-version={search_api_version}"

    headers = {
        "Content-Type": "application/json",
        "api-key": f"{search_key}",
    }
    
    vector = generate_embeddings(user_question)

    data = {
        "vectors": [
            {
                "value": vector,
                "fields": "embedding",
                "k": top_k
            },
        ],
        "select": "content",
    }

    results = requests.post(url, headers=headers, data=json.dumps(data))    
    results_json = results.json()
    
    # Extracting the required fields from the results JSON
    product_data = results_json['value'][0] # hard limit to top result for now

    response_data = {
        "content": product_data.get('content')
    }
    
    return json.dumps(response_data)
    
def open_checking_account(name, address, nationality):
    
    return json.dumps({
        "name": name,
        "address": address,
        "nationality": nationality
    })
    
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    messages = json.loads(req.get_body())
    response = chat_complete(messages, tools= tools, tool_choice='auto')
    response_message = response.choices[0].message
    
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "open_checking_account": open_checking_account,
            "get_product_information": get_product_information,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = chat_complete(messages, tools= tools, tool_choice='auto')
        response_message = second_response.choices[0].message.content
        
    messages.append({'role' : response_message.role, 'content' : response_message.content})
    #logging.info(json.dumps(response_message))

    response_object = {"messages": messages}

    return func.HttpResponse(
        json.dumps(response_object),
        status_code=200
    )
