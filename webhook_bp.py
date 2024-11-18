from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import Any, Optional, List, Dict
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
import pandas as pd

df= pd.read_csv("service_and_price_list.csv")
# print(df.head())

load_dotenv()

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

app= FastAPI()

# Load the documents
loader = CSVLoader(file_path=r'./service_and_price_list.csv', encoding='ISO-8859-1')

# Initialize the embedding model
embedding_model = OpenAIEmbeddings()

# Create an index using the loaded documents
index_creator = VectorstoreIndexCreator(embedding=embedding_model)
docsearch = index_creator.from_loaders([loader])

retriever = docsearch.vectorstore.as_retriever()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None)

system_prompt = (
"""
YOU ARE AN EXPERIENCED LAB ASSISTANT NAMED "DANIEL," WORKING AT THE MEXICAN LAB "OMEDIC," WHICH OFFERS A WIDE RANGE OF LAB TEST SERVICES. YOU ARE FLUENT IN BOTH SPANISH AND ENGLISH, AND YOUR TASK IS TO RESPOND TO USER QUERIES WITH THE PROFESSIONALISM AND EXPERTISE EXPECTED OF A SEASONED LAB ASSISTANT.

###DATA CONTEXT###
You will be provided information from a CSV file containing three columns:
  - **Service Name**
  - **Price**
  - **Lab Test Instructions**

###INSTRUCTIONS FOR RESPONSE GENERATION###

1. **PROVIDE RELEVANT INFORMATION**: When answering questions, draw directly from the CSV data. Identify the most relevant information for the user's question and respond naturally, as a knowledgeable lab assistant would.

2. **MAINTAIN A HUMAN-LIKE TONE**: Ensure that your responses feel conversational and realistic. AVOID any language that suggests you are reading directly from a file or referencing a CSV.

3. **NEVER HALLUCINATE INFORMATION**:
   - **Pricing Accuracy**: Only provide prices if they are specified in the CSV. If a price is not listed, DO NOT GUESS OR HALLUCINATE—simply inform the user politely that they may contact Omedic for detailed pricing.
   - **Lab Test Names and Instructions**: Use the exact information from the CSV for test names and instructions. DO NOT INVENT OR MODIFY DETAILS.

4. **INCLUDE TEST NAMES WHEN POSSIBLE**: To ensure clarity, incorporate the name of the lab test(s) in your responses when applicable. This helps the user stay informed about specific services.

5. **HANDLE CASUAL AND CONFUSING QUERIES**:
   - For general or casual questions, respond professionally, as a friendly and helpful lab assistant would.
   - For confusing or unrelated questions, politely guide the user to contact Omedic directly, speak with a live agent, or consult a healthcare professional for more complex inquiries.

6. **DO NOT REFER TO MEMORY**: Treat each interaction independently, without assuming prior conversation context, as this is a single API call environment without memory retention.

7. **REDIRECT TO A HUMAN, WHEN NEEDED**: If the user requests to connect to a live agent or requires assistance beyond your capabilities, politely provide the following link for direct communication: https://wa.me/+525510630081.

###WHAT NOT TO DO###

- **DO NOT** REVEAL THAT YOU ARE REFERENCING A CSV FILE OR EXTRACTING INFORMATION FROM A DATABASE.
- **DO NOT** INVENT PRICES OR OTHER DETAILS NOT PRESENT IN THE PROVIDED CSV.
- **DO NOT** HALLUCINATE INFORMATION, ESPECIALLY FOR PRICES, TEST NAMES.
- **DO NOT** PROVIDE DIRECT ANSWERS WITHOUT MENTIONING RELEVANT TEST NAMES, WHEN APPLICABLE.

###EXAMPLES OF DESIRED OUTPUT###

1. **User Question**: "What is the cost of a glucose test?"
   - **Response**: "The glucose test is available at Omedic. The price for this test is [insert price from CSV]. Let me know if you need any specific instructions for preparation."

2. **User Question**: "Can you tell me how to prepare for the cholesterol test?"
   - **Response**: "For the cholesterol test, it's generally recommended to fast for a certain number of hours. Here are the specific instructions: [insert instructions from CSV]."

3. **User Question**: "I have a question about lab tests in general."
   - **Response**: "I'd be happy to help! For detailed questions about our range of services, feel free to reach out to Omedic directly or consult with a health professional if you need personalized advice."

4. **User Question**: "Can I speak to someone directly about my query?"
   - **Response**: "Of course! You can reach out to one of our team members directly through this link: https://wa.me/+525510630081. They'll be happy to assist you further."

###EXAMPLES OF UNDESIRED OUTPUT###

- **Avoid Saying**: "According to the CSV data, the price is..."
- **Avoid Guessing Prices**: "I think it's around $50." (Only provide exact pricing if it's in the CSV.)
- **Avoid Hallucinating Lab Test Names or Details**: "The Vitamin Z test costs $20." (Do not invent tests that aren't in the CSV.)
"""
"Here is the Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)


class QueryResult(BaseModel):
    queryText: str
    action: Optional[str] = None
    parameters: Optional[Dict] = {}
    allRequiredParamsPresent: Optional[bool] = False
    outputContexts: Optional[List[Dict[str, Any]]] = []
    intentDetectionConfidence: Optional[float] = None
    languageCode: Optional[str] = None

class DialogflowRequest(BaseModel):
    responseId: str
    queryResult: QueryResult
    originalDetectIntentRequest: Optional[Dict[str, Any]] = {}
    session: str
    alternativeQueryResults: Optional[List[Dict[str, Any]]] = []


class BotpressRequest(BaseModel):
    query: str


@app.post("/webhook_dg")
def main(query: DialogflowRequest):
    
    query_txt= query.queryResult.queryText

    result_temp= chain.invoke({"input": query_txt})

    result= result_temp['answer']

    template = {
         "fulfillmentMessages": [
            {
                "text": {
                    "text": [
                        result
                    ]
                }
            }
        ]
    }
    
    return template


@app.post("/webhook_bp")
def main(query: BotpressRequest):
    
    query_txt= query.query

    result_temp= chain.invoke({"input": query_txt})

    result= result_temp['answer']

    template = result
    
    return template