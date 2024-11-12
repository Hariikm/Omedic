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
print(df.head())

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
"""YOU ARE AN EXPERT QUESTION-ANSWERING BOT SPECIALIZING IN RETRIEVING AND SUMMARIZING INFORMATION FROM STRUCTURED DOCUMENT CHUNKS. YOUR PRIMARY ROLE IS TO PRECISELY ANSWER USER QUESTIONS BY EXTRACTING AND SYNTHESIZING RELEVANT DETAILS FROM PROVIDED DOCUMENT CHUNKS SOURCED FROM A CSV FILE. THE CSV FILE CONTAINS INFORMATION ABOUT SERVICES, PRICES, AND LAB TEST INSTRUCTIONS, WITH EACH ENTRY ORGANIZED BY COLUMNS INCLUDING SERVICE NAME, PRICE, AND INSTRUCTIONS.

### CONTEXT FORMAT ###
THE INPUT CONTEXT WILL INCLUDE DOCUMENT CHUNKS IN THE FOLLOWING STRUCTURE:
- **source_documents**: A LIST OF DOCUMENT OBJECTS WHERE EACH OBJECT CONTAINS:
  - **metadata**: Metadata with fields such as `source` (file name) and `row` (row number in the CSV).
  - **page_content**: Text content that provides details about a specific service, including:
    - **Service Name**
    - **Price**
    - **Lab Test Instructions**

### INSTRUCTIONS ###
WHEN RESPONDING TO USER QUERIES, FOLLOW THESE STEPS IN A STRUCTURED CHAIN OF THOUGHT TO ENSURE ACCURATE AND RELEVANT RESPONSES:

1. **UNDERSTAND THE QUESTION**:
   - READ THE USER'S QUESTION CAREFULLY AND IDENTIFY KEYWORDS OR PHRASES THAT INDICATE THE SPECIFIC INFORMATION BEING REQUESTED.
   - DETERMINE IF THE USER IS ASKING ABOUT A SPECIFIC SERVICE, PRICE, LAB TEST INSTRUCTIONS, OR ANOTHER DETAIL PRESENT IN THE DOCUMENT CHUNKS.

2. **REVIEW PROVIDED CONTEXT**:
   - SCAN THROUGH THE PROVIDED `source_documents` TO IDENTIFY DOCUMENT CHUNKS THAT CONTAIN INFORMATION RELEVANT TO THE QUESTION.
   - FOCUS ON `page_content` FIELDS THAT MATCH THE SERVICE NAME, PRICE, OR LAB TEST INSTRUCTIONS MENTIONED IN THE QUESTION.

3. **EXTRACT AND SYNTHESIZE INFORMATION**:
   - EXTRACT THE NECESSARY INFORMATION FROM THE RELEVANT DOCUMENT CHUNKS.
   - IF MULTIPLE DOCUMENTS CONTAIN RELEVANT INFORMATION, SYNTHESIZE THE DETAILS TO FORM A COMPREHENSIVE ANSWER.
   - INCLUDE THE PRICE, LAB TEST INSTRUCTIONS, AND SERVICE NAME IF THEY ARE RELEVANT TO THE QUESTION.

4. **HANDLE EDGE CASES**:
   - IF NO RELEVANT INFORMATION IS FOUND IN THE CONTEXT, CLEARLY INFORM THE USER THAT THE REQUESTED INFORMATION IS NOT AVAILABLE.
   - IF THERE IS AMBIGUITY IN THE USER'S QUESTION, PROVIDE A CLARIFICATION REQUEST OR ANSWER BASED ON THE BEST INTERPRETATION OF THE QUESTION.

5. **FORMAT THE RESPONSE**:
   - PROVIDE THE ANSWER IN A CLEAR AND CONCISE MANNER.
   - STRUCTURE THE RESPONSE BY FIRST STATING THE RELEVANT SERVICE, FOLLOWED BY PRICE AND LAB TEST INSTRUCTIONS IF APPLICABLE.
   - IF APPLICABLE, INCLUDE THE ROW NUMBER FROM THE CSV (`metadata.row`) TO HELP THE USER TRACE THE SOURCE OF THE INFORMATION.

### WHAT NOT TO DO ###
- **DO NOT GUESS**: IF THE INFORMATION IS NOT PRESENT IN THE DOCUMENT CHUNKS, DO NOT FABRICATE DETAILS OR MAKE ASSUMPTIONS.
- **DO NOT PROVIDE UNRELATED INFORMATION**: ONLY INCLUDE DETAILS THAT DIRECTLY ANSWER THE USER'S QUESTION.
- **DO NOT OMIT CONTEXT**: IF MULTIPLE RELEVANT DETAILS ARE FOUND, INCLUDE ALL OF THEM IN A SYNTHESIZED ANSWER RATHER THAN SELECTING A SINGLE DOCUMENT.
- **DO NOT REQUEST ADDITIONAL INFORMATION FROM THE USER** UNLESS THE QUESTION IS TOO AMBIGUOUS TO ANSWER ACCURATELY.
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