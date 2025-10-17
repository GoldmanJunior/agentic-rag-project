import os
import requests
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from crewai import Agent, Task, Crew,LLM
from crewai_tools import RagTool
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.callbacks import StreamingStdOutCallbackHandler


# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Pydantic model for structured output
class Art(BaseModel):
    article_num: List[str]
    chapter: List[str]
    section: List[str]
    source: List[str]
    text: List[str]
    content: str

# Custom RAG tool for Milvus
class CustomMilvusRagTool(RagTool):
    def _run(self, query: str) -> str:
        embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        embedding = embedding_model.embed_query(query)
        payload = {
            "collectionName": self.config["vectordb"]["config"]["collection_name"],
            "data": [embedding],
            "limit": self.config["vectordb"]["config"]["top_k"],
            "scoreThreshold": self.config["vectordb"]["config"]["score_threshold"],
            "output_fields": ["article_num", "chapter", "section", "source", "text"]
        }
        response = requests.post(self.config["vectordb"]["config"]["api_url"], json=payload)
        results = response.json().get("data", [])
        print(json.dumps(results, indent=2))
        print("Payload sent to Milvus:", payload)
        print("Response from Milvus:", response.text)


        # Extract enriched fields
        article_nums = []
        chapters = []
        sections = []
        sources = []
        texts = []

        for doc in results:
            if doc.get("score", 0) >= self.config["vectordb"]["config"]["score_threshold"]:
                article_nums.append(doc.get("article_num", ""))
                chapters.append(doc.get("chapter", ""))
                sections.append(doc.get("section", ""))
                sources.append(doc.get("source", ""))
                texts.append(doc.get("text", ""))

        # Return structured data as JSON for the agent to parse
        return json.dumps({
            "article_num": article_nums,
            "chapter": chapters,
            "section": sections,
            "source": sources,
            "text": texts
        })

    

# Configuration for Milvus
config = {
    "vectordb": {
        "config": {
            "api_url": "http://34.56.86.27:19530/v2/vectordb/entities/search",
            "collection_name": "all_legal_data",
            "top_k": 5,
            "score_threshold": 0.0
        }
    },
    "embedding_model": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-ada-002"
        }
    }
}

# Initialize custom RAG tool
rag_tool = CustomMilvusRagTool(config=config)

# LLM setup
crewllm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.7,
    max_tokens=500,
    api_key=GOOGLE_API_KEY,
)
llmo = ChatGroq(
    model="grok/llama-3.1-8b-instant",
    temperature=0,
    max_tokens=500,
    timeout=None,
    max_retries=2,
    api_key=GROQ_API_KEY,
    callbacks=[StreamingStdOutCallbackHandler()],
)
llm = LLM(
    model="openai/gpt-4", # call model by provider/model_name
    temperature=0.8,
    max_tokens=1000,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    stop=["END"],
    seed=42
)
# Crew setup
def knowledge():
    legal_agent = Agent(
        role="legal expert",
        goal="Answer accurately and helpfully any legal questions you may have.",
        backstory="""You are a highly knowledgeable legal expert with access to a vast database of legal documents. 
        You can provide accurate and detailed answers to complex legal questions.""",
        tools=[rag_tool],
        llm=llm,
        verbose=True
    )

    task = Task(
        description=(
            "The user asked: '{{query}}'.\n"
            "Always use the RAG tool first to retrieve relevant legal texts from the vector database, even if you think you know the answer. "
            "The tool returns a JSON string with 'article_num', 'chapter', 'section', 'source', and 'text' fields as lists. "
            "Parse the JSON returned by the tool to extract the data. "
            "Your Response must be a JSON object strictly compliant with the Art model, with the following fields: article_num, chapter, section, source, text, and content. "
            "- Populate 'article_num', 'chapter', 'section', 'source', and 'text' fields with the corresponding lists from the parsed JSON. "
            "- The 'content' field must contain a well-written final answer in natural language, "
            "summarizing and explaining the law in plain terms based on the retrieved texts, and must cite the sources (article_num, chapter, section, source) where applicable."
            "réponds uniquement en français."
        ),
        agent=legal_agent,
        expected_output=(
            "A JSON object with 'article_num', 'chapter', 'section', 'source', 'text', and 'content' fields. "
            "'content' must be a reasoned answer that includes references to the sources, not a raw concatenation."
        ),

        tools=[rag_tool],
        output_pydantic=Art,
        input_variables=["query"]
    )

    crew = Crew(
        agents=[legal_agent],
        tasks=[task],
        verbose=True
    )
    return crew

def get_knowledge_content(query):
    """Get knowledge content from Milvus vector database and generate structured legal answer"""

    crew = knowledge()
    result = crew.kickoff(inputs={"query": query})
    return result.pydantic

def serialize_result(result):
    """
    Retourne un dict python à partir de `result` en essayant plusieurs méthodes:
    - si c'est déjà un dict -> renvoie tel quel
    - si c'est un modèle pydantic -> model_dump() (v2) ou dict() (v1)
    - si c'est une string JSON -> json.loads(...)
    - si l'objet a .json() -> parse le JSON retourné
    - sinon essaye vars(result)
    - en dernier recours: {'content': str(result)}
    """
    if result is None:
        return {}

    # déjà dict
    if isinstance(result, dict):
        return result

    # Modèle pydantic
    if isinstance(result, BaseModel):
        # pydantic v2
        if hasattr(result, "model_dump"):
            try:
                return result.model_dump()
            except Exception:
                pass

    # objets qui exposent .model_dump / .json
    for attr in ("model_dump", "json"):
        if hasattr(result, attr):
            try:
                value = getattr(result, attr)()
                # si .json() renvoie string, on la parse en dict
                if attr == "json" and isinstance(value, str):
                    try:
                        return json.loads(value)
                    except Exception:
                        return {"content": value}
                return value
            except Exception:
                continue

    # si c'est déjà une string (peut-être JSON)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return {"content": result}

    # dernier recours: vars()
    try:
        return dict(vars(result))
    except Exception:
        return {"content": str(result)}

def main():
    print("Start chatting with our CIV law specialist! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        print("\nAI: ", end="", flush=True)  # On écrit juste le préfixe une seule fois
        result = get_knowledge_content(query) 
        data=serialize_result(result)
        print("\n") 
        assistant_text = data.get("content") if isinstance(data, dict) and "content" in data else str(data)
        # Update the chat history
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": assistant_text, "full": data})
        print(assistant_text)

if __name__ == "__main__":
    main()