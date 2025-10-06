import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai import Agent, Task, Crew, LLM
from langchain.callbacks import StreamingStdOutCallbackHandler

# Charger les variables d'environnement
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db=Chroma(persist_directory=persistent_directory, embedding_function=embeddings)


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=500,
    timeout=None,
    max_retries=2,
    api_key=GROQ_API_KEY,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


crewllm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.7,
    max_tokens=500,
    api_key=GOOGLE_API_KEY,
)


def check_local_knowledge(query, context):
    """Router function to determine if we can answer from local knowledge"""
    prompt = f"""Role: Question-Answering Assistant
Task: Determine whether the system can answer the user's question based on the provided text.
Instructions:
    - Answer strictly with one word: Yes or No.

Examples:
Text: The capital of France is Paris.
User Question: What is the capital of France?
Answer: Yes

Text: The population of the United States is over 330 million.
User Question: What is the population of China?
Answer: No

Text: {context}
User Question: {query}
Answer:"""

    response = llm.invoke(prompt)
    answer_text = getattr(response, "content", str(response)).strip().lower()
    return "yes" in answer_text


def setup_web_scraping_agent():
    """Set up the web scraping agent with SerperDevTool"""
    search_tool = SerperDevTool()
    scrape_website = ScrapeWebsiteTool()

    web_search_agent = Agent(
        role="Expert web research agent",
        goal="Identify and retrieve relevant web data for user queries",
        backstory="An expert to identify valuable web sources for user's needs",
        allow_delegation=False,
        verbose=True,
        llm=crewllm,
    )

    web_scraper_agent = Agent(
        role="Expert web scraping agent",
        goal="Extract and analyze content from specific web pages identified by the search agent",
        backstory="A highly skilled web scraper, capable of analyzing and summarizing website content accurately",
        allow_delegation=False,
        verbose=True,
        llm=crewllm,
    )

    search_task = Task(
        description="Identify the most relevant web page or article for the topic: '{topic}'.",
        expected_output="Summary of the most relevant page for '{topic}', including link and key points.",
        tools=[search_tool],
        agent=web_search_agent,
    )

    scraping_task = Task(
        description="Extract and analyze data from the given web page about '{topic}'.",
        expected_output="Detailed summary of content with key insights for '{topic}'.",
        tools=[scrape_website],
        agent=web_scraper_agent,
    )

    crew = Crew(
        agents=[web_search_agent, web_scraper_agent],
        tasks=[search_task, scraping_task],
        verbose=1,
        memory=False,
    )
    return crew


def get_web_content(query):
    """Get web content from web scraping"""
    crew = setup_web_scraping_agent()
    result = crew.kickoff(inputs={"topic": query})
    return getattr(result, "raw", str(result))


#def setup_vector_db(pdf_path):
    """Set up the vector database from a PDF file"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
    vector_db.persist()
    return vector_db


def get_local_content(query, vector_db):
    """Get content from the local vector database"""
    docs = vector_db.similarity_search(query, k=5)
    context = "\n".join([doc.page_content for doc in docs])
    return context


def generate_final_answer(query, context):
    """Generate the final answer using the LLM"""
    prompt = f"""You are a helpful assistant.
Context:
{context}

Question:
{query}

Answer:"""
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


def process_query(query, vector_db):
    """Main function to process the user query"""
    local_context = get_local_content(query, vector_db)
    can_answer_locally = check_local_knowledge(query, local_context)

    print(f"Can answer locally: {can_answer_locally}")

    if can_answer_locally:
        print("Answering from local knowledge...")
        answer = generate_final_answer(query, local_context)
    else:
        print("Fetching information from the web...")
        web_content = get_web_content(query)
        combined_context = local_context + "\n" + web_content
        answer = generate_final_answer(query, combined_context)

    return answer


def main():
    print("Start chatting with our CIV law specialist! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        print("\nAI: ", end="", flush=True)  # On écrit juste le préfixe une seule fois
        result = process_query(query, vector_db)  # Les tokens vont s'afficher en live via le callback

        print("\n") 
        # Update the chat history
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": result})


if __name__ == "__main__":
    main()
