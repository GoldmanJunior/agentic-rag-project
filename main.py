import os
from dotenv import load_dotenv
#from langchain.vectorstores import Chroma
#from langchain.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, RagTool
from crewai import Agent, Task, Crew, LLM
from langchain.callbacks import StreamingStdOutCallbackHandler
from db_vectorielle import RemoteVectorDB
from langchain_openai import OpenAIEmbeddings

# Charger les variables d'environnement
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#current_dir = os.path.dirname(os.path.abspath(__file__))
#persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

#embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

vector_db = RemoteVectorDB(
    base_url="http://34.56.86.27:19530/v2/vectordb/entities/query",
    collection_name="all_legal_data"
)

print("Méthodes dispo sur vector_db:", dir(vector_db))


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
    prompt = '''Role: Question-Answering Assistant
Task: Determine whether the system can answer the user's question based on the provided text.
Instructions:
    - Analyze the text and identify if it contains the necessary information to answer the user's question.
    - Provide a clear and concise response indicating whether the system can answer the question or not.
    - Your response should include only a single word. Nothing else, no other text, information, header/footer. 
Output Format:
    - Answer: Yes/No
Study the below examples and based on that, respond to the last question. 
Examples:
    Input: 
        Text: The capital of France is Paris.
        User Question: What is the capital of France?
    Expected Output:
        Answer: Yes
    Input: 
        Text: The population of the United States is over 330 million.
        User Question: What is the population of China?
    Expected Output:
        Answer: No
    Input:
        User Question: {query}
        Text: {text}
'''
    formatted_prompt = prompt.format(text=context, query=query)
    response = llm.invoke(formatted_prompt)
    return response.content.strip().lower() == "yes"

#def setup_web_scraping_agent():
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


#def get_web_content(query):
    """Get web content from web scraping"""
    crew = setup_web_scraping_agent()
    result = crew.kickoff(inputs={"topic": query})
    return getattr(result, "raw", str(result))


def get_local_content(query, vector_db):
    """Get content from the local vector database"""
    docs = vector_db.similarity_search(query, k=5)
    context = "\n".join(docs)
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
    print(f"Local context: {local_context}")

    if can_answer_locally:
        print("Answering from local knowledge...")
        answer = generate_final_answer(query, local_context)
    #else:
        #print("Fetching information from the web...")
        #web_content = get_web_content(query)
        #combined_context = local_context + "\n" + web_content
       # answer = generate_final_answer(query, combined_context)

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
