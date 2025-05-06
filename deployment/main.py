import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import tool
from langgraph.prebuilt import create_react_agent
from langchain.prompts import PromptTemplate
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
from csv import DictWriter
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Cybersecurity RAG API", description="API for cybersecurity knowledge retrieval")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# ========== Load env and clients ==========
message_history = InMemoryChatMessageHistory()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
client = OpenAI(api_key=OPENAI_API_KEY)

def chat_complete(client, system_message: str, user_message: str, model: str = "gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content.strip()

# ========== Build vectorstore ==========
def build_vectorstore_from_transcripts(folder_path, persist_path, openai_api_key):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, filename), encoding="utf-8")
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(persist_path)
    return vectorstore

# Fast keyword-based filter for quick initial screening
CYBER_KEYWORDS = [
    "cyber", "security", "phishing", "malware", "encryption", "authentication", "authorization",
    "firewall", "password", "hacking", "threat", "breach", "hashing", "vpn"
]

# ✅ Basic keyword filter
def is_likely_cyber_keywords(text):
    text = text.lower()
    return any(re.search(rf"\b{word}\b", text) for word in CYBER_KEYWORDS)

# ✅ Smart LLM-based filter
def is_likely_cyber_by_llm(query):
    prompt = PromptTemplate.from_template("""
    You are a strict cybersecurity content classifier.

    Decide whether the following question is strictly about cybersecurity (e.g., threats, encryption, phishing, authentication, etc).

    Question: {question}

    Answer with only "yes" or "no".
    """)
    chain = (
        {"question": lambda _: query}
        | prompt
        | llm
    )
    result = chain.invoke({}).content.strip().lower()
    return result.startswith("y")

def get_qa_tool_fn(vectorstore):
    def qa_fn(query, history=None):
        import os

        print("\n🔥 qa_fn called directly")

        # ✅ Retrieve the top 3 most similar chunks
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)

        similarity_score = 0.0
        context_text = ""

        if docs_with_scores:
            doc0, distance = docs_with_scores[0]
            similarity_score = 1 - distance
            context_text = "\n\n".join(doc.page_content for doc, _ in docs_with_scores)

            print("\n🗍 Combined Top 3 Chunks:")
            print(context_text[:500])
        else:
            print("⚠️ No chunks found.")

        print(f"\n📊 Similarity Score: {round(similarity_score, 2)}")

        # ✅ Smart filter combining both methods
        is_cyber = is_likely_cyber_keywords(query) or is_likely_cyber_by_llm(query)
        print(f"🔍 is_likely_cyber: {is_cyber}")

        answer = ""
        contexts_used = ""

        # ✅ CASE 1: High similarity match
        if similarity_score > 0.65:
            prompt = PromptTemplate.from_template("""
            You are a cybersecurity tutor. ONLY answer based on the following:

            Context:
            {context}

            Question:
            {question}
            """)
            chain = (
                {"context": lambda _: context_text, "question": lambda _: query}
                | prompt
                | llm
            )
            answer = chain.invoke({}).content
            contexts_used = context_text  

        # ✅ CASE 2: Medium similarity + relevant to cybersecurity
        elif similarity_score > 0.3 and is_cyber:
            prompt = PromptTemplate.from_template("""
            You are a cybersecurity tutor. This question is OUTSIDE the prepared playlist, but it's still related to cybersecurity.
            Answer clearly and concisely.

            Question:
            {question}
            """)
            chain = (
                {"question": lambda _: query}
                | prompt
                | llm
            )
            answer = chain.invoke({}).content
            contexts_used = "N/A (cyber-related, no strong match)"

        # ❌ CASE 3: Completely out of scope
        else:
            answer = "❌ This question is outside the scope of the cybersecurity playlist."
            contexts_used = "N/A"

        # ✅ Log the interaction in a CSV file
        try:
            # Save the context only if it was actually used
            csv_data = {
                "query": query,
                "answer": answer,
                "contexts": (
                    "" if contexts_used in ["", "N/A", "N/A (cyber-related, no strong match)"]
                    else contexts_used.replace("\n", " ")
                )
            }
            csv_file = "rag_log.csv"
            file_exists = os.path.isfile(csv_file)

            with open(csv_file, mode="a", encoding="utf-8", newline="") as f:
                writer = DictWriter(f, fieldnames=["query", "answer", "contexts"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(csv_data)

            print("📁 Saved to rag_log.csv ✅")
        except Exception as e:
            print("❌ Failed to save CSV:", str(e))

        return answer

    return qa_fn

# ========== Agent Creation ==========
def create_agent_with_memory(vectorstore):
    qa_fn = get_qa_tool_fn(vectorstore)

    @tool
    def CyberSecurityQA_tool(input: str) -> str:
        """
        YOU MUST use this tool for all questions. 
        Do NOT answer directly.
        """
        print("\n🔥 CyberSecurityQA_tool was called")

        # Always pass the question to qa_fn even if it's out of scope
        return qa_fn(input)

    agent = create_react_agent(model=llm, tools=[CyberSecurityQA_tool])

    return RunnableWithMessageHistory(
        agent,
        lambda: message_history,
        input_messages_key="messages",
        history_messages_key="history",
    )

# ========== Request Model ==========
class QuestionRequest(BaseModel):
    question: str

# ========== FastAPI Endpoints ==========
@app.on_event("startup")
def startup_event():
    global vectorstore, agent_with_history
    
    # Initialize vectorstore
    if not os.path.exists("vector_db"):
        build_vectorstore_from_transcripts("transcripts", "vector_db", OPENAI_API_KEY)
    
    vectorstore = FAISS.load_local(
        "vector_db",
        OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        allow_dangerous_deserialization=True
    )
    
    # Initialize agent
    agent_with_history = create_agent_with_memory(vectorstore)
    
    print("🔐 Cybersecurity Agent Ready!")

@app.post("/ask", response_model=dict)
async def ask_question(request: QuestionRequest):
    try:
        user_input = request.question.strip()
        if not user_input:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Check if the question is cybersecurity related using our filters
        is_cyber = is_likely_cyber_keywords(user_input) or is_likely_cyber_by_llm(user_input)
        
        # Process the question through the agent
        previous_messages = message_history.messages[-5:]  # Get last 5 messages for context
        new_message = HumanMessage(content=user_input)
        all_messages = previous_messages + [new_message]
        
        # Invoke the agent
        response = agent_with_history.invoke({"messages": all_messages})
        
        # Extract the final AI response
        final_answers = [
            m for m in response['messages']
            if m.type == "ai" and m.content.strip() != ""
        ]
        
        if final_answers:
            answer = final_answers[-1].content.strip()
        else:
            answer = "No response generated."
        
        return {"question": user_input, "answer": answer}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/generate-question", response_model=dict)
async def generate_question():
    system_prompt = (
        "You are a cybersecurity expert and AI interview assistant. Your task is to generate a single, "
        "high-quality technical cybersecurity question suitable for evaluating a candidate’s understanding. "
        "The question should be clear, concise, and challenging, suitable for intermediate to advanced learners. "
        "Do not include the answer, and avoid yes/no questions. Focus on practical or conceptual knowledge in areas "
        "like network security, malware analysis, encryption, vulnerabilities, or penetration testing."
    )
    user_message = "Generate a cybersecurity interview question."
    question = chat_complete(client, system_prompt, user_message)
    return {"question": question}

class AnswerPayload(BaseModel):
    question: str
    answer: str

@app.post("/evaluate-answer", response_model=dict)
async def evaluate_answer(payload: AnswerPayload):
    system_prompt = (
        "You are an AI evaluator for a cybersecurity quiz. Your job is to analyze whether the user's answer is correct. "
        "Consider correctness and clarity, but allow for partial answers that show some understanding of the topic. "
        "Return only one word: \"True\" if the answer is acceptable (correct or close to correct), or \"False\" if it's clearly wrong or insufficient."
    )
    user_message = f"Question: {payload.question}\nAnswer: {payload.answer}"
    evaluation = chat_complete(client, system_prompt, user_message)
    result = evaluation.strip().lower()
    return {"evaluation": result == "true"}