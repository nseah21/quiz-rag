from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

file_name = "java-se-language-updates"
file_path = f"./notes/{file_name}.pdf"
persist_directory = f"./chromadb/{file_name}"
model = "gpt-3.5-turbo-0125"

llm = ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"))


class QuizQuestionAndAnswer(BaseModel):
    question: str = Field(description="The question to be given to the student")
    correct_option: str = Field(description="The ONLY correct answer for this question")
    incorrect_option_1: str = Field(
        description="An incorrect answer for this question, distinct from all the other options"
    )
    incorrect_option_2: str = Field(
        description="An incorrect answer for this question, distinct from all the other options"
    )
    incorrect_option_3: str = Field(
        description="An incorrect answer for this question, distinct from all the other options"
    )


def get_sample_quiz_question_format():
    return [
        QuizQuestionAndAnswer(
            question="Which of the following are birds?",
            correct_option="Magpie",
            incorrect_option_1="Beaver",
            incorrect_option_2="Hornet",
            incorrect_option_3="Guppy",
        ).model_dump_json(indent=4),
        QuizQuestionAndAnswer(
            question="Which of the following is not a cat?",
            correct_option="Penguin",
            incorrect_option_1="Calico cat",
            incorrect_option_2="Sabertooth tiger",
            incorrect_option_3="Pallas cat",
        ).model_dump_json(indent=4),
    ]


def load_file():
    print("Loading PDF file from `notes` folder")
    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.load():
        pages.append(page)
    return pages


def create_vector_store():
    docs = load_file()

    print("Creating vector store from loaded PDF...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    _ = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory,
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_prompt_from_template(question_count, question_topic, extra_context):
    additional_info = extra_context if extra_context else ""
    return f"""
Please generate {question_count} quiz question(s) on the topic \"{question_topic}\".
The quiz questions need to be in multiple choice format, where each question has 4 options. There should only be ONE correct answer for each quiz question.
Try to base your questions and answers on the following context:
{extra_context}
The output should be in the following format:
{get_sample_quiz_question_format()}
    """


def cosine_similarity_filter(question_topic, retrieved_docs, embedding_function):
    """Filter documents based on cosine similarity with the question topic."""
    # Convert the question topic into an embedding vector
    question_embedding = embedding_function.embed_query(question_topic)

    # Embed each retrieved document
    doc_embeddings = [
        embedding_function.embed_query(doc.page_content) for doc in retrieved_docs
    ]

    # Calculate cosine similarity between question and each document
    similarities = cosine_similarity([question_embedding], doc_embeddings)[0]

    # Sort the documents by similarity score (descending) and return top 3 most relevant docs
    ranked_docs = sorted(
        zip(similarities, retrieved_docs), key=lambda x: x[0], reverse=True
    )
    return [doc for _, doc in ranked_docs[:3]]  # Returning top 3 relevant docs


def get_rag_chain(question_count, question_topic, extra_context=None):
    if not os.path.exists(persist_directory):
        create_vector_store()

    print("Using vector store for RAG...")
    vectorstore = Chroma(
        persist_directory=persist_directory, embedding_function=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever()

    # Retrieve relevant documents
    retrieved_docs = retriever.retrieve(question_topic)

    # Apply cosine similarity filtering
    filtered_docs = cosine_similarity_filter(
        question_topic, retrieved_docs, OpenAIEmbeddings()
    )

    # Format the filtered documents
    formatted_context = format_docs(filtered_docs)

    # Construct the prompt using filtered context
    prompt_runnable = RunnableLambda(
        func=lambda _: get_prompt_from_template(
            question_count, question_topic, formatted_context
        )
    )

    parser = JsonOutputParser(pydantic_object=QuizQuestionAndAnswer)

    rag_chain = (
        {"context": retrieved_docs, "question": RunnablePassthrough()}
        | prompt_runnable
        | llm
        | parser
    )
    return rag_chain


def vanilla_gpt(question_count, question_topic, extra_context=None):
    print("Generating response using vanilla GPT...")
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an instructor who is specialised in creating quiz questions.",
            },
            {
                "role": "user",
                "content": get_prompt_from_template(
                    question_count, question_topic, extra_context
                ),
            },
        ],
    )
    return completion.choices[0].message.content


### With RAG
result = get_rag_chain(5, "Latest features in Java").invoke(
    "Please generate me only the latest features in Java (Java 18 and above)!"
)
print(json.dumps(result, indent=4))

### Without RAG
# result = vanilla_gpt(5, "Latest features in Java", "")
# print(result)
