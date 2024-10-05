from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import json

load_dotenv()

file_name = "psychological-disorders"
file_path = f"./textbooks/{file_name}.pdf"
persist_directory = f"./chromadb/{file_name}"

llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))


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
    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.load():
        pages.append(page)
    return pages


def create_vector_store():
    docs = load_file()

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
As much as possible, try to generate the questions and answers based on the vector store, as well as your existing knowledge of the topic.
The output should be in the following format:
{get_sample_quiz_question_format()}
{additional_info}
    """


def get_rag_chain(question_count, question_topic, extra_context=None):
    if not os.path.exists(persist_directory):
        create_vector_store()

    vectorstore = Chroma(
        persist_directory=persist_directory, embedding_function=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever()
    parser = JsonOutputParser(pydantic_object=QuizQuestionAndAnswer)

    prompt_runnable = RunnableLambda(
        func=lambda _: get_prompt_from_template(
            question_count, question_topic, extra_context
        )
    )
    format_docs_runnable = RunnableLambda(func=format_docs)

    rag_chain = (
        {"context": retriever | format_docs_runnable, "question": RunnablePassthrough()}
        | prompt_runnable
        | llm
        | parser
    )
    return rag_chain


result = get_rag_chain(3, "Mood Disorders").invoke()
print(json.dumps(result, indent=4))
