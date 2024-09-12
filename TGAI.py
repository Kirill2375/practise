import telebot
from dotenv import load_dotenv
import PyPDF2
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# Настройка окружения и переменных
def setup_environment():
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBFGavIgm697DrT3iUnSrhrBlJmA1_BuCY"

# Чтение и извлечение текста из PDF-файла
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        return "".join(page.extract_text() for page in pdf_reader.pages)

# Разделение текста на части
def split_text_into_chunks(text, max_size=10000, overlap=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_size, chunk_overlap=overlap)
    return text_splitter.split_text(text)

# Построение векторного индекса
def build_vector_index(chunks, embedding_model_name="models/embedding-001", index_file_name="faiss_index"):
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name)
    vector_index = FAISS.from_texts(chunks, embedding=embeddings)
    vector_index.save_local(index_file_name)

# Настройка цепочки для ответа на вопросы
def setup_conversational_chain():
    prompt_template = """
    В этом файле приведены различные вопросы и ответы о компьютерных играх. Дайте максимально подробный ответ на вопрос, используя предоставленный контекст.\n\n
    Контекст:\n {context}\n
    Вопрос:\n {question}\n
    Ответ:
    """
    chat_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, max_length=10000)
    formatted_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(chat_model, chain_type="stuff", prompt=formatted_prompt)

# Обработка пользовательского запроса с использованием векторного индекса
def process_user_request(query, index_file_name="faiss_index", embedding_model_name="models/embedding-001"):
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name)
    vector_index = FAISS.load_local(index_file_name, embeddings, allow_dangerous_deserialization=True)
    relevant_documents = vector_index.similarity_search(query)

    conversational_chain = setup_conversational_chain()
    return conversational_chain.invoke({"input_documents": relevant_documents, "question": query}, return_only_outputs=True)

# Функция запуска телеграм-бота
def run_telegram_bot(api_token, pdf_path):
    bot = telebot.TeleBot(api_token)

    @bot.message_handler(content_types=['text'])
    def respond_to_message(message):
        pdf_text = extract_text_from_pdf(pdf_path)
        text_chunks = split_text_into_chunks(pdf_text)
        build_vector_index(text_chunks)
        response = process_user_request(message.text)
        bot.send_message(message.chat.id, response['output_text'])

    bot.polling(none_stop=True, interval=0)

# Основной блок программы
if __name__ == "__main__":
    setup_environment()
    API_TOKEN = "7153499614:AAGAVCTvaZcMnphT0hjj5H6RO06Wiyprr3Q"
    PDF_PATH = "question.pdf"
    run_telegram_bot(API_TOKEN, PDF_PATH)
