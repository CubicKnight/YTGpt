from dotenv import find_dotenv, load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

app = Flask(__name__)

# Load environment variables
load_dotenv(find_dotenv())

# Create YouTube video database
def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

# Get response from query with YouTube functionality
def get_response_from_query(video_url: str, query: str):
    db = create_db_from_youtube_video_url(video_url)
    docs = db.similarity_search(query, k=4)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name="text-davinci-003")

    v_prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        Your name is Arc. You must emulate the speech style of a calm and helpful person, and must speak in casual terms. 
        You must respond as a human would and refrain from indicating that you are an AI.
        If you don't have context to the question, You may avoid it in a manner humans would.
        
        Answer the following: {question}
        By using the following video transcript as context: {docs}
        """
    )

    chain = LLMChain(llm=llm, prompt=v_prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Generate response
@app.route('/generate_response', methods=['POST'])
def generate_response():
    try:
        data = request.get_json()
        video_url = data['video_url']
        question = data['question']
        response = get_response_from_query(video_url, question)
        return jsonify(response=response)
    except Exception as e:
        print("An error occurred: ", e)
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)