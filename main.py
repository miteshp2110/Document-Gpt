from datetime import time

import requests
from bs4 import BeautifulSoup
import markdown
from dotenv import load_dotenv
from ollama import chat
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import gradio as gr
import os
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
load_dotenv()
api_key= os.environ.get('API_KEY')
pc = Pinecone(api_key=api_key)

load_dotenv()

client = Groq(
    api_key=os.environ.get("API_KEY2"),
)


def fetch_readme_from_github(url, attempts=1):
    url = url.replace(".git", "")
    if attempts:
        initUrl = url + "/refs/heads/main/README.md"
    else:
        initUrl = url + "/refs/heads/master/README.md"

    raw_url = initUrl.replace("github.com", "raw.githubusercontent.com")
    response = requests.get(raw_url)
    if response.status_code == 200:
        return response.text
    else:
        if attempts:
            return fetch_readme_from_github(url, 0)
        else:
            raise Exception("Failed to fetch README file")



def md_to_html_to_text(md_data):
    html = markdown.markdown(md_data)
    soup = BeautifulSoup(html,'html.parser')
    return soup.get_text()


def text_to_chunks(text,chunk_size=200):
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        chunk.append(word)
        if len(" ".join(chunk)) >= chunk_size:
            chunks.append(" ".join(chunk))
            chunk = []
    if chunk :
        chunks.append(" ".join(chunk))
    return chunks

def chunks_to_object(chunks):
    nam = "vec"
    data = []
    for i in range (0,len(chunks)):
        obj = {"id":f"{nam}{i+1}","text":chunks[i]}
        i+=1
        data.append(obj)
    return data

def pine_db(data):
    index_name = "documentation"

    # pc.create_index(
    #     name=index_name,
    #     dimension=1024,
    #     metric="cosine",
    #     spec=ServerlessSpec(
    #         cloud="aws",
    #         region="us-east-1"
    #     )
    # )
    embeddings = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[d['text'] for d in data],
        parameters={"input_type": "passage", "truncate": "END"}
    )
    # print(embeddings[0])

    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    index = pc.Index(index_name)

    vectors = []
    for d, e in zip(data, embeddings):
        vectors.append({
            "id": d['id'],
            "values": e['values'],
            "metadata": {'text': d['text']}
        })

    # print(vectors)

    index.upsert(
        vectors=vectors,
        namespace="ns2"
    )
    # print(index.describe_index_stats())
    return index

def query_pine(index,query):

    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={
            "input_type": "query"
        }
    )
    results = index.query(
        namespace="ns2",
        vector=embedding[0].values,
        top_k=4,
        include_values=False,
        include_metadata=True
    )

    # print(results)
    res_array = results.matches
    fin=[]
    for res in res_array:
        fin.append(res.metadata['text'])


    finl = ""
    for f in fin:
        finl += f
        finl += "\n"
    return finl


def create_vector_db(chunks,model="llama3"):
    embeddings = OllamaEmbeddings(model=model)
    vectorstore = InMemoryVectorStore.from_texts(
        chunks,
        embedding=embeddings
    )
    return vectorstore,embeddings


def query_vector_db(vectorstore, embeddings, query):
    query_embedding = embeddings.embed_query(query)

    results = vectorstore.similarity_search_by_vector(query_embedding, k=10)

    query_result = ""
    for result in results:
        text_content = result.page_content + " "
        query_result += text_content
    return query_result


def generate_llama_response(query,context):
    prompt = context + "\n\n" + query
    response  = chat(model='llama3',messages=[{'role':'user','content':prompt}])

    return response.message.content



def handle_submit(url, query, query_state):

    query_state["prev_query"] = query
    if not query or not url:
        return "provide a query and repo url",gr.update(interactive=False)

    readme_data = fetch_readme_from_github(url)
    text_data = md_to_html_to_text(readme_data)
    chunks = text_to_chunks(text_data)

    data = chunks_to_object(chunks)
    index=pine_db(data)
    res_pine=query_pine(index,query)

    # vectorstore,embeddings = create_vector_db(chunks)


    # context = query_vector_db(vectorstore,embeddings,query)

    prompt = res_pine + "\n\n" + query

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.3-70b-versatile"
    )


    # final_response = generate_llama_response(query,prompt)
    final_response = chat_completion.choices[0].message.content


    return final_response, gr.update(interactive=False)


def enable_submit(query, query_state):

    return gr.update(interactive=(query != query_state.get("prev_query", "")))


def clear_fields():

    return "", "", gr.update(interactive=True), gr.update(interactive=False)


# Define Gradio components and interface
with gr.Blocks() as demo:
    query_state = gr.State({"prev_query": ""})

    with gr.Row():
        url_input = gr.Textbox(label="GitHub Repo URL")
        query_input = gr.Textbox(label="Query", lines=2)

    submit_button = gr.Button("Submit")
    clear_button = gr.Button("Clear")

    output = gr.Textbox(label="Output")

    submit_button.click(handle_submit, inputs=[url_input, query_input, query_state], outputs=[output, url_input])
    query_input.change(enable_submit, inputs=[query_input, query_state], outputs=submit_button)
    clear_button.click(clear_fields, outputs=[url_input, query_input, url_input, submit_button])

demo.launch()
