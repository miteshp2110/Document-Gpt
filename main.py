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

load_dotenv()

client = Groq(
    api_key=os.environ.get("API_KEY"),
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




# def gradio_interface(repo_url, query):
#
#     print(repo_url)
#     print(query)
#     return "working"
#     # readme_content = fetch_readme(repo_url)
#     # if "Error" not in readme_content:
#     #     return process_query(readme_content, query)
#     # else:
#     #     return readme_content
#     # return han
#
# # Create Gradio interface
# iface = gr.Interface(
#     fn=gradio_interface,
#     inputs=[gr.Textbox(label="GitHub Repo URL"), gr.Textbox(label="Query", lines=2)],
#     outputs="text",
#     title="Documentation GPT",
#     description="Enter a GitHub repository URL and ask questions about its README."
# )
#
# # Launch the Gradio app
# iface.launch()


# if __name__ == "__main__":
    # url = input("Enter URL: ")
    #
    # readme_data = fetch_readme_from_github(url)
    # text_data = md_to_html_to_text(readme_data)
    # chunks = text_to_chunks(text_data)
    #
    # vectorstore,embeddings = create_vector_db(chunks)
    #
    # query = input("Enter query: ")
    #
    # context = query_vector_db(vectorstore,embeddings,query)
    # final_response = generate_llama_response(query,context)
    #
    # print(final_response)


def handle_submit(url, query, query_state):

    query_state["prev_query"] = query
    if not query or not url:
        return "provide a query and repo url",gr.update(interactive=False)

    readme_data = fetch_readme_from_github(url)
    text_data = md_to_html_to_text(readme_data)
    chunks = text_to_chunks(text_data)

    vectorstore,embeddings = create_vector_db(chunks)


    context = query_vector_db(vectorstore,embeddings,query)

    prompt = context + "\n\n" + query

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.3-70b-versatile"
    )

    # final_response = generate_llama_response(query,context)
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

demo.launch(share=True)
