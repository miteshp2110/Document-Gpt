import requests
from bs4 import BeautifulSoup
import markdown
from ollama import chat
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


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

    results = vectorstore.similarity_search_by_vector(query_embedding, k=5)

    query_result = ""
    for result in results:
        text_content = result.page_content + " "
        query_result += text_content
    return query_result


def generate_llama_response(query,context):
    prompt = context + "\n\n" + query
    response  = chat(model='llama3',messages=[{'role':'user','content':prompt}])

    return response.message.content

if __name__ == "__main__":
    url = input("Enter URL: ")

    readme_data = fetch_readme_from_github(url)
    text_data = md_to_html_to_text(readme_data)
    chunks = text_to_chunks(text_data)

    vectorstore,embeddings = create_vector_db(chunks)

    query = input("Enter query: ")

    context = query_vector_db(vectorstore,embeddings,query)
    final_response = generate_llama_response(query,context)

    print(final_response)