import streamlit as st
import chromadb
import json
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
import torch

# Function to load JSON data
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Function to split content into sentences
def split_into_sentences(content):
    sentences = [sentence.strip() for sentence in content.split('.') if sentence.strip()]
    return sentences

# Load the JSON data
json_file = 'data.json'
data = load_json(json_file)

# Extract sentences and metadata (web links) from loaded data
sentences = []
metadata = []

for entry in data:
    content = entry['content']
    url = entry['url']
    entry_sentences = split_into_sentences(content)
    sentences.extend(entry_sentences)
    metadata.extend([url] * len(entry_sentences))

# Create a dictionary of sentences with IDs
document_dictionary = {str(i+1): sentence for i, sentence in enumerate(sentences)}

# Initialize the ChromaDB client
client = chromadb.Client() # internally, it is using: all-MiniLM-L6-v2

# Corrected collection name
collection_name = 'Nvidia_Docs'
client.create_collection(name=collection_name)

# Get the collection
collection = client.get_or_create_collection(name=collection_name)

# Store text into the database - behind the scenes it will automatically create embeddings from the text
collection.add(
    documents=[x for x in list(document_dictionary.values())],
    metadatas=[{"source": metadata[i]} for i in range(len(metadata))],
    ids=[str(x) for x in document_dictionary]
)

# Load the RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

st.set_page_config(page_title="NVIDIA CUDA Q&A", page_icon=":robot_face:", layout="wide")

st.title("NVIDIA CUDA Documentation Q&A")

question = st.text_input("Enter your question:", "")

if question:
    # Query ChromaDB
    results = collection.query(
        query_texts=[question],
        n_results=10
    )

    # Concatenate the retrieved documents
    retrieved_docs = " ".join([doc for sublist in results['documents'] for doc in sublist])

    # Preprocess the input
    inputs = tokenizer(question, retrieved_docs, return_tensors="pt")

    # Perform the QA inference
    outputs = model(**inputs)

    # Parse the output to get answer
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.decode(inputs['input_ids'][0][start_idx:end_idx])

    st.subheader("Answer:")
    st.write(answer)
