import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import chromadb
import json
import random


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
# showWarningOnDirectExecution = False

def split_into_sentences(content):
    sentences = [sentence.strip() for sentence in content.split('.') if sentence.strip()]
    return sentences

# @st.cache(allow_output_mutation=True)
def initialize_chromadb():
    client = chromadb.Client()
    collection_name = 'Nvidia_Docs'
    collection = client.get_or_create_collection(name=collection_name)
    doc_count = collection.count()
    print(f"Number of documents in the collection after addition: {doc_count}")
    return collection


# @st.cache(allow_output_mutation=True)
def insert_data_into_chromadb(data, collection):
    progress_bar = st.progress(0)
    progress_text = st.empty()
    sentences = []
    metadata = []

    total_entries = len(data)
    progress_step = 100 / total_entries

    for idx, entry in enumerate(data):
        content = entry['content']
        url = entry['url']
        entry_sentences = split_into_sentences(content)
        sentences.extend(entry_sentences)
        metadata.extend([url] * len(entry_sentences))

        progress_bar.progress(int((idx + 1) * progress_step))
        progress_text.text(f'Inserting data: {idx + 1}/{total_entries}')

    progress_bar.empty()
    progress_text.empty()

    document_dictionary = {str(i + 1): sentence for i, sentence in enumerate(sentences)}

    collection.add(
        documents=[x for x in list(document_dictionary.values())],
        metadatas=[{"source": metadata[i]} for i in range(len(metadata))],
        ids=[str(x) for x in document_dictionary]
    )
    return collection


json_file = 'data.json'
data = load_json(json_file)


collection = initialize_chromadb()
# if st.button("Insert Data into ChromaDB"):
#     collection = insert_data_into_chromadb(data, collection)
#     st.write("Data inserted into ChromaDB.")


tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
device = torch.device('cuda')
model.to(device)

print("Using device:", device)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# UI Setup
st.title("NVIDIA Docs Chatbot")
st.write("Ask questions about the NVIDIA CUDA Toolkit documentation!")

# Chat interface
chat_container = st.container()
with chat_container:
    for chat in st.session_state.chat_history:
        user_query, bot_response = chat['user'], chat['bot']
        st.markdown(f"""
        <div style='display: flex; justify-content: flex-end;'>
            <div style='border: 1px solid #c31432; padding: 10px; border-radius: 10px; margin: 5px; background-color:#222222; color: white;'>
                <b>You:</b> {user_query}
            </div>
        </div>
        <div style='display: flex; justify-content: flex-start;'>
            <div style='border: 1px solid #240b36; padding: 10px; border-radius: 10px; margin: 5px; background-color: #001a00; color: white;'>
                <b>Bot:</b> {bot_response}
            </div>
        </div>
        """, unsafe_allow_html=True)

responses = [
 "Sorry, I couldn't find an answer to your question.",
 "Unfortunately, I couldn't locate an answer in the provided context.",
 "Apologies, I couldn't find relevant information to answer your question.",
 "I'm sorry, I couldn't find a suitable response in the given context.",
 "Regrettably, I couldn't extract an answer from the provided information.",
 "I apologize, I couldn't locate pertinent details to answer your question.",
 "Sorry, the question seems out of scope for the available information.",
 "Unfortunately, I couldn't determine a valid response given the context.",
 "I'm sorry, the question doesn't appear to match the available data.",
 "Apologies, I couldn't find a matching answer based on the provided context."]

# User input
question = st.text_input("Enter your question here...", key="input")

# Process user input
if st.button("Send"):
    if question:
        # Add user question to chat history
        st.session_state.chat_history.append({'user': question, 'bot': '...'})
        st.session_state.spinner = st.spinner(text="Loading...")
        # Scroll to the bottom of the chat
        st.experimental_rerun()
        # Query ChromaDB for relevant documents
        results = collection.query(
            query_texts=[question],
            n_results=10
        )

        # Process results and perform QA
        if results and 'documents' in results and results['documents']:
            retrieved_docs = " ".join([doc for sublist in results['documents'] for doc in sublist])
            inputs = tokenizer(question, retrieved_docs, return_tensors="pt").to(device)
            outputs = model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer = tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end], skip_special_tokens=True )
            answer = answer.replace(question, '').strip()
            if len(answer) == 0:
                answer = (responses[random.randint(0, len(responses)-1)])
        else:
            answer = "Sorry, I couldn't find any relevant information."

        # Update chat history with bot's response
        st.session_state.chat_history[-1]['bot'] = answer
        # Scroll to the bottom of the chat
        st.experimental_rerun()

# Styling for Streamlit elements
st.markdown("""
    <style>
    .stTextInput>div>input {
        width: 100%;
        padding: 10px;
        border-radius: 10px;
        border: none;
        margin-bottom: 10px;
        background: linear-gradient(to bottom, #33ccff, #0066ff); /* Cyberpunk blue gradient */
        color: #ffffff; /* White text */
        font-size: 16px;
    }
    .stButton>button {
        width: 100%;
        padding: 10px;
        border-radius: 10px;
        border: none;
        background: linear-gradient(to bottom, #c31432, #240b36); /* Cyberpunk pink gradient */
        color: #ffffff; /* White text */
        font-size: 16px;
        margin-bottom: 10px;
    }
    .stContainer {
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #333; /* Dark gray border */
        background: linear-gradient(to bottom, #444, #666); /* Dark gray gradient */
        margin-bottom: 10px;
    }
    .stContainer div {
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)
