import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import chromadb
import json
import random

# Disable the warning about direct execution
st.set_option('deprecation.showfileUploaderEncoding', False)

# Function to load JSON data
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Initialize ChromaDB (cached to persist across reruns)
# @st.cache_resource
def initialize_chromadb():
    client = chromadb.Client()
    collection_name = 'Nvidia_Docs'
    collection = client.get_or_create_collection(name=collection_name)
    return collection

# Load model and tokenizer (cached to persist across reruns)
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return tokenizer, model, device

# Initialize ChromaDB and load model
collection = initialize_chromadb()
doc_count = collection.count()
print(f"Number of documents in the collection after addition: {doc_count}")
tokenizer, model, device = load_model_and_tokenizer()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'is_first_message' not in st.session_state:
    st.session_state.is_first_message = True

# UI Setup
st.title("NVIDIA Docs Chatbot")
st.write("Ask questions about the NVIDIA CUDA Toolkit documentation!")

# Welcome message
if st.session_state.is_first_message:
    st.session_state.chat_history.append({
        'user': '',
        'bot': "Welcome! I'm here to help you with questions about the NVIDIA CUDA Toolkit. How can I assist you today?"
    })
    st.session_state.is_first_message = False

# Chat container
chat_container = st.container()

# Function to display chat messages
def display_chat():
    with chat_container:
        for chat in st.session_state.chat_history:
            user_query, bot_response = chat['user'], chat['bot']
            if user_query:
                st.markdown(f"""
                <div class='chat-message' style='display: flex; justify-content: flex-end;'>
                    <div style='border: 1px solid #c31432; padding: 10px; border-radius: 10px; margin: 5px; background-color:#222222; color: white;'>
                        <b>You:</b> {user_query}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class='chat-message' style='display: flex; justify-content: flex-start;'>
                <div style='border: 1px solid #9216AB; padding: 10px; border-radius: 10px; margin: 5px; background-color: #001a00; color: white;'>
                    <b>Bot:</b> {bot_response}
                </div>
            </div>
            """, unsafe_allow_html=True)

# Display chat history
display_chat()

# Clear chat button
# if st.sidebar.button("Clear Chat"):
#     st.session_state.chat_history = []
#     st.session_state.is_first_message = True
#     st.experimental_rerun()

# Suggested questions
suggested_questions = [
    "What is CUDA?",
    "How do I install CUDA Toolkit?",
    "What are the system requirements for CUDA?",
    "Can you explain CUDA cores?",
    "How do I debug CUDA programs?"
]

# List of default responses when no answer is found
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
    "Apologies, I couldn't find a matching answer based on the provided context."
]

# Process user input
def process_input():
    question = st.session_state.input
    if question:
        # Add user question to chat history
        st.session_state.chat_history.append({'user': question, 'bot': '...'})

        with st.spinner(text="Thinking..."):
            try:
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
                    answer = tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end], skip_special_tokens=True)
                    answer = answer.replace(question, '').strip()
                    if len(answer) == 0:
                        answer = random.choice(responses)
                else:
                    answer = "Sorry, I couldn't find any relevant information."
            except Exception as e:
                answer = f"An error occurred: {str(e)}"

            # Update chat history with bot's response
            st.session_state.chat_history[-1]['bot'] = answer

        # Clear the input field
        st.session_state.input = ""

        # Rerun to update the chat display
        st.experimental_rerun()

# # Display suggested questions
# st.sidebar.header("Suggested Questions")
# for question in suggested_questions:
#     if st.sidebar.button(question):
#         st.session_state.input = question
#         process_input()
# [Previous code remains unchanged until the styling section]

# Styling
st.markdown("""
<style>
.stTextInput>div>input {
    width: 100%;
    padding: 10px;
    border-radius: 10px;
    border: none;
    background: linear-gradient(to bottom, #33ccff, #0066ff);
    color: #ffffff;
    font-size: 16px;
}
.stButton>button {
    width: 100%;
    margin-bottom: 100px;
    padding: 12px 24px;
    border-radius: 15px;
    border: none;
    background: radial-gradient(circle, #FD0008, #611316);
    color: #000000;
    font-size: 16px;
    font-weight: bold;
    text-transform: uppercase;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    transform: scale(1.01);
    box-shadow: 0 0 100px rgba(195, 20, 50, 0.4);
}
#fixed-bottom {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: rgba(0, 0, 0, 0.8);
    padding: 20px;
    z-index: 1000;
}
#fixed-bottom .stTextInput, #fixed-bottom .stButton {
    margin-bottom:10px;
}
.chat-message {
    opacity: 0;
    transform: translateY(20px);
    animation: slide-in 0.5s forwards;
}
@keyframes slide-in {
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
.main .block-container {
    max-width: 100%;
    padding-top: 2rem;
    padding-bottom: 10rem;
    padding-left: 5rem;
    padding-right: 5rem;
}
</style>
""", unsafe_allow_html=True)

# Auto-scroll script
st.markdown("""
<script>
    var chatContainer = window.parent.document.querySelector('.main');
    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    scrollToBottom();
    var observer = new MutationObserver(scrollToBottom);
    observer.observe(chatContainer, {childList: true, subtree: true});
</script>
""", unsafe_allow_html=True)

# Chat container
chat_container = st.container()

# Function to display chat messages
def display_chat():
    with chat_container:
        for chat in st.session_state.chat_history:
            user_query, bot_response = chat['user'], chat['bot']
            if user_query:
                st.markdown(f"""
                <div class='chat-message' style='display: flex; justify-content: flex-end;'>
                    <div style='border: 1px solid #c31432; padding: 10px; border-radius: 10px; margin: 5px; background-color:#222222; color: white;'>
                        <b>You:</b> {user_query}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class='chat-message' style='display: flex; justify-content: flex-start;'>
                <div style='border: 1px solid #9216AB; padding: 10px; border-radius: 10px; margin: 5px; background-color: #001a00; color: white;'>
                    <b>Bot:</b> {bot_response}
                </div>
            </div>
            """, unsafe_allow_html=True)

# Display chat history
display_chat()

# Create a placeholder for the fixed bottom elements
fixed_bottom = st.empty()

# Input field and send button at the bottom (fixed position)
with fixed_bottom.container():
    st.markdown('<div id="fixed-bottom">', unsafe_allow_html=True)
    st.text_input("Enter your question here...", key="input", on_change=process_input)
    st.button("Send", on_click=process_input)
    st.markdown('</div>', unsafe_allow_html=True)