import streamlit as st
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

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
