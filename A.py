import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import streamlit as st
import random

# Define a class for question answering
class QuestionAnsweringModel:
    def __init__(self, model_name):
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def answer_question(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors="pt")
        inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        answer_start = torch.argmax(start_logits)
        answer_end = torch.argmax(end_logits) + 1

        answer = self.tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end], skip_special_tokens=True)
        return answer

# Create Streamlit app
def main():
    st.set_page_config(page_title="Question Answering App", layout="wide")

    st.title("Question Answering App")
    st.markdown("""
        Welcome to the Question Answering App powered by RoBERTa!
        Ask a question and provide some context to get an answer.
    """)

    question = st.text_input("Enter your question here:")
    context = st.text_area("Enter the context (e.g., paragraph or document):", height=150)

    if st.button("Get Answer"):
        if question and context:
            qa_model = QuestionAnsweringModel("deepset/roberta-base-squad2")
            answer = qa_model.answer_question(question, context)
            if len(answer) == 0:
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
                st.warning(responses[random.randint(0, len(responses)-1)])
            else:
                st.success(f"**Question:** {question}\n\n**Answer:** {answer}")
        else:
            st.warning("Please provide both a question and context.")

    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This app uses RoBERTa model for question answering. "
        "Enter a question and context to get an answer."
    )
    
    st.sidebar.markdown("### Example")
    st.sidebar.code(
        """
        Question: "Why is model conversion important?"
        Context: "The option to convert models between FARM and transformers gives freedom to the user..."
        """
    )

if __name__ == "__main__":
    main()
