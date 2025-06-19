import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="Hugging Face Chatbot")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

tokenizer, model = load_model()

st.title("Hugging Face Chatbot")
st.write("Ask me anything! (Type 'bye' to reset the chat.)")

# Initialize session state
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
    st.session_state.past_inputs = []
    st.session_state.responses = []

user_input = st.text_input("You:", "")

if user_input:
    # Tokenize user input
    new_input = tokenizer(user_input + tokenizer.eos_token, return_tensors='pt')
    input_ids = new_input["input_ids"]
    attention_mask = new_input["attention_mask"]

    # Append to chat history
    if st.session_state.chat_history_ids is not None:
        input_ids = torch.cat([st.session_state.chat_history_ids, input_ids], dim=-1)
        attention_mask = torch.cat([
            torch.ones_like(st.session_state.chat_history_ids), attention_mask
        ], dim=-1)

    # Generate reply
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    # Update session history
    st.session_state.chat_history_ids = output
    reply = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Save conversation
    st.session_state.past_inputs.append(user_input)
    st.session_state.responses.append(reply)

    # Reset chat if user types bye
    if user_input.lower() == "bye":
        st.session_state.chat_history_ids = None
        st.session_state.past_inputs = []
        st.session_state.responses = []
        st.success("Chat reset!")

# Display conversation
for user, bot in zip(st.session_state.past_inputs, st.session_state.responses):
    st.markdown(f"**You:** {user}")
    st.markdown(f"**Bot:** {bot}")
