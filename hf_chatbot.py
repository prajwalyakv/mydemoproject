from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Track chat history
chat_history_ids = None

print("Hugging Face Chatbot ready! Type 'bye' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "bye":
        print("Chatbot: Goodbye!")
        break

    # Tokenize user input with attention mask
    new_input = tokenizer(user_input + tokenizer.eos_token, return_tensors='pt')
    input_ids = new_input["input_ids"]
    attention_mask = new_input["attention_mask"]

    # Append to chat history
    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)
        attention_mask = torch.cat([
            torch.ones_like(chat_history_ids), attention_mask
        ], dim=-1)

    # Generate response
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    # Update chat history
    chat_history_ids = output

    # Decode and print reply
    reply = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Chatbot:", reply)
