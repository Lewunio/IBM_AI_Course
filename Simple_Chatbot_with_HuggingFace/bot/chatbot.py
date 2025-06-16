from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = "facebook/blenderbot-400M-distill"
HISTORY = []

def chat(input_text):
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    history_string = "\n".join(HISTORY)

    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    outputs = model.generate(**inputs,max_length=60)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    HISTORY.append(input_text)
    HISTORY.append(response)

    return response

if __name__ == "__main__":
    print(chat("Hello, how are you doing?"))