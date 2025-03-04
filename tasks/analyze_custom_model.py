from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# Define the model path (where you saved it using trainer.save_model)
model_path = "./ner_model"

# Load the model and tokenizer
model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Force the model to run on CPU
device = torch.device("cpu")
model.to(device)
model.eval()  # Set to evaluation mode


def read_conll_file(file_path):
    with open(file_path, "r") as f:
        content = f.read().strip()
        sentences = content.split("\n\n")
        data = []
        for sentence in sentences:
            tokens = sentence.split("\n")
            token_data = []
            for token in tokens:
                token_data.append(token.split())
            data.append(token_data)
    return data


def analyze_text(sentence):
    named_entities = extract_named_entities(sentence)
    print("Named Entities - Example 1:", named_entities)
    return named_entities


def extract_named_entities(sentence):
    train_data = read_conll_file('./resources/eng.train')

    label_list = sorted(list(set([token_data[3] for sentence in train_data for token_data in sentence])))
    # Tokenize input sentence
    tokenized_input = tokenizer(sentence, return_tensors="pt", truncation=True).to(model.device)

    # Get model outputs
    outputs = model(**tokenized_input)

    # Get predicted labels
    predicted_labels = outputs.logits.argmax(-1)[0].tolist()

    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])

    # Map predicted labels to their actual label names
    predicted_entities = [(token, label_list[label]) for token, label in zip(tokens, predicted_labels)]

    return predicted_entities

if __name__ == "__main__":
    analyze_text("Michael graduated from MIT in 2010. The MIT university is in Paris and USA")