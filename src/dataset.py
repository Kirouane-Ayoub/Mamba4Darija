import settings
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# Function to get training data
def get_training_corpus():
    dataset = load_dataset("text", data_files={"train": settings.TRAIN_DATA_PATH})
    for i in range(0, len(dataset["train"]), 1000):
        yield dataset["train"][i : i + 1000]["text"]


# Initialize the base tokenizer
base_tokenizer = AutoTokenizer.from_pretrained(settings.TOKENIZER_NAME)

# Train the new tokenizer
new_tokenizer = base_tokenizer.train_new_from_iterator(
    get_training_corpus(), vocab_size=settings.VOCAB_SIZE
)

# Save the new tokenizer
new_tokenizer.save_pretrained("darija_tokenizer")

new_tokenizer.pad_token = new_tokenizer.eos_token

fim_prefix_token = "<fim_prefix>"
fim_middle_token = "<fim_middle_token>"
fim_suffix_token = "<fim_suffix_token>"
fim_pad_token = "<fim_pad>"

# Get the FIM-specific tokens and get their token ids
new_tokenizer.add_tokens(
    [
        fim_prefix_token,
        fim_middle_token,
        fim_middle_token,
        fim_pad_token,
    ]
)
prefix_tok_id = new_tokenizer.convert_tokens_to_ids(fim_prefix_token)
middle_tok_id = new_tokenizer.convert_tokens_to_ids(fim_middle_token)
suffix_tok_id = new_tokenizer.convert_tokens_to_ids(fim_middle_token)
pad_tok_id = None

fim_tokens = [prefix_tok_id, middle_tok_id, suffix_tok_id]


# If truncate_or_pad is on, also get pad token id
truncate_or_pad = True
if truncate_or_pad:
    pad_tok_id = new_tokenizer.convert_tokens_to_ids(fim_pad_token)
    fim_tokens.append(pad_tok_id)


# Custom Dataset Class
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, context_len=384):
        self.tokenizer = tokenizer
        self.context_len = context_len

        # Load and tokenize data
        with open(file_path, encoding="utf-8") as f:
            self.data = f.read()

        self.tokens = tokenizer(
            self.data,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=context_len,
        )

    def __len__(self):
        return len(self.tokens["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokens["input_ids"][idx],
            "labels": self.tokens["input_ids"][idx],
        }
