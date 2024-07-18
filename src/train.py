import argparse

import settings
import torch
from dataset import TextDataset, new_tokenizer
from model import Mamba, ModelArgs
from torch.utils.data import DataLoader
from transformers import get_scheduler

"""
mamba-370m :

        d_model: 1024
        n_layer: 48
        vocab_size: 50280
        d_state: 4096
        expand: 4
        d_conv: 4



mamba-130m :

        d_model: 768
        n_layer: 24
        vocab_size: 50280
        d_state: 3072
        expand: 4
        d_conv: 4
"""


def train(
    pre_train=True,
    finetune=False,
    pretrained_model_name="state-spaces/mamba-130m",
    n_epochs=100,
):
    if pre_train is True:
        # model args
        args = ModelArgs(
            d_model=768,  # Hidden dimension size
            n_layer=24,  # Number of layers
            vocab_size=settings.VOCAB_SIZE,  # Vocabulary size
            d_state=3072,  # Latent state dimension
            expand=4,  # Expansion factor
            dt_rank="auto",  # Rank of delta
            d_conv=4,  # Convolution kernel size
            pad_vocab_size_multiple=8,
            conv_bias=True,
            bias=False,
        )
        model = Mamba(args)
    if finetune:
        model = Mamba.from_pretrained(pretrained_model_name)

    # Training Args
    class Args:
        dataset_path = settings.TRAIN_DATA_PATH  # Path to your raw text file
        eval_path = settings.EVAL_DATA_PATH
        lr = 1e-4
        epochs = n_epochs
        context_len = 384
        train_batch_size = 8
        valid_batch_size = 8
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_dataset = TextDataset(
        Args.dataset_path, new_tokenizer, context_len=Args.context_len
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=Args.train_batch_size, shuffle=True
    )

    eval_dataset = TextDataset(
        Args.eval_path, new_tokenizer, context_len=Args.context_len
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=Args.valid_batch_size, shuffle=False
    )

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=Args.lr)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * Args.epochs,
    )
    model.to(Args.device)
    for epoch in range(Args.epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            batch = {k: v.to(Args.device) for k, v in batch.items()}

            outputs = model(batch["input_ids"])

            # Compute the loss manually
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{Args.epochs}, Loss: {total_loss / len(train_dataloader)}"
        )
        # Evaluation
        model.eval()
        total_eval_loss = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(Args.device) for k, v in batch.items()}

                outputs = model(batch["input_ids"])

                # Compute the loss manually
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

                total_eval_loss += loss.item()

        avg_eval_loss = total_eval_loss / len(eval_dataloader)
        print(f"Epoch {epoch+1}/{Args.epochs}, Evaluation Loss: {avg_eval_loss}")
        model_save_path = "mamba_darija.pt"
        torch.save(model.state_dict(), model_save_path)
        print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-train or fine-tune a Mamba model."
    )
    parser.add_argument(
        "--pre_train",
        type=bool,
        default=True,
        help="Set to True to pre-train a new model.",
    )
    parser.add_argument(
        "--finetune",
        type=bool,
        default=False,
        help="Set to True to fine-tune a pretrained model.",
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="state-spaces/mamba-130m",
        help="Name of the pretrained model to load.",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=100, help="Number of epochs for training."
    )

    args = parser.parse_args()

    train(
        pre_train=args.pre_train,
        finetune=args.finetune,
        pretrained_model_name=args.pretrained_model_name,
        n_epochs=args.n_epochs,
    )
