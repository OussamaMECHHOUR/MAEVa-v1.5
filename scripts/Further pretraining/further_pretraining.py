import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class StreamingTextDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                sentence = line.strip()
                if sentence:
                    encoded = self.tokenizer(
                        sentence,
                        max_length=self.max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    yield {
                        "input_ids": encoded["input_ids"].squeeze(),
                        "attention_mask": encoded["attention_mask"].squeeze(),
                    }


def pretrain(args):
    tokenizer = BertTokenizer.from_pretrained(args.checkpoint)
    model = BertForMaskedLM.from_pretrained(args.checkpoint).to(device)

    dataset = StreamingTextDataset(args.corpus_path, tokenizer)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    model.train()

    total_lines = sum(1 for _ in open(args.corpus_path, "r", encoding="utf-8"))
    total_steps = (total_lines // args.batch_size) * args.epochs

    with tqdm(total=total_steps, desc="Training Progress", unit="batch") as bar:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            optimizer.zero_grad()

            for step, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / args.accumulation_steps
                loss.backward()

                if (step + 1) % args.accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                bar.update(1)
                bar.set_postfix({"Loss": loss.item(), "Epoch": epoch + 1})

    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"\nFinal model and tokenizer saved to: {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Further pretraining BERT/AgriBERT on the masked language modeling (MLM) task.")

    parser.add_argument("--corpus_path", default="datasets/corpora/Corpus (PMC).txt", help="Path to the TXT corpus file to be used for further pretraining")
    parser.add_argument("--checkpoint", default="recobo/agriculture-bert-uncased", help="Pretrained BERT or AgriBERT checkpoints ('bert-base-uncased' or 'recobo/agriculture-bert-uncased')")
    parser.add_argument("--save_dir", default="scripts/Further pretraining/saved models", help="Directory path to save the final further pretrained model weights and tokenizer")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per step")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--mlm_prob", type=float, default=0.15, help="MLM masking probability")

    args = parser.parse_args()
    pretrain(args)
