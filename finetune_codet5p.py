import os
import json
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer


def load_data(cache_data_path, tokenizer, data_num=-1, max_source_len=768, max_target_len=256):
    # Carica JSON 
    with open(cache_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)

    def preprocess(examples):
        inputs = tokenizer(
            examples["input"],
            max_length=max_source_len,
            padding="max_length",
            truncation=True
        )
        labels = tokenizer(
            examples["target"],
            max_length=max_target_len,
            padding="max_length",
            truncation=True
        )
        inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        return inputs

    tokenized = dataset.map(preprocess, batched=True)

    if data_num > 0:
        print(f"✂️  Limitazione del dataset a {data_num} esempi")
        tokenized = tokenized.select(range(data_num))

    return tokenized


def train_model(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # Carica tokenizer e registra il token custom <SEP>
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.add_tokens(["<SEP>"])

    # Carica modello e aggiorna embedding per i token aggiunti
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))

    print(f"✅ Modello e tokenizer caricati da {args.model_name}")

    # Prepara dataset
    dataset = load_data(
        cache_data_path=args.cache_data,
        tokenizer=tokenizer,
        data_num=args.data_num,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len
    )

    dataset = dataset.train_test_split(test_size=0.2)
    print(f"📊 Dataset diviso: {len(dataset['train'])} train - {len(dataset['test'])} test")

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_dir=os.path.join(args.save_dir, "logs"),
        logging_steps=10,
        report_to="none",
        fp16=args.fp16,
        save_total_limit=1,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    print("🧠 Inizio dell’addestramento...\n")
    trainer.train()
    print("\n✅ Addestramento completato!")

    # Salvataggio finale: modello + tokenizer
    output_dir = os.path.join(args.save_dir, "final_checkpoint")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Modello e tokenizer salvati in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Salesforce/codet5p-220m")
    parser.add_argument("--cache-data", type=str, default="train.json")
    parser.add_argument("--save-dir", type=str, default="codet5p_output")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--data-num", type=int, default=-1, help="Limita il numero di esempi da usare")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument('--max-source-len', default=320, type=int)
    parser.add_argument('--max-target-len', default=128, type=int)
    args = parser.parse_args()

    train_model(args)
