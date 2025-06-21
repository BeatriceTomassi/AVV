import os
import json
import argparse
import numpy as np
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

def parse_execution_list(text):
    try:
        match = re.search(r"\[(.*?)\]", text)
        if not match:
            return []
        list_str = match.group(1)
        return [int(x.strip()) for x in list_str.split(",") if x.strip().isdigit()]
    except:
        return []

def extract_lines_n(input_text):
    match = re.search(r"lines:\s*(\d+)", input_text)
    return int(match.group(1)) if match else None

def pad_or_trim(pred_list, target_len):
    if len(pred_list) > target_len:
        return pred_list[:target_len]
    elif len(pred_list) < target_len:
        return pred_list + [0] * (target_len - len(pred_list))
    return pred_list

def evaluate(model_path, test_file, max_input_length=768, max_output_length=256, save_preds=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = data[:200]

    dataset = Dataset.from_list(data)
    inputs = [ex["input"] for ex in data]
    targets = [ex["target"] for ex in data]

    preds = []
    errors = []

    print(f"🔍 Inizio inferenza su {len(inputs)} esempi...")

    generation_config = GenerationConfig(
        max_length=max_output_length,
        do_sample=False,
        num_beams=2
    )

    for i, input_text in enumerate(inputs):
        encoded = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_input_length)
        output = model.generate(**encoded, generation_config=generation_config)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        pred_list = parse_execution_list(decoded)
        true_list = parse_execution_list(targets[i])

        # 👉 Estrai N da "lines: N"
        expected_len = extract_lines_n(input_text)
        if expected_len:
            pred_list = pad_or_trim(pred_list, expected_len)
            true_list = pad_or_trim(true_list, expected_len)

            abs_errors = [abs(p - t) for p, t in zip(pred_list, true_list)]
            mean_error = np.mean(abs_errors)
            errors.append(mean_error)
        else:
            mean_error = None
            errors.append(None)

        preds.append({
            "input": input_text,
            "target": targets[i],
            "prediction": decoded,
            "mae": mean_error
        })

        if i < 3:
            print(f"🔹 Esempio {i+1}")
            print("Input:", input_text)
            print("Target:", targets[i])
            print("Predizione:", decoded)
            print("MAE parziale:", mean_error, "\n")

    valid_errors = [e for e in errors if e is not None]
    final_mae = np.mean(valid_errors) if valid_errors else float("inf")

    print(f"\n✅ MAE totale su {len(valid_errors)} esempi validi: {final_mae:.3f}")

    if save_preds:
        with open("predictions.json", "w", encoding="utf-8") as f:
            json.dump(preds, f, indent=2)
        print("📁 Predizioni salvate in predictions.json")

    return final_mae

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="codet5p_output/final_checkpoint")
    parser.add_argument("--test-file", type=str, default="test.json")
    parser.add_argument("--save-preds", action="store_true")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        test_file=args.test_file,
        save_preds=args.save_preds
    )
