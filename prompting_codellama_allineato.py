import json
import re
import subprocess
from statistics import mean
import multiprocessing
import gc
import os
import time

RESULTS_FILE = "results.jsonl"

def pad_or_trim(lst, target_len):
    if len(lst) > target_len:
        return lst[:target_len]
    elif len(lst) < target_len:
        return lst + [0] * (target_len - len(lst))
    return lst

def extract_code_input_lines(entry):
    raw = entry["input"]
    match = re.search(r'code:\s*(.*?)\s*<SEP>\s*input:\s*(.*?)\s*<SEP>\s*lines:\s*(\d+)', raw, re.DOTALL)
    if not match:
        raise ValueError("Formato input JSON non valido.")
    code, input_text, num_lines = match.groups()
    return code.strip(), input_text.strip(), int(num_lines)

def extract_target(entry):
    raw = entry["target"]
    match = re.search(r'execution:\s*\[([^\]]+)\]', raw)
    if not match:
        raise ValueError("Formato target JSON non valido.")
    return list(map(int, match.group(1).split(',')))

def build_prompt(code, input_text, num_lines):
    prompt = f"""
You are a C static analysis assistant.  
Given a C program and its input, return a Python list of integers.  
Each number must represent how many times the corresponding line of code is executed during runtime.  
Count each code line once per execution.  
The output must be only the list. No explanation. No extra text.

Follow this format strictly:

Example 1:

C code:
int main() {{
    int i;
    scanf("%d", &x);
    for (i = 0; i < 3; i++) {{
        printf("Loop %d\\n", i);
    }}
    printf("Done\\n");
    return 0;
}}

Input:
3

There are 7 code lines to evaluate. The output must be a list of 7 integers.

Output:
[1, 1, 1, 4, 3, 1, 1]

---

Example 2:

C code:
int main() {{
    printf("Hello");
    return 0;
}}

Input:
(none)

There are 3 code lines to evaluate. The output must be a list of 3 integers.

Output:
[1, 1, 1]

---

Now do the same for this program:

C code:
{code}

Input:
{input_text if input_text else '(none)'}

There are {num_lines} code lines to evaluate. The output must be a list of {num_lines} integers.

Output:
""".strip()
    return prompt

def _call_ollama_inner(prompt):
    result = subprocess.run(
        ["ollama", "run", "codellama:7b-instruct"],
        input=prompt.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60
    )
    output = result.stdout.decode().strip()
    match = re.search(r'\[.*?\]', output)
    if not match:
        return None
    try:
        return list(map(int, match.group(0).strip("[]").split(",")))
    except:
        return None

def call_ollama(prompt):
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        try:
            result = pool.apply(_call_ollama_inner, (prompt,))
            return result
        except:
            return None

def mean_absolute_error(pred, target):
    if not pred or not target or len(pred) != len(target):
        return float('inf')
    return mean(abs(p - t) for p, t in zip(pred, target))

def load_completed_indexes():
    if not os.path.exists(RESULTS_FILE):
        return set()
    with open(RESULTS_FILE, "r") as f:
        return {int(json.loads(line)["index"]) for line in f}

def main():
    with open("test.json", "r") as f:
        data = json.load(f)

    completed = load_completed_indexes()
    total_mae_raw = 0
    count_raw = 0
    total_mae_aligned = 0
    count_aligned = 0

    for i, entry in enumerate(data):
        if i in completed:
            print(f"[{i}] ✅ Già completato.")
            continue
        try:
            code, input_text, num_lines = extract_code_input_lines(entry)
            target = extract_target(entry)
            prompt = build_prompt(code, input_text, num_lines)
            prediction = call_ollama(prompt)

            if prediction is None:
                print(f"[{i}] ❌ Nessuna predizione.")
                continue

            
            # MAE grezzo (senza adattamento)
            mae_raw = mean_absolute_error(prediction, target)

            # MAE allineato (padding/trimming)
            pred_aligned = pad_or_trim(prediction, len(target))
            mae_aligned = mean_absolute_error(pred_aligned, target)

            # Salvataggio dei risultati
            result_entry = {
                "index": i,
                "prediction": prediction,
                "target": target,
                "mae_raw": mae_raw,
                "mae_aligned": mae_aligned
            }

            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(result_entry) + "\n")

            # Log su console
            print(f"[{i}] ✅ MAE grezzo: {mae_raw:.2f} | MAE allineato: {mae_aligned:.2f}")

            if mae_raw != float("inf"):
                total_mae_raw += mae_raw
                count_raw += 1
            if mae_aligned != float("inf"):
                total_mae_aligned += mae_aligned
                count_aligned += 1

        except Exception as e:
            print(f"[{i}] ⚠️ Errore: {e}")
            continue

        gc.collect()
        time.sleep(1)

    # Report finale
    if count_raw > 0:
        print(f"\n📊 MAE grezzo medio su {count_raw} esempi: {total_mae_raw / count_raw:.3f}")
    else:
        print("\n⚠️ Nessun MAE grezzo valido.")

    if count_aligned > 0:
        print(f"📏 MAE allineato medio su {count_aligned} esempi: {total_mae_aligned / count_aligned:.3f}")
    else:
        print("\n⚠️ Nessun MAE allineato valido.")

if __name__ == "__main__":
    main()
