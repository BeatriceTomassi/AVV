import json
import re
import subprocess
from statistics import mean
import multiprocessing
import gc
import os
import time

RESULTS_FILE = "results.jsonl"

def extract_code_input_lines(entry):
    """Estrai codice, input e numero di linee significative dal campo input del JSON"""
    raw = entry["input"]
    match = re.search(r'code:\s*(.*?)\s*<SEP>\s*input:\s*(.*?)\s*<SEP>\s*lines:\s*(\d+)', raw, re.DOTALL)
    if not match:
        raise ValueError("Formato input JSON non valido.")
    code, input_text, num_lines = match.groups()
    return code.strip(), input_text.strip(), int(num_lines)

def extract_target(entry):
    """Estrai la lista target di esecuzioni"""
    raw = entry["target"]
    match = re.search(r'execution:\s*\[([^\]]+)\]', raw)
    if not match:
        raise ValueError("Formato target JSON non valido.")
    return list(map(int, match.group(1).split(',')))

def build_prompt(code, input_text, num_lines):
    """Costruisci il prompt da dare in input al modello"""
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
    if len(pred) != len(target):
        return float('inf')  # penalizza predizioni con lunghezza errata
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
    total_mae = 0
    count = 0

    for i, entry in enumerate(data):
        if i in completed:
            print(f"[{i}] ✅ Già completato.")
            continue
        try:
            code, input_text, num_lines = extract_code_input_lines(entry)
            target = extract_target(entry)
            prompt = build_prompt(code, input_text, num_lines)
            prediction = call_ollama(prompt)

            # ✅ Qui metti il controllo per scartare predizioni sbagliate
            if prediction is None or len(prediction) != len(target):
                print(f"[{i}] ❌ Predizione non valida o lunghezza errata. Pred: {prediction}, Target: {target}")
                continue

            mae = mean_absolute_error(prediction, target)
            result_entry = {
                "index": i,
                "mae": mae,
                "prediction": prediction,
                "target": target
            }

            
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(result_entry) + "\n")

            print(f"[{i}] ✅ MAE: {mae:.2f} | pred: {prediction} | target: {target}")
            total_mae += mae
            count += 1

        except Exception as e:
            print(f"[{i}] ⚠️ Errore: {e}")
            continue

        gc.collect()
        time.sleep(1)

    if count > 0:
        print(f"\n📊 Media MAE (solo nuovi) su {count} esempi: {total_mae / count:.3f}")
    else:
        print("\n⚠️ Nessun nuovo esempio processato.")

    print(f"\nMedia MAE su {count} esempi validi: {total_mae / count:.3f}")

if __name__ == "__main__":
    main()
