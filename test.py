import os
import glob
import json
import re
import time
from typing import List, Dict
from tqdm import tqdm
from google import genai
from openai import OpenAI

client = OpenAI(api_key='',
                base_url='https://generativelanguage.googleapis.com/v1beta/openai/')


def clean_markdown(text: str) -> str:
    text = re.sub(r'^---[\s\S]*?---', '', text, flags=re.MULTILINE)
    text = re.sub(r'', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    chunks = []
    current_chunk = ""
    
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk += "\n\n" + para
            
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

SYSTEM_PROMPT = """
You are a Data Generation Expert for GitLab.
Task: Read the provided handbook text, generate high-quality instruction-tuning pairs in JSON format. The number of pairs is decided by the content.

Output Format (Strict JSON List):
[
  {
    "instruction": "The user's question or request",
    "input": "Optional context if needed, otherwise empty string",
    "output": "The detailed, correct answer based strictly on the text",
    "category": "Scenario" 
  }
]

Requirements:
1. **Scenario-based**: Do not just ask "What is X?". Instead, create a situation. E.g., "I am a manager and my employee wants to move to handle data. What is the process?"
2. **Compliance**: Focus on rules, "musts", and forbidden actions.
3. **Reasoning**: The output should contain the step-by-step logic found in the text.

If the text implies no useful rules, return an empty list [].
"""
def generate_qa_pairs(chunk: str) -> List[Dict]:

    if len(chunk) < 100: 
        return []

    try:
        response = client.chat.completions.create(
            model='gemini-2.5-flash',
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Handbook Content:\n{chunk}"}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        
        try:
            data = json.loads(content)
            print(data)
            if isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            print(f"Warning: Model produced invalid JSON. Skipping chunk.")
            return []
            
    except Exception as e:
        print(f"API Error: {e}")
        time.sleep(4) 
        return []

AUGMENT_PROMPT = """
You are a Data Augmentation Specialist.
I will provide you with a specific QA pair (Instruction + Output) derived from a company handbook.
Your task is to generate **3 new, distinct variations** of this interaction to improve model robustness.

**Input QA Pair:**
Instruction: {instruction}
Output: {output}

**Required Variations:**
1. **The "Frustrated/Urgent" User:** Rewrite the instruction as someone who is in a rush or slightly annoyed. Keep the output helpful but concise.
2. **The "Reverse/Negative" Case:** Ask about the consequences of NOT following the rule, or if there is an exception. (Adjust the output to answer this correctly based on the original context).
3. **The "Short/Keyword" Query:** A very short, search-engine style query (e.g., "expense limit approval"). The output should be direct.

**Output Format (Strict JSON List):**
[
  {{"instruction": "...", "output": "...", "category": "Urgent"}},
  {{"instruction": "...", "output": "...", "category": "Negative"}},
  {{"instruction": "...", "output": "...", "category": "Keyword"}}
]
"""

def augment_pair(original_pair):

    instruction = original_pair.get('instruction')
    output = original_pair.get('output')
    
    try:
        response = client.chat.completions.create(
            model='gemini-2.5-flash',
            messages=[
                {"role": "user", "content": AUGMENT_PROMPT.format(instruction=instruction, output=output)}
            ],
            temperature=0.8, 
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        new_pairs = json.loads(content)
        
        if isinstance(new_pairs, dict):
            for v in new_pairs.values():
                if isinstance(v, list): return v
            return []
        elif isinstance(new_pairs, list):
            return new_pairs
        return []
        
    except Exception as e:
        print(f"Augment Error: {e}")
        time.sleep(2)
        return []

    
def main():
    # STAGE 1 generate QA pairs
    input_dir = "./handbook-main/content/handbook/" 
    output_file = "gitlab_finetune_data.json"

    print("Scanning files...")
    files = glob.glob(os.path.join(input_dir, "**/*.md"), recursive=True)
    print(f"Found {len(files)} files.")

    files = files[:1] 
    
    all_dataset = []

    for file_path in tqdm(files, desc="Processing Files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception:
            continue

        # text prepressing
        clean_text = clean_markdown(text)
        chunks = chunk_text(clean_text)

        for chunk in chunks:
            # generate pairs
            pairs = generate_qa_pairs(chunk)

            # store source input
            for p in pairs:
                p['source'] = file_path
                
            all_dataset.extend(pairs)
            
            time.sleep(4)

    print(f"Generating complete. Total pairs: {len(all_dataset)}")


    # STAGE 2 augument data
    augmented_data = all_dataset[:]
    
    
    for item in tqdm(all_dataset, desc="Augmenting"):
        
        augmented_data.append(item)

        # data augment
        variations = augment_pair(item)
        
        for var in variations:
            var['source'] = item.get('source', 'unknown')
            augmented_data.append(var)
            

        time.sleep(4) 

    print(f"Augmentation complete. Total pairs: {len(augmented_data)} (Original: {len(all_dataset)})")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, indent=2, ensure_ascii=False)
        
    print(f"Saved to {output_file}")

    
if __name__ == "__main__":
    main()
