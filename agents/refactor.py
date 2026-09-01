from langchain.prompts import ChatPromptTemplate
import json

from agents.model_config import get_llm

# === Refactor prompt template ===
refactor_prompt = ChatPromptTemplate.from_template("""
[INST] <<SYS>>
You are a senior software engineer specializing in clean code and refactoring.
Take the following function/class along with its critique and produce a **refactored version**.
Make sure the new code is:
- Cleaner and more modular
- Matches Python best practices (PEP8)
- Preserves the original functionality
- Easy to maintain and test
<</SYS>>

--- CONTEXT ---
- **File**: {file}
- **Entity Type**: {type}
- **Entity Name**: {name}
- **Original Critique**: {critique}

--- ORIGINAL CODE ---
{code}

--- TASK ---
Refactor this code. Only output the full improved code (no explanations).
[/INST]
""")

def run_refactor(input_file, output_file=None, model_tier=None):
    """Refactor every chunk in input_file that the Critic reviewed
    (i.e. has an llm_response). Writes chunk['refactored_code'] in
    place and saves to output_file (defaults to overwriting input_file,
    same convention as critic.py)."""
    output_file = output_file or input_file
    llm = get_llm(model_tier)

    # === Load the JSON file with critic responses ===
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # === Run through each chunk and refactor ===
    for chunk in chunks:
        if "llm_response" not in chunk:
            continue  # skip chunks not reviewed by critic

        messages = refactor_prompt.format_messages(
            file=chunk["file"],
            type=chunk["type"],
            name=chunk["name"],
            critique=chunk["llm_response"],
            code=chunk["code"],
        )

        prompt_text = messages[0].content
        print(f"\nRefactoring {chunk['name']}...\n")

        response = llm.create_completion(
            prompt=prompt_text,
            max_tokens=1024,  # more room for full function/class rewrite
            temperature=0.3   # lower temp for more deterministic, clean code
        )

        llm_text = response["choices"][0]["text"].strip()
        print(llm_text)

        # Save refactored code back into the chunk
        chunk["refactored_code"] = llm_text

    # === Save updated JSON with refactored code ===
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    return chunks


if __name__ == "__main__":
    run_refactor("parsed files/Python-Speech-Recognition.json")
