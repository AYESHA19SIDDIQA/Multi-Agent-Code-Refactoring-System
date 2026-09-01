from langchain.prompts import ChatPromptTemplate
import json
from collections import defaultdict

from agents.model_config import get_llm

# --- Prompt Template ---
prompt_template = ChatPromptTemplate.from_template("""
[INST] <<SYS>>
You are a seasoned software critic and refactoring expert.  
Your task is to analyze an **entire source file**.  
Detect both **entity-level issues** (functions/classes) and **file-level issues** (overall design, cohesion, structure).  
Focus on code smells and **refactoring opportunities** that improve maintainability.
<</SYS>>

--- FILE CONTEXT ---
- File: {file}
- Total Entities: {entity_count}
- Total Lines: {total_lines}
- Imports: {all_imports}
- Dependencies: {all_dependencies}

--- ENTITY METRICS ---
{entity_metrics}

--- FULL SOURCE CODE ---
{all_code}

--- TASK ---
1. **Entity-Level Analysis (Local)**  
   For each function/class:  
   - Code smells (long methods, deep nesting, magic numbers, etc.)  
   - Refactoring opportunities (extract method, better naming, simplify logic, reduce duplication)  
   - Documentation/style issues  
   - Maintainability assessment  

2. **File-Level Analysis (Global)**  
   For the file as a whole:  
   - Cohesion and separation of concerns  
   - Module/file structure  
   - Imports and dependency usage  
   - Refactoring opportunities (splitting file, reorganizing responsibilities, improving dependency management)  
   - Cross-cutting code smells across entities  

3. **Summary Recommendation**  
   Conclude with a **global maintainability rating** (Low / Medium / High refactoring priority) and justify.

Return the answer in structured sections:  
- **Local Analysis**  
- **Global Analysis**  
- **Summary Recommendation**
[/INST]
""")

def run_grouped_critic(input_file, txt_output_file, model_tier=None):
    """File-level companion to critic.py's per-entity pass. Groups chunks
    by file, asks the model to critique each file as a whole, and writes
    the plain-text responses to txt_output_file (one section per file)."""
    llm = get_llm(model_tier)

    # --- Load parsed JSON ---
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # --- Group chunks by file ---
    files_data = defaultdict(list)
    for chunk in chunks:
        files_data[chunk["file"]].append(chunk)

    # --- Process each file ---
    all_responses_txt = []   # collect plain-text responses for txt file

    for file, entities in files_data.items():
        # Aggregate metrics
        total_lines = sum(ent["code_metrics"]["line_count"] for ent in entities)
        max_complexity = max(ent["code_metrics"]["cyclomatic_complexity"] for ent in entities)
        max_depth = max(ent["code_metrics"]["nesting_depth"] for ent in entities)
        missing_docstrings = [ent["name"] for ent in entities if not ent["code_metrics"].get("has_docstring", False)]
        all_magic_numbers = sum((ent["code_metrics"]["magic_numbers"] for ent in entities), [])
        all_imports = list({imp for ent in entities for imp in ent.get("imports", [])})
        all_dependencies = list({dep for ent in entities for dep in ent.get("dependencies", [])})

        # Build per-entity metrics summary
        entity_metrics_text = "\n".join(
            f"- {ent['type']} {ent['name']} | "
            f"Lines={ent['code_metrics']['line_count']}, "
            f"Complexity={ent['code_metrics']['cyclomatic_complexity']}, "
            f"Depth={ent['code_metrics']['nesting_depth']}, "
            f"Docstring={ent['code_metrics']['has_docstring']}, "
            f"MagicNumbers={ent['code_metrics']['magic_numbers']}"
            for ent in entities
        )

        # Reconstruct entire file source from all entities
        all_code = "\n\n".join(ent['code'] for ent in entities)

        # Format prompt
        messages = prompt_template.format_messages(
            file=file,
            entity_count=len(entities),
            total_lines=total_lines,
            all_imports=all_imports,
            all_dependencies=all_dependencies,
            entity_metrics=entity_metrics_text,
            all_code=all_code,
        )
        print(f"\nCritiquing {file} as a whole...\n")
        final_prompt = messages[0].content
        response = llm.create_completion(
            prompt=final_prompt,
            max_tokens=2048,
            temperature=0.7
        )

        llm_text = response["choices"][0]["text"].strip()
        print(llm_text)

        # Collect plain-text for .txt file
        all_responses_txt.append(f"===== {file} =====\n{llm_text}\n\n")

    # --- Save all responses to a single TXT ---
    with open(txt_output_file, "w", encoding="utf-8") as f:
        f.writelines(all_responses_txt)

    return all_responses_txt


if __name__ == "__main__":
    run_grouped_critic(
        "parsed files/Python-Speech-Recognition.json",
        "parsed files/Python-Speech-Recognition-llm-responses.txt",
    )
