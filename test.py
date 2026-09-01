from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from llama_cpp import Llama
import os
import json



with open("parsed files/Python-Speech-Recognition.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

for chunk in chunks:
    llm_text = "1. Code Smells:\n* Long Method: The code in the `with` block is too long and complex, making it difficult to understand and maintain.\n* Data Clump: The `duration` and `timeout` arguments for `adjust_for_ambient_noise` and `listen` are repeated multiple times, which can lead to bugs if they are inconsistently used.\n* Magic Numbers: The `duration` and `timeout` arguments are used throughout the code without being declared as constants, making it difficult to change them without finding all occurrences in the code.\n* Low Cohesion: The code in the `with` block does not have a clear purpose, and the `recognize_google` function call is not immediately obvious from reading the code.\n* High Coupling: The code depends on too many external variables and functions, making it difficult to change the behavior without affecting other parts of the code.\n* Deep Nesting: The `try` block is nested within the `with` block, making it difficult to read and understand the code.\n2. Refactoring Opportunities:\n* Extract Method: Extract the code in the `with` block into a separate method to make it more readable and maintainable.\n* Reduce Nesting: Use a separate method to call `adjust_for_ambient_noise` and `listen` to reduce the nesting level and make the code more readable.\n* Improve Naming: Use more descriptive variable and function names to make the code more readable.\n* Decouple Responsibilities: Move the `recognize_google` call outside of the `with` block and into a separate method to decouple the code and make it easier to change the behavior.\n* Use Constants: Declare the `duration` and `timeout` arguments as constants to make it easier to change their values without finding all occurrences in the code.\n3. Documentation & Style:\n* Docstrings: None\n* Comments: None\n* Naming Conventions: Use uppercase for constants and descriptive variable names to make the code more readable.\n* Style Guides: Follow the PEP 8 style guide for Python code.\n4. Maintainability:\n* The code is not very readable, and the dependencies are not well-organized.\n* The"
    chunk["llm_response"] = llm_text.strip()
    break
with open("parsed files/Python-Speech-Recognition.json", "w") as f:
    json.dump(chunks, f, indent=2)
