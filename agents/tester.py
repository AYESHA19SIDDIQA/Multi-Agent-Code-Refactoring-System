import ast
import json
import subprocess
from pathlib import Path


def check_syntax(code):
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"syntax error: {e}"


def check_interface(original_code, refactored_code, entity_type, entity_name):
    """Heuristic: the entity's name should still exist in the refactored
    code, and a function shouldn't have silently lost parameters."""
    try:
        orig_tree = ast.parse(original_code)
        new_tree = ast.parse(refactored_code)
    except SyntaxError as e:
        return False, f"refactored code does not parse: {e}"

    def find_entity(tree):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == entity_name:
                    return node
        return None

    orig_node = find_entity(orig_tree)
    new_node = find_entity(new_tree)

    if new_node is None:
        return False, f"'{entity_name}' no longer exists in the refactored code"

    if entity_type in ("FunctionDef", "AsyncFunctionDef") and orig_node is not None:
        orig_args = len(orig_node.args.args)
        new_args = len(new_node.args.args)
        if new_args < orig_args:
            return False, (
                f"argument count dropped from {orig_args} to {new_args} "
                f"-- likely to break existing callers"
            )

    return True, "syntax and interface checks passed"


def run_repo_test_suite(repo_path, timeout=120):
    """Best-effort: run pytest inside repo_path if it looks testable.
    Returns None if there's nothing to run, else a dict summary."""
    repo_path = Path(repo_path)
    if not repo_path.exists():
        return None
    has_tests = any(repo_path.rglob("test_*.py")) or any(repo_path.rglob("*_test.py"))
    if not has_tests:
        return None
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "-q", "--no-header"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        last_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        return {"returncode": result.returncode, "summary": last_line}
    except Exception as e:
        return {"returncode": None, "summary": f"pytest run failed: {e}"}


def run_tester(input_file, output_file=None, repo_path=None):
    """Attach a `test_result` dict to every chunk that has refactored_code."""
    output_file = output_file or input_file

    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    for chunk in chunks:
        if "refactored_code" not in chunk:
            continue

        syntax_ok, syntax_err = check_syntax(chunk["refactored_code"])
        if not syntax_ok:
            chunk["test_result"] = {"passed": False, "reason": syntax_err}
            print(f"[FAIL] {chunk['name']}: {syntax_err}")
            continue

        iface_ok, iface_msg = check_interface(
            chunk["code"], chunk["refactored_code"], chunk["type"], chunk["name"]
        )
        chunk["test_result"] = {"passed": iface_ok, "reason": iface_msg}
        print(f"[{'OK' if iface_ok else 'FAIL'}] {chunk['name']}: {iface_msg}")

    if repo_path:
        suite_result = run_repo_test_suite(repo_path)
        if suite_result is not None:
            print(f"Repo test suite result: {suite_result}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    return chunks


if __name__ == "__main__":
    run_tester("parsed files/Python-Speech-Recognition.json", repo_path="repo-clone")
