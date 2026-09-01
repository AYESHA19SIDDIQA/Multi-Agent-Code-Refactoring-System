import json
from pathlib import Path

from langchain.prompts import ChatPromptTemplate

from agents.model_config import get_llm

review_prompt = ChatPromptTemplate.from_template("""
[INST] <<SYS>>
You are the final reviewer on a code refactoring team. You receive the
original code, a critique of it, a proposed refactor, and an automated
test result. Decide whether the refactor should be ACCEPTED, sent back
for REVISION, or REJECTED.
<</SYS>>

--- ENTITY ---
File: {file}
Type: {type}
Name: {name}

--- ORIGINAL CODE ---
{original_code}

--- CRITIQUE ---
{critique}

--- REFACTORED CODE ---
{refactored_code}

--- AUTOMATED TEST RESULT ---
{test_result}

--- TASK ---
Respond with exactly one verdict word on the first line -- ACCEPT,
REVISE, or REJECT -- followed by a short (2-4 sentence) rationale on
the next line. Reject if functionality was clearly changed. Ask for
revision if the refactor is only partially satisfactory. Accept only
if the refactor is clean, correct, and preserves behaviour.
[/INST]
""")


def _parse_verdict(llm_text):
    lines = [l.strip() for l in llm_text.strip().splitlines() if l.strip()]
    if not lines:
        return "REVISE", "empty response from reviewer model"
    verdict = lines[0].upper().strip(":*# ")
    if verdict not in ("ACCEPT", "REVISE", "REJECT"):
        verdict = "REVISE"
    rationale = " ".join(lines[1:]) if len(lines) > 1 else "no rationale given"
    return verdict, rationale


def run_reviewer(input_file, output_file=None, report_file=None, model_tier=None):
    output_file = output_file or input_file
    llm = get_llm(model_tier)

    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    for chunk in chunks:
        if "refactored_code" not in chunk:
            continue

        test_result = chunk.get("test_result", {"passed": True, "reason": "not tested"})
        if not test_result.get("passed", False):
            chunk["review_verdict"] = "REJECT"
            chunk["review_rationale"] = f"failed automated tests: {test_result.get('reason')}"
            print(f"[REJECT] {chunk['name']}: {chunk['review_rationale']}")
            continue

        messages = review_prompt.format_messages(
            file=chunk["file"],
            type=chunk["type"],
            name=chunk["name"],
            original_code=chunk["code"],
            critique=chunk.get("llm_response", "(no critique available)"),
            refactored_code=chunk["refactored_code"],
            test_result=test_result.get("reason", ""),
        )
        prompt_text = messages[0].content
        print(f"\nReviewing {chunk['name']}...\n")
        response = llm.create_completion(prompt=prompt_text, max_tokens=256, temperature=0.3)
        llm_text = response["choices"][0]["text"]
        verdict, rationale = _parse_verdict(llm_text)
        chunk["review_verdict"] = verdict
        chunk["review_rationale"] = rationale
        print(f"  -> {verdict}: {rationale}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    if report_file:
        write_markdown_report(chunks, report_file)

    return chunks


def write_markdown_report(chunks, report_file):
    reviewed = [c for c in chunks if "review_verdict" in c]
    accepted = sum(1 for c in reviewed if c["review_verdict"] == "ACCEPT")
    revise = sum(1 for c in reviewed if c["review_verdict"] == "REVISE")
    rejected = sum(1 for c in reviewed if c["review_verdict"] == "REJECT")

    lines = ["# Refactoring Report", ""]
    lines.append(f"Entities reviewed: **{len(reviewed)}**  ")
    lines.append(f"Accepted: **{accepted}** | Needs revision: **{revise}** | Rejected: **{rejected}**")
    lines.append("")

    for chunk in reviewed:
        metrics = chunk.get("code_metrics", {})
        lines.append(f"## `{chunk['file']}` :: {chunk['name']} ({chunk['type']})")
        lines.append(f"- Verdict: **{chunk['review_verdict']}** -- {chunk['review_rationale']}")
        lines.append(
            f"- Complexity: {metrics.get('cyclomatic_complexity', '?')} | "
            f"Lines: {metrics.get('line_count', '?')} | "
            f"Nesting: {metrics.get('nesting_depth', '?')}"
        )
        lines.append("")

    Path(report_file).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {report_file}")


if __name__ == "__main__":
    run_reviewer(
        "parsed files/Python-Speech-Recognition.json",
        report_file="parsed files/Python-Speech-Recognition-report.md",
    )
