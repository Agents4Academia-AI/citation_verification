"""
Hello-world agent — your starting point.

A minimal agent: one LLM call with one tool. Read it, run it, then replace it
with what your team is actually building.

For richer examples, see:
  https://github.com/Agents4Academia-AI/example-agents
"""

import anthropic

client = anthropic.Anthropic()


# ── A toy tool the model can call ─────────────────────────────────────────────
def python_eval(expression: str) -> str:
    """Evaluate a Python expression. Toy example — eval is unsafe in real code."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


TOOLS = [
    {
        "name": "python_eval",
        "description": (
            "Evaluate a Python expression and return the result. "
            "Use for arithmetic or simple computations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python expression, e.g. '2 + 2'",
                },
            },
            "required": ["expression"],
        },
    },
]


# ── The agent loop ────────────────────────────────────────────────────────────
def run_agent(task: str, max_iterations: int = 5) -> str:
    messages = [{"role": "user", "content": task}]

    for _ in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            text = next((b.text for b in response.content if b.type == "text"), "")
            return text

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "python_eval":
                    result = python_eval(**block.input)
                    print(f"  [tool] python_eval({block.input}) -> {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
        messages.append({"role": "user", "content": tool_results})

    return "Hit max iterations."


if __name__ == "__main__":
    answer = run_agent("What is the 12th Fibonacci number? Use python_eval.")
    print("\nFinal answer:", answer)
