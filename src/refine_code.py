from openai import OpenAI
import re

client = OpenAI(
    api_key="",
    base_url="http://0.0.0.0:8007/v1"
)


def build_question_with_choices(question, choices):
    letters = [chr(65 + i) for i in range(len(choices))]
    choices_str = "\n".join(f"{L}. {c}" for L, c in zip(letters, choices))
    return f"Question: {question}\nPossible answer choices:\n{choices_str}"


def get_oai_chat_response(prompt, video_path, model="qwen3vl", max_tokens=32000):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
    )

    output_text = response.choices[0].message.content if response.choices else ""
    print("prompt:\n", prompt)
    print("output_text:\n", output_text)
    return output_text


def extract_refined_code(output_text):
    if not output_text:
        return ""

    output_text = output_text.strip()

    m = re.search(r"<code>\s*(.*?)\s*</code>", output_text, flags=re.S)
    if m:
        return m.group(1).strip()

    m = re.search(r"```python\s*(.*?)\s*```", output_text, flags=re.S)
    if m:
        return m.group(1).strip()

    m = re.search(
        r"def\s+execute_command\s*\(\s*video_path\s*,\s*question\s*,\s*choices\s*,\s*duration\s*\)\s*:.*",
        output_text,
        flags=re.S,
    )
    if m:
        return m.group(0).strip()

    return output_text


def build_refine_prompt(question, choices, current_code, error_log=None):
    question_with_choices = build_question_with_choices(question, choices)

    if "query_native" in current_code:
        ot = """
def execute_command(video_path, question, choices, duration):
    return query_native(video_path, question, choices)
""".strip()

        refine_prompt1 = """
You will receive a multiple-choice question about a video and an existing
visual program that only uses the native-mode helper API query_native.

{}

Current native visual program:
{}

Refine this visual program.
""".strip()

        prompt = refine_prompt1.format(question_with_choices, ot) + "\n<code>\ndef execute_command"
        prompt_type = "native"

    elif error_log is not None and str(error_log).strip():
        refine_prompt2 = """
You will receive a multiple-choice question about a video and a Python visual program in the
execute_command format, and a runtime error log from running this program.

{}

Buggy visual program:
{}

Runtime error log:
{}

Refine this visual program by fixing the bugs.
""".strip()

        prompt = refine_prompt2.format(
            question_with_choices,
            current_code,
            error_log
        ) + "\n<code>\ndef execute_command"
        prompt_type = "bug_fix"

    else:
        refine_prompt3 = """
You will receive a multiple-choice question about a video and an existing
visual program.

{}

Current visual program:
{}

Refine this visual program to improve its reasoning and correctness.
""".strip()

        prompt = refine_prompt3.format(
            question_with_choices,
            current_code
        ) + "\n<code>\ndef execute_command"
        prompt_type = "general"

    return prompt, prompt_type


def refine_code(video_path, question, choices, current_code, error_log=None, model="qwen3vl", max_tokens=32000):
    prompt, prompt_type = build_refine_prompt(
        question=question,
        choices=choices,
        current_code=current_code,
        error_log=error_log,
    )

    output_text = get_oai_chat_response(
        prompt=prompt,
        video_path=video_path,
        model=model,
        max_tokens=max_tokens,
    )

    refined_code = extract_refined_code(output_text)

    return {
        "prompt_type": prompt_type,
        "prompt": prompt,
        "output_text": output_text,
        "refined_code": refined_code,
    }


if __name__ == "__main__":
    video_path = "/inspire/hdd/global_user/lichenglin-253208540324/VidePro/__Bchxr3ejw.mp4"
    question = "What is the person doing in the video?"
    choices = [
        "Cooking in the kitchen",
        "Playing guitar",
        "Riding a bicycle",
        "Swimming in a pool",
    ]

    current_code = """
def execute_command(video_path, question, choices, duration):
    return query_native(video_path, question, choices)
""".strip()

    error_log = None

    result = refine_code(
        video_path=video_path,
        question=question,
        choices=choices,
        current_code=current_code,
        error_log=error_log,
        model="qwen3vl",
        max_tokens=32000,
    )

    print("\n=== prompt_type ===")
    print(result["prompt_type"])

    print("\n=== refined_code ===")
    print(result["refined_code"])
