from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="http://0.0.0.0:8007/v1"
)


INSTRUCTION_TEMPLATE = """You will receive a multiple-choice question about a video.Your output must define a Python function in the following format::

<planning>
Briefly judge whether the question can be direct solved by videoLLM and plan the main API calls and reasoning steps you will use.
   - If it is can be answered,  use native mode with a single query_native call.
   - If it is needs long-range or multi-step reasoning, use more detailed visual program mode with other APIs (retrieval, subtitles, frame analysis, etc.).
</planning>

<code>
Write a Python function in the following format.

def execute_command(video_path, question, choices, duration):
    # Visual program code (no comments needed inside the code body).
    ...
    return answer
</code>
"""


def build_prompt(question, options):
    letters = [chr(65 + i) for i in range(len(options))]
    choices_str = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options))
    prompt = (
        f"{INSTRUCTION_TEMPLATE}\n\n"
        f"Question: {question}\n"
        f"Possible answer choices:\n{choices_str}\n"
    )
    return prompt


def infer_video_mcq(video_path, question, options, model="qwen3vl", max_tokens=200000):
    prompt = build_prompt(question, options)
    print(prompt)

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

    print("output_text:", output_text)

    return {
        "prompt": prompt,
        "output_text": output_text,
    }


if __name__ == "__main__":
    video_path = "/inspire/hdd/global_user/lichenglin-253208540324/VidePro/__Bchxr3ejw.mp4"
    question = "What is the person doing in the video?"
    options = [
        "Cooking in the kitchen",
        "Playing guitar",
        "Riding a bicycle",
        "Swimming in a pool",
    ]
    result = infer_video_mcq(
        video_path=video_path,
        question=question,
        options=options,
        model="qwen3vl",
    )

    # print("\n=== Final Result ===")
    # print(result)




# <code>
# def execute_command(video_path, question, choices, duration):
#     try:
#         intervals, clip_paths = get_informative_clips(
#             video_path,
#             "person doing an activity",
#             top_k=2,
#             total_duration=duration
#         )
#     except Exception:
#         intervals, clip_paths = [], []
#     frames = []
#     for clip in clip_paths:
#         try:
#             frames.extend(extract_frames(clip, num_frames=16))
#         except Exception:
#             pass
#     if not frames:
#         frames = extract_frames(video_path, num_frames=32)
#     person_frames = []
#     for frame in frames:
#         try:
#             boxes = detect_object(frame, "person")
#         except Exception:
#             boxes = []
#         if boxes:
#             person_frames.append(frame)
#     if not person_frames:
#         person_frames = frames
#     prompt = "What is the person doing in the video?"
#     answer = query_mc(person_frames, prompt, choices)
#     return answer
# </code>