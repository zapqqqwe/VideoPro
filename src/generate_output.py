
from openai import OpenAI
import time
import numpy as np
import re
import json
client = OpenAI(api_key="", base_url="http://0.0.0.0:8007/v1")
import os
def get_oai_chat_response(prompt,video_path, model='qwen3vl', n=1, temperature=0, max_tokens=200000, retries=0):
    
        
        print(prompt)
   

        messages = [{'role': 'user', 'content': [
    {'type': 'video', 'video': video_path},
    {'type': 'text', 'text': prompt}
]}]
       
        response = client.chat.completions.create(
            model=model,  
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
            logprobs=True,
            top_logprobs=5
        )
        
      
        output_text = response.choices[0].message.content
        confidence = 0.0
        for choice in response.choices:
            if hasattr(choice, "logprobs") and choice.logprobs and choice.logprobs.content:
                first_token_info = choice.logprobs.content[0]  # 第一个 token
                confidence = float(np.exp(first_token_info.logprob))  # logprob -> prob
                # print(
                #     f"First token: {first_token_info.token}, "
                #     f"LogProb: {first_token_info.logprob:.4f}, "
                #     f"Confidence: {confidence:.4f}"
                # )
                break
            else:
                print("⚠️ No logprobs found for this choice.")

        # # 8. 从 output_text 中抽取最终答案（沿用你原来的规则）
        match = re.match(r'^\s*([A-J]|Yes|No)\b', output_text.strip())
        final_answer = match.group(1) if match else ""

        valid_answers = [
            "A", "B", "C", "D", "E",
            "F", "G", "H", "I", "J",
            "Yes", "No"
        ]

        if final_answer not in valid_answers:
            final_answer = ""
        print(123,output_text,final_answer,confidence)
        # # 9. 把 score 直接用上面算出来的 confidence
        score = confidence if final_answer else 0.0
        return output_text,score

# import json
# with open('dataset/LongVideoBench/lvb_val.json') as f:
#     longvideo_datas=json.load(f)
# data_dir='dataset/LongVideoBench/'
# mapped_choice = {1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 0: 'A', 7: 'G'}
# print(len(longvideo_datas),longvideo_datas[0])

# import os
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import time

# def process_video_data(data, data_dir, mapped_choice):
#     video_path = os.path.join(data_dir, 'videos', data['video_path'])
#     question = data['question']
#     choices = data['candidates']
#     data['choices'] = choices
#     data['duration']=int(data['duration'])
#     data['answer'] = mapped_choice.get(data['correct_choice'])
#     duration = int(data['duration'])
#     letters = [chr(65 + i) for i in range(len(choices))]
#     choices_str = "\n".join(f"{L}. {c}" for L, c in zip(letters, choices))

#     # instruction = (
# #         f"""You will receive a multiple-choice question about a video.Your output must define a Python function in the following format::

# # <planning>
# # Briefly judge whether the question can be direct solved by videoLLM and plan the main API calls and reasoning steps you will use.
# #    - If it is can be answered,  use native mode with a single query_native call.
# #    - If it is needs long-range or multi-step reasoning, use more detailed visual program mode with other APIs (retrieval, subtitles, frame analysis, etc.).
# # </planning>

# # <code>
# # Write a Python function in the following format.

# # def execute_command(video_path, question, choices, duration):
# #     # Visual program code (no comments needed inside the code body).
# #     ...
# #     return answer
# # </code>\n"""


    
#     question_with_choices = f"Question: {question}\n Possible answer choices:\n{choices_str}"
#     prompt = f" Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D or other letter) of the correct option.\n{question_with_choices}\nOutput a single letter. The best answer is: "
#     # print(prompt)
#     try:
#         output = get_oai_chat_response(prompt,video_path)
#     except:
#          output=''
#     data['output'] = output
#     return data

# def process_all_data(longvideo_datas, data_dir, mapped_choice):
#     all_data = []
#     with ThreadPoolExecutor(max_workers=16) as executor:
#         futures = []
#         for data in longvideo_datas:
#             futures.append(executor.submit(process_video_data, data, data_dir, mapped_choice))
        
#         for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Videos"):
#             processed_data = future.result()
#             all_data.append(processed_data)

#     return all_data

# all_valid_data = process_all_data(longvideo_datas[:], data_dir, mapped_choice)

# with open('output_v2/longvideobench_qwen2.5vl_answer.json','w') as f:
#     json.dump(all_valid_data,f,indent=4)
# # len(all_valid_data)


import json
import os
from decord import VideoReader,cpu
def get_video_info(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = total_frames / fps if fps > 0 else 0
    return int(duration)
# with open('dataset/Video-MME/videomme/data.json') as f:
#     videomme_datas=f.readlines()
# data_dir='dataset/Video-MME/'
# print(len(videomme_datas),videomme_datas[0])
# data=json.loads(videomme_datas[0])
# video_path = os.path.join(data_dir, 'video', data['videoID']+'.mp4')
# question = data['question']
# choices = data['options']
# data['choices'] = choices
# duration = get_video_info(video_path)


# import os
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import time

# def process_video_data(data, data_dir, mapped_choice):
#     video_path =  os.path.join(data_dir, 'video', data['videoID']+'.mp4')
#     question = data['question']
#     choices = data['options']
#     data['choices'] = choices
#     duration = get_video_info(video_path)
#     data['duration']=int(duration)
   
#     letters = [chr(65 + i) for i in range(len(choices))]
#     choices_str = "\n".join(f"{c}" for L, c in zip(letters, choices))

#     # instruction = (
# #         f"""You will receive a multiple-choice question about a video.Your output must define a Python function in the following format::

# # <planning>
# # Briefly judge whether the question can be direct solved by videoLLM and plan the main API calls and reasoning steps you will use.
# #    - If it is can be answered,  use native mode with a single query_native call.
# #    - If it is needs long-range or multi-step reasoning, use more detailed visual program mode with other APIs (retrieval, subtitles, frame analysis, etc.).
# # </planning>

# # <code>
# # Write a Python function in the following format.

# # def execute_command(video_path, question, choices, duration):
# #     # Visual program code (no comments needed inside the code body).
# #     ...
# #     return answer
# # </code>\n"""


#     #     f"The video has a duration of {duration} seconds.\n"
#     #     f"Question: {question}\n"
#     #     f"Choices:\n{choices_str}\nOutput a single letter. Best option:"
#     # )
#     question_with_choices = f"Question: {question}\n Possible answer choices:\n{choices_str}"
#     prompt = f" Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D or other letter) of the correct option.\n{question_with_choices}\nOutput a single letter. The best answer is: "
#     try:
#         output = get_oai_chat_response(prompt,video_path)
#     except:
#         output=''
#     data['duration']=int(duration)
#     data['instruction'] = prompt
#     data['output'] = output
#     # print(data['output'])
#     return data

# def process_all_data(longvideo_datas, data_dir, mapped_choice):
#     all_data = []
#     with ThreadPoolExecutor(max_workers=16) as executor:
#         futures = []
#         for data in longvideo_datas:
#             data=json.loads(data)
#             futures.append(executor.submit(process_video_data, data, data_dir, mapped_choice))
        
#         for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Videos"):
#             processed_data = future.result()
#             all_data.append(processed_data)

#     return all_data

# all_valid_data = process_all_data(videomme_datas, data_dir, mapped_choice)
# # longvideobench_qwen2.5vl_answer.json
# with open('output_v2/videomme_qwen2.5vl_answer.json','w') as f:
#     json.dump(all_valid_data,f,indent=4)

# import json
# import os
# from decord import VideoReader,cpu
# def get_video_info(video_path):
#     vr = VideoReader(video_path, ctx=cpu(0))
#     total_frames = len(vr)
#     fps = vr.get_avg_fps()
#     duration = total_frames / fps if fps > 0 else 0
#     return int(duration)

# import re
# with open('dataset/LVBench/data/data.json') as f:
#     videomme_datas=f.readlines()
# data_dir='dataset/LVBench/'
# print(len(videomme_datas),videomme_datas[0])
# data=json.loads(videomme_datas[0])
# video_path = os.path.join(data_dir, 'video', data['key']+'.mp4')
# question = data['question']
# # choices = data['options']
# data['choices'] = re.findall(r"\([A-L]\)\s*(.+)", question)
# # duration = get_video_info(video_path)


# import os
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import time

# def process_video_data(data, data_dir):
#     video_path =  os.path.join(data_dir, 'video', data['key']+'.mp4')
#     question = data['question']
#     choices =  re.findall(r"\([A-L]\)\s*(.+)", question)
#     data['choices'] = choices
#     duration = int(get_video_info(video_path))
#     letters = [chr(65 + i) for i in range(len(choices))]
#     choices_str = "\n".join(f"{c}" for L, c in zip(letters, choices))
#     data['duration']=duration
# #     instruction = (
# # #         f"""You will receive a multiple-choice question about a video.Your output must define a Python function in the following format::

# # # <planning>
# # # Briefly judge whether the question can be direct solved by videoLLM and plan the main API calls and reasoning steps you will use.
# # #    - If it is can be answered,  use native mode with a single query_native call.
# # #    - If it is needs long-range or multi-step reasoning, use more detailed visual program mode with other APIs (retrieval, subtitles, frame analysis, etc.).
# # # </planning>

# # # <code>
# # # Write a Python function in the following format.

# # # def execute_command(video_path, question, choices, duration):
# # #     # Visual program code (no comments needed inside the code body).
# # #     ...
# # #     return answer
# # # </code>\n"""


# #         f"The video has a duration of {duration} seconds.\n"
# #         f"Question: {question}\nOutput a single letter. Best option:"
       
# #     )
# #     output = get_oai_chat_response(instruction,video_path)
#     question_with_choices = f"Question: {question}"
#     prompt = f" Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D or other letter) of the correct option.\n{question_with_choices}\nOutput a single letter. The best answer is: "
#     try:
#         output = get_oai_chat_response(prompt,video_path)
#     except:
#         output=''
#     data['duration']=int(duration)
#     # data['instruction'] = instruction
#     data['output'] =output
#     return data

# def process_all_data(longvideo_datas, data_dir):
#     all_data = []
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         futures = []
#         for data in longvideo_datas:
#             data=json.loads(data)
#             futures.append(executor.submit(process_video_data, data, data_dir))
        
#         for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Videos"):
#             processed_data = future.result()
#             all_data.append(processed_data)

#     return all_data

# all_valid_data = process_all_data(videomme_datas, data_dir)

# with open('output_v2/lvbench_qwen2.5vl_answer.json','w') as f:
#     json.dump(all_valid_data,f,indent=4)
# # len(all_valid_data)









import re
with open('dataset/MLVU_Test/test-ground-truth/test_mcq_gt.json') as f:
    videomme_datas=json.load(f)
data_dir="dataset/MLVU_Test/MLVU_Test"
print(len(videomme_datas),videomme_datas[0])
# data=json.loads(videomme_datas[0])
# video_path =  os.path.join(data_dir, 'video', data['video'])
# question = data['question']
# choices = data['options']
# duration = get_video_info(video_path)


import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def process_video_data(data, data_dir):
    video_path =  os.path.join(data_dir, 'video', data['video'])
    question = data['question']
    choices =  data['candidates']
    data['choices'] = data['candidates']
    duration = int(data['duration'])
    letters = [chr(65 + i) for i in range(len(choices))]
    choices_str = "\n".join(f"{L}. {c}" for L, c in zip(letters, choices))
    data['duration']=duration
#     instruction = (
# #         f"""You will receive a multiple-choice question about a video.Your output must define a Python function in the following format::
# # <planning>
# # Briefly judge whether the question can be direct solved by videoLLM and plan the main API calls and reasoning steps you will use.
# #    - If it is can be answered,  use native mode with a single query_native call.
# #    - If it is needs long-range or multi-step reasoning, use more detailed visual program mode with other APIs (retrieval, subtitles, frame analysis, etc.).
# # </planning>

# # <code>
# # Write a Python function in the following format.

# # def execute_command(video_path, question, choices, duration):
# #     # Visual program code (no comments needed inside the code body).
# #     ...
# #     return answer
# # </code>\n"""


#         f"The video has a duration of {duration} seconds.\n"
#         f"Question: {question}\nOutput a single letter. Best option:"
       
#     )
#     output = get_oai_chat_response(instruction,video_path)
    question_with_choices = f"Question: {question}\n Possible answer choices:\n{choices_str}"
    prompt = f" Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D or other letter) of the correct option.\n{question_with_choices}\nOutput a single letter. The best answer is: "
    try:
        output = get_oai_chat_response(prompt,video_path)
    except:
        output=''
    data['duration']=int(duration)
    # data['instruction'] = instruction
    data['output'] =output
    return data

def process_all_data(longvideo_datas, data_dir):
    all_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for data in longvideo_datas:
            # data=json.loads(data)
            futures.append(executor.submit(process_video_data, data, data_dir))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Videos"):
            processed_data = future.result()
            all_data.append(processed_data)

    return all_data

all_valid_data = process_all_data(videomme_datas, data_dir)

with open('output_v2/mlvu_qwen2.5vl_answer.json','w') as f:
    json.dump(all_valid_data,f,indent=4)