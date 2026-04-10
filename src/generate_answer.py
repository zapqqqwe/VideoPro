import torch
import argparse
import json
import os
import re
import textwrap
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from retriever import Retrieval_Manager
from analysis import AnalysisManager
from video_utils import process_code
from video_utils import *
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0, help="Start index")
parser.add_argument("--end", type=int, default=100000, help="End index")
parser.add_argument("--output_file", type=str, default="output", help="Output file")
parser.add_argument("--input_file", type=str, default="input", help="Output file")
# CUDA_VISIBLE_DEVICES=4 python generate_answer.py --input_file output_v3/longvideobench_qwen3vl_sft_program_35.json --output_file output_v4/longvideobench_qwen3vl_sft_program_35.json
args = parser.parse_args()
video_dir=os.path.join(os.path.dirname(__file__), "..", "dataset", "LongVideoBench")
clip_save_folder = os.path.join(video_dir, "clips/10/")
retrieval = Retrieval_Manager(
    clip_save_folder=clip_save_folder,
    dataset_folder=video_dir,
   
)
retrieval.load_model_to_gpu(0)
# 初始化分析模块
analysis = AnalysisManager(retrieval=retrieval)

# 读取数据
with open(args.input_file, "r") as f:
    datas = json.load(f)
datas = sorted(datas, key=lambda x: x["id"])
datas=datas[:]
# 设置保存路径
output_path = args.output_file
cnt=0
with open(output_path, "w") as fw:
    for idx, data in tqdm(enumerate(datas), total=len(datas)):
        if "Based on the video and the question, I will use native videoLLM reasoning mode to solve this task." in data['output'][0]:
            data['pred']=data['native']
            fw.write(json.dumps(data, ensure_ascii=False) + "\n")
            fw.flush()
            continue
        if idx < args.start or idx >= args.end:
            continue
        video_path = os.path.join(video_dir, "videos", data["video_path"])
        question = data["question"]
        choices = data["candidates"]
        duration = data["duration"]
        output = data["output"][0]
        try:
            code_txt = re.search(r"<code>(.*?)</code>", output, flags=re.DOTALL).group(1)
            result = process_code(code_txt).replace('top_k=1','top_k=3')#.replace('top_k=2','top_k=3')
            exec(result)
            pred = execute_command(video_path, question, choices, duration)
        except Exception as e:
                data['error'] = str(e)
                pred=""    
        data["pred"] = pred
        fw.write(json.dumps(data, ensure_ascii=False) + "\n")
        fw.flush()

print(f"Results saved to {output_path}")


