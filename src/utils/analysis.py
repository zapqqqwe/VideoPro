import tempfile
import os
import sympy as sp
from pathlib import Path
from typing  import Any, List, Dict, Tuple, Sequence
import numpy as np
import torch
from PIL import Image
import cv2
import math
from typing import Any, List
from pathlib import Path
from io import BytesIO
import math
from PIL import Image
import torch
import numpy as np
from PIL import Image
import cv2
import requests
import os
import math
import uuid
import shutil
from pathlib import Path
from typing import List, Tuple, Callable, Any

from PIL import Image
def build_messages_with_local_jpg(
    frames: List[Image.Image],
    question: str,
    sample_k: int = 64,
    max_side: int = 256,
    max_pixels: int = 500 * 300,
    images_root: str = "images",
    use_file_url: bool = True,   # True: file:///abs/path.jpg  False: /abs/path.jpg
    jpeg_quality: int = 95,
) -> Tuple[List[dict], str]:
    """
    1) 计算统一 base_size (<=max_side & <=max_pixels)
    2) resize 所有帧到 base_size（拉伸，不保比例，与原逻辑一致）
    3) 保存到 ./images/<run_id>/*.jpg
    4) 均匀抽样最多 sample_k 帧
    5) 生成 messages（image_url 使用 file://... 或直接 path）
    返回: (messages, run_dir)
    """

    # --- 0. 生成唯一目录 ---
    run_id = uuid.uuid4().hex[:12]
    out_dir = Path(images_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. 计算 base_size（沿用你的逻辑） ---
    max_w, max_h = 0, 0
    for frame in frames:
        w, h = frame.size
        max_w = max(max_w, w)
        max_h = max(max_h, h)

    # 先把尺寸夹到不超过 256
    max_w = min(max_w, 500)
    max_h = min(max_h, 300)
    
    # 再保证像素不超过 256*256（必要时按比例缩小）
    pixels = max_w * max_h
    if pixels > max_pixels:
        scale = math.sqrt(max_pixels / float(pixels))
        max_w = max(1, int(max_w * scale))
        max_h = max(1, int(max_h * scale))
    max_w=max(max_w,224)
    max_h=max(max_h,224)
    base_size = (max_w, max_h)
    
    # --- 2. resize 并保存 jpg ---
    saved_paths: List[Path] = []
    for idx, frame in enumerate(frames):
        if frame.mode != "RGB":
            frame = frame.convert("RGB")

        if frame.size != base_size:
            frame = frame.resize(base_size, Image.LANCZOS)

        jpg_path = out_dir / f"{idx:06d}.jpg"
        frame.save(jpg_path, format="JPEG", quality=jpeg_quality, optimize=True)
        saved_paths.append(jpg_path)

    # --- 3. 均匀抽样到 sample_k 帧（沿用你的逻辑但修一下边界） ---
    selected_paths = saved_paths
    n = len(selected_paths)

    if n > 0 and sample_k is not None and n > sample_k:
        step = n / float(sample_k)
        indices = [int(i * step) for i in range(sample_k)]
        # 去重 + 防越界
        indices = sorted(set(min(n - 1, i) for i in indices))
        selected_paths = [selected_paths[i] for i in indices]

    # --- 4. 构造 messages ---
    if len(selected_paths) == 0:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            }
        ]
        return messages #, str(out_dir)

    content = []
    for p in selected_paths:
        abs_path = str(p.resolve())
        url = f"{abs_path}" if use_file_url else abs_path
        content.append(
            {
                "type": "image",
                "image":  url,
            }
        )

    content.append(
        {
            "type": "text",
            "text": "Based the frames. " + question,
        }
    )

    messages = [{"role": "user", "content": content}]
    return messages

from openai import OpenAI
from transformers import (
    AutoProcessor,CLIPProcessor, CLIPModel,
    AutoProcessor, AutoModelForZeroShotObjectDetection
)



import torch.nn.functional as F
from io import BytesIO
import easyocr
from moviepy import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
from video_utils import *
from decord import VideoReader, cpu
from transformers import (

    
    AutoProcessor,CLIPProcessor, CLIPModel,
    AutoProcessor, AutoModelForZeroShotObjectDetection
)
import re
import tempfile
from torchvision.ops import box_convert
import supervision as sv
import numpy.typing as npt
# from paddleocr import PaddleOCR
# from sam2.build_sam import build_sam2_video_predictor
from groundingdino.util.inference import load_model as gdino_load_model
from groundingdino.util.inference import load_image as gdino_load_image
from groundingdino.util.inference import predict as gdino_predict
from torchvision.ops import box_convert
import supervision as sv

# from sam2.build_sam import build_sam2_video_predictor
from groundingdino.util.inference import (
    load_model as gdino_load_model,
    load_image as gdino_load_image,
    predict as gdino_predict,
)



def _to_image(obj: Any):
    """
    Return PIL.Image ONLY if mode == 'RGB'.
    If not RGB, return None.
    """

    img: Image.Image

    # 1. PIL Image
    if isinstance(obj, Image.Image):
        img = obj

    # 2. path
    elif isinstance(obj, (str, Path)):
        img = Image.open(obj)

    # 3. numpy array (OpenCV BGR)
    elif isinstance(obj, np.ndarray):
        if obj.ndim != 3 or obj.shape[2] != 3:
            return None
        img = Image.fromarray(cv2.cvtColor(obj, cv2.COLOR_BGR2RGB))

    # 4. bytes / bytearray (e.g. requests.raw)
    elif isinstance(obj, (bytes, bytearray)):
        img = Image.open(BytesIO(obj))

    else:
        return None

    # ❗关键点：只接受原生 RGB
    if img.mode != "RGB":
        return None

    return img

class AnalysisManager:
    def __init__(self, device_qwen="0",retrieval=None):
        self.device_track = "cuda:0"
        self.dtype = torch.float16
        # self.qwen_model = None
        self.device_qwen=device_qwen
        assert retrieval!=None
        self.retrieval=retrieval
        self.llm = OpenAI(
            api_key='',
            base_url=f"http://localhost:8007/v1",
        )
        
        self.model_name = 'Qwen2.5-7B-Instruct'
        
        # Grounding DINO（按你的路径）
        self.gdino_cfg_path = "models/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.gdino_ckpt_path = "models/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"

        # SAM2（按你的路径）
        self.sam2_cfg_path = "models/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam2_ckpt_path = "models/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"

        # 模型
        # self.gdino_proc=''
        # self.gdino_model=''
        model_id = "models/grounding-dino-base"
        self.gdino_proc  = AutoProcessor.from_pretrained(model_id)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id).to('cuda')
        self.sam2_video_predictor = None
    def run_ocr(self,frame):
        return ''
        frame=_to_image(frame)
        
    def detect_object(
        self,
        frame: Any,
        text: str,
        box_threshold: float = 0.5,
        text_threshold: float = 0.25,
    ) -> List[List[float]]:
        """
        """
        if text is None or text.strip() == "":
            return []
        caption = text.strip().lower()
        if not caption.endswith("."):
            caption += "."
        if getattr(self, "gdino_model", None) is None:
            model_id = "models/grounding-dino-base"
            self.gdino_proc = AutoProcessor.from_pretrained(model_id)
            self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id
            ).to("cuda")

        try:
            img = _to_image(frame)
        except Exception as e:
            print(f"[detect_object] fail to convert frame to image: {e}")
            return []
        if img is None:
            return []
        return []
        inputs = self.gdino_proc(images=img, text=caption, return_tensors="pt").to('cuda')
        # print(1234,inputs.device)
        # --------- 前向推理 ----------
        with torch.no_grad():
            outputs = self.gdino_model(**inputs)

        # --------- GroundingDINO 后处理 ----------
        # target_sizes: (H, W)，PIL.size = (W, H)，所以要 img.size[::-1]
        processed = self.gdino_proc.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            # box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[img.size[::-1]],
        )[0]  # dict: boxes, scores, labels
        boxes = processed["boxes"]   # 已经是 xyxy，且是绝对像素坐标（float tensor）
        scores = processed["scores"]
        labels = processed["labels"]  # 文本片段

        if boxes is None or len(boxes) == 0:
            return []

        # --------- 只要 bbox 坐标，不管 label ----------
        results: List[List[float]] = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            if x2 > x1 and y2 > y1:
                results.append([float(x1), float(y1), float(x2), float(y2)])

        return results


    def crop(self, frame: Any, box) -> Image.Image:
        img=_to_image(frame)
        if img is None:
            return frame
        return img
        return img.crop(box)

    def crop_left(self, frame: Any) -> Image.Image:
        img=_to_image(frame)
        if img is None:
            return frame
        return img
        w, h = img.size
        return img.crop((0, 0, w // 2, h))

    def crop_right(self, frame: Any) -> Image.Image:
        return frame
        img = _to_image(frame)
        if img is None:
            return frame
        w, h = img.size
        return img
        return img.crop((w // 2, 0, w, h))

    def crop_top(self, frame: Any) -> Image.Image:
        return frame
        img = _to_image(frame)
        if img is None:
            return frame
        return img
        w, h = img.size
        return img.crop((0, 0, w, h // 2))

    def crop_bottom(self, frame: Any) -> Image.Image:
        return frame
        img = _to_image(frame)
        if img is None:
            return frame
        return img
        w, h = img.size
        return img.crop((0, h // 2, w, h))

    def crop_left_top(self, frame: Any) -> Image.Image:
        return frame
        img = _to_image(frame)
        if img is None:
            return frame
        return img
        w, h = img.size
        return img.crop((0, 0, w // 2, h // 2))

    def crop_right_top(self, frame: Any) -> Image.Image:
        return frame
        img = _to_image(frame)
        if img is None:
            return frame
        return img
        w, h = img.size
        return img.crop((w // 2, 0, w, h // 2))

    def crop_left_bottom(self, frame: Any) -> Image.Image:
        return frame
        img = _to_image(frame)
        if img is None:
            return frame
        return img
        w, h = img.size
        return img.crop((0, h // 2, w // 2, h))

    def crop_right_bottom(self, frame: Any) -> Image.Image:
        return frame
        img = _to_image(frame)
        if img is None:
            return frame
        return img
        w, h = img.size
        return img.crop((w // 2, h // 2, w, h))
    
    def query_video(self, frames: List[Any], question: str) -> str:
        return self.query_frames(frames, question)

    def query_mc(self, frames: List[Any], query: str, choices: List[str]) -> str:

        letters = [chr(65 + i) for i in range(len(choices))]  # 65 是 'A'
        choices_str = "\n".join([f"{letter}. {choice}" for letter, choice in zip(letters, choices)])
        question_with_choices = f"Question: {query}\nChoices:\n{choices_str}"
        prompt = f"{question_with_choices}\nYou must just output a single letter. Best option:"
        
        answer=self.query_frames(frames, prompt)
        # print(1234,prompt,len(frames),123,answer)
        return answer[-1],answer[-2]
    
    
    def query_native(self,video_path, query: str, choices: List[str],num_frames=64,threshold=0.75) -> str:
        # return 0,1,0
        # threshold=0.4
        cho=choices.copy()
        # cho.append('unknown (can\'t find the answer in the video)')
        letters = [chr(65 + i) for i in range(len(cho))]  # 65 是 'A'
        choices_str = "\n".join([f"{letter}. {choice}" for letter, choice in zip(letters, cho)])
        question_with_choices = f"Question: {query}\n Possible answer choices:\n{choices_str}"
        prompt = f" Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D or other letter) of the correct option.\n{question_with_choices}\nOutput a single letter. The best answer is: "

   

        messages = [{'role': 'user', 'content': [
    {'type': 'video', 'video': video_path},
    {'type': 'text', 'text': prompt}
]}]
     
        
         # 4. 调用 vLLM OpenAI 接口
        chat_response = self.llm.chat.completions.create(
            model='qwen3vl',
            messages=messages,
            n=1, 
            temperature=0, 
            max_tokens=20000,
            logprobs=True,
            top_logprobs=5,
)

        # # 6. 拿到模型输出文本
        output_text = chat_response.choices[0].message.content
        # print(output_text)
        # return output_text
        # # 7. 从 logprobs 里算置信度（第一个生成 token 的概率）
        confidence = 0.0
        for choice in chat_response.choices:
            if hasattr(choice, "logprobs") and choice.logprobs and choice.logprobs.content:
                first_token_info = choice.logprobs.content[0]  # 第一个 token
                confidence = float(np.exp(first_token_info.logprob))  # logprob -> prob
                print(
                    f"First token: {first_token_info.token}, "
                    f"LogProb: {first_token_info.logprob:.4f}, "
                    f"Confidence: {confidence:.4f}"
                )
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
        # print(123,output_text)
        # # 9. 把 score 直接用上面算出来的 confidence
        score = confidence if final_answer else 0.0
        
        
       
        
        
        
        # final_answer, score, output_text=self.query_frames(pil_frames, prompt)
        # if score < threshold:
        return  final_answer,score
        # else:
        #     return final_answer,score

    def query_frames(self, frames, question: str, video_fps: int = 8):
        messages=build_messages_with_local_jpg(frames,question)
        # print(frames,1234)
        # max_w, max_h = 0, 0
        # for frame in frames:
        #     w, h = frame.size
        #     max_w = max(max_w, w)
        #     max_h = max(max_h, h)

        # # 先把尺寸夹到不超过 256
        # max_w = min(max_w, 256)
        # max_h = min(max_h, 256)

        # # 再保证像素不超过 256*256（必要时按比例缩小）
        # max_pixels = 256 * 256
        # pixels = max_w * max_h
        # if pixels > max_pixels:
        #     scale = math.sqrt(max_pixels / float(pixels))
        #     max_w = max(1, int(max_w * scale))
        #     max_h = max(1, int(max_h * scale))

        # base_size = (max_w, max_h)

        # resized_frames = [
        #     frame.resize(base_size, Image.LANCZOS) if frame.size != base_size else frame
        #     for frame in frames
        # ]
        # # 2. 每一帧转 JPEG base64（不再拼成 video，只保留列表）
        # b64_frames = []
        # for frame in resized_frames:
        #     import io
        #     buffer = io.BytesIO()
        #     frame.save(buffer, format="JPEG")
        #     base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        #     b64_frames.append(base64_image)

        # # 1. 创建保存帧的文件夹（不存在就自动建）
      
       
       
        # selected_frames = b64_frames
        # n = len(selected_frames)
        # step = n / 128
        # indices = [int(i * step) for i in range(128)]
        # indices = sorted(set(indices))
        # selected_frames = [selected_frames[i] for i in indices]
        # if not len(selected_frames):
        #     messages = [
        #         {
        #             "role": "user",
        #             "content": 
        #             [ {
        #                     "type": "text",
        #                     "text": question,   # prompt / question 都行
        #                 },
        #             ],
        #         }
        #     ]
        
        # else:
     
        #     messages = [
                
        #         {
        #             "role": "user",
        #             "content": [
        #                 *map(
        #                     lambda x: {
        #                         "type": "image_url",
        #                         "image_url": {
        #                             "url": f"data:image/jpeg;base64,{x}",  # 或者 image/jpeg
        #                             # "detail": "low",
        #                         },
        #                     },
        #                     selected_frames,
        #                 ),
        #                 {
        #                     "type": "text",
        #                     "text": 'Based the above video frames. '+ question,   # prompt / question 都行
        #                 },
        #             ],
        #         }
        #     ]
        
        # 4. 调用 vLLM OpenAI 接口
        # print(messages,12345)
        print(messages,123)
        chat_response = self.llm.chat.completions.create(
            model='qwen3vl',
            messages=messages,
            temperature=0,
            top_p=0.01,
            logprobs=True,
            top_logprobs=5,
            max_tokens=20000,
        
        )
        # 6. 拿到模型输出文本
        output_text = chat_response.choices[0].message.content
        # 7. 从 logprobs 里算置信度（第一个生成 token 的概率）
        confidence = 0.0
        for choice in chat_response.choices:
            if hasattr(choice, "logprobs") and choice.logprobs and choice.logprobs.content:
                first_token_info = choice.logprobs.content[0]  # 第一个 token
                confidence = float(np.exp(first_token_info.logprob))  # logprob -> prob
                (
                    f"First token: {first_token_info.token}, "
                    f"LogProb: {first_token_info.logprob:.4f}, "
                    f"Confidence: {confidence:.4f}"
                )
                break
            else:
                print("⚠️ No logprobs found for this choice.")

        # 8. 从 output_text 中抽取最终答案（沿用你原来的规则）
        match = re.match(r'^\s*([A-J]|Yes|No)\b', output_text.strip())
        final_answer = match.group(1) if match else ""

        valid_answers = [
            "A", "B", "C", "D", "E",
            "F", "G", "H", "I", "J",
            "Yes", "No"
        ]

        if final_answer not in valid_answers:
            final_answer = ""
        # 9. 把 score 直接用上面算出来的 confidence
        score = confidence if final_answer else 0.0

        return final_answer, score, output_text
    

    
    def query_yn(self, frames: List[Any], query: str) -> str:
        prompt = f"{query}\nOutput 'Yes' or 'No': Answer:"
        answer=self.query_frames(frames, prompt)
        return answer[-1]#,answer[-2]

    def get_subtitle_hints(self, video_path, question, choices, duration, word_number=300):
        # Get subtitles within the duration
        # return ""
        subtitles = get_subtitles_in_range(video_path,(0, duration))

        # Build choices as A, B, C...
        letters = [chr(65 + i) for i in range(len(choices))]  # 65 = 'A'
        choices_str = "\n".join([f"{letter}. {choice}" for letter, choice in zip(letters, choices)])
        # Combine question and choices
        question_with_choices = f"Question: {question}\nChoices:\n{choices_str}"

   
        # Clear and explicit instruction
        prompt = (
            f"{question_with_choices}\n\n"
            f"### Please summarize the relevant information from the following video subtitles "
            f"that could help answer the above question. "
            f"The summary must be clear, accurate. ###\n\n"
            f"Subtitles start:\n{subtitles}"
            f"\nSubtitles end.\n"
            f"The important summary tip of the question: "
    )
        # summary=get_response(prompt)
        # summary=prompt
        



        try:
            url = "https://modelservice.jdcloud.com/v1"
            resp = requests.get(url, timeout=1)
            intervals=self.retrieval.get_informative_subtitles(video_path,question_with_choices,top_k=50)
            retrieval_subtitles=get_subtitles_in_range(video_path,intervals)
            prompt = (
            f"{question_with_choices}\n\n"
            f"### Please summarize the relevant information from the following video subtitles "
            f"that could help answer the above question. "
            f"The summary must be clear, accurate. ###\n\n"
            f"The retrieval subtitles start:\n{retrieval_subtitles}"
            f"\nSubtitles end.\n"
            f"The important summary tip of the question: "
    )   
            response=get_oai_chat_response(prompt)
        except:
            try:
                intervals=self.retrieval.get_informative_subtitles(video_path,question_with_choices,top_k=50)
                retrieval_subtitles=get_subtitles_in_range(video_path,intervals)
                prompt = (
                f"{question_with_choices}\n\n"
                f"### Please summarize the relevant information from the following video subtitles "
                f"that could help answer the above question. "
                f"The summary must be clear, accurate. ###\n\n"
                f"The retrieval subtitles start:\n{retrieval_subtitles}"
                f"\nSubtitles end.\n"
                f"The important summary tip of the question: "
        )   
                response=get_oai_chat_response_qwen3(prompt)
            except:
                print('error get subtitle hint! no network!')
                intervals=self.retrieval.get_informative_subtitles(video_path,question_with_choices,top_k=20)
                retrieval_subtitles=get_subtitles_in_range(video_path,intervals)
                prompt = (
                f"{question_with_choices}\n\n"
                f"### Please summarize the relevant information from the following video subtitles "
                f"that could help answer the above question. "
                f"The summary must be clear, accurate. ###\n\n"
                f"The retrieval subtitles start:\n{retrieval_subtitles}"
                f"\nSubtitles end.\n"
                f"The important summary tip of the question: "
        )   
                response=str(get_subtitles_in_range(video_path,intervals))
        if ('</think>') in response:
            result= response.split('</think>')[-1]
        else:
            result=response
        return result



    # def semantic_filter(self, frames: List[Any], timestamps: List[float], text: str, top_k: int = 16) -> Tuple[List[float], List[Any], List[float]]:
    #     return filter_by_semantic(frames, timestamps, text, top_k)

    # def generate_caption(self, frames: List[Any], timestamps: List[float], hint: str) -> str:
    #     return get_caption(frames, timestamps, hint)

    # def summarize_video(self, video_path: str, hint_instruction: str) -> str:
    #     return get_summary(video_path, hint_instruction)


    def _generate_trim_path(self,original_path: str, tag: str) -> str:
        base_dir = os.path.dirname(original_path)
        filename = os.path.splitext(os.path.basename(original_path))[0]
        new_filename = f"{filename}_trim-{tag}.mp4"
        return os.path.join(base_dir, new_filename)

# out_path = "temp.mp4"
# clip = VideoFileClip('../../../data/CG-Bench/video/BV1s94y1G7RD.mp4').subclipped(0,10)
# clip.write_videofile(out_path, codec="libx264", audio_codec="aac")
    def trim_frames(self, video_path: str, start: float, end: float,num_frames=64) -> str:
        start=max(0,start)
        out_path = self._generate_trim_path(video_path, f"between-{int(start)}-{int(end)}")
       # return extract_frames(video_path,num_frames)
        if not os.path.exists(out_path): 
            try:      
                clip = VideoFileClip(video_path).subclipped(start, end)
                clip.write_videofile(out_path, codec="libx264", audio_codec="aac")
            except:
                return extract_frames(video_path,num_frames)
        return extract_frames(out_path,num_frames)
    def trim_around(self, video_path: str, timestamp: float, intervals=30, num_frames=64) -> str:
        # intervals=30
      #  return extract_frames(video_path,num_frames)
        half = intervals / 2
        start = max(0, timestamp - half)
        with VideoFileClip(video_path) as clip:
            end_time = clip.duration
        end = min(timestamp + half, end_time - 1)
        out_path = self._generate_trim_path(video_path, f"around-{int(timestamp)}")
        if not os.path.exists(out_path):
            with VideoFileClip(video_path) as clip:
                subclip = clip.subclipped(start, end)
                subclip.write_videofile(out_path, codec="libx264", audio_codec="aac")
        return extract_frames(out_path, num_frames)
    def trim_before(self, video_path: str, timestamp: float, intervals=30,num_frames=64) -> str:
        # intervals=30
      #  return extract_frames(video_path,num_frames)
        out_path = self._generate_trim_path(video_path, f"before-{int(timestamp)}")
        if not os.path.exists(out_path):
            start=max(0,timestamp-intervals)
            clip = VideoFileClip(video_path).subclipped(start, timestamp)
            clip.write_videofile(out_path, codec="libx264", audio_codec="aac")
        return extract_frames(out_path,num_frames)
    def trim_after(self,video_path: str, timestamp: float, intervals=30,num_frames=64) -> str:
        # intervals=30
        #return extract_frames(video_path,num_frames)
        clip = VideoFileClip(video_path)
        end_time = clip.duration
        clip.close()
        out_path = self._generate_trim_path(video_path, f"after-{int(timestamp)}")
        
        if not os.path.exists(out_path):
            end=min(timestamp+intervals,end_time-1)
            clip = VideoFileClip(video_path).subclipped(timestamp,end)
            clip.write_videofile(out_path, codec="libx264", audio_codec="aac")
        return extract_frames(out_path,num_frames)








    # def query_count(self, frames: List[Any], event: str) -> int:
    #     prompt = """
    # You are a precise video analysis assistant. Count how many times the given event occurs.
    # Always answer with an integer only.
    # Q: Count how many times {}.
    # A:"""
    #     _,_,output_text = self.query_frames(frames, prompt)
    #     try:
    #         return int(output_text.strip())
    #     except:
    #         return 0
