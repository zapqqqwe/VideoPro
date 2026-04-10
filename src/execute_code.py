from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from analysis import AnalysisManager
from retriever import Retrieval_Manager
from video_utils import process_code


EPS = 1.0
CLIP_PREFIX = "clip_"
CLIP_SUFFIX = ".mp4"


def seconds_to_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def probe_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    output = subprocess.check_output(cmd, text=True).strip()
    duration = max(0.0, float(output))
    return math.floor(duration * 1000) / 1000.0


def clip_index(name: str) -> int:
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return 10**9


def list_clip_files(clip_dir: str) -> list[str]:
    path = Path(clip_dir)
    files = [
        str(p)
        for p in path.iterdir()
        if p.is_file() and p.name.startswith(CLIP_PREFIX) and p.suffix == CLIP_SUFFIX
    ]
    return sorted(files, key=lambda p: clip_index(Path(p).name))


def is_clip_file_valid(path: str) -> bool:
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def clean_invalid_clips(paths: list[str]) -> list[str]:
    valid = []
    for path in paths:
        try:
            if is_clip_file_valid(path):
                valid.append(path)
            elif os.path.exists(path):
                os.remove(path)
        except OSError:
            continue
    return valid


def split_video_to_clips(
    video_path: str,
    clip_dir: str,
    clip_duration: int = 60,
    workers: int = 8,
    tolerate_missing: int = 2,
    overwrite: bool = True,
) -> tuple[list[str], float]:
    Path(clip_dir).mkdir(parents=True, exist_ok=True)

    duration = probe_duration(video_path)
    expected_count = math.ceil((duration - EPS) / clip_duration) if duration > EPS else 0

    existing = list_clip_files(clip_dir)
    if len(existing) >= max(0, expected_count - tolerate_missing):
        return existing, duration

    def cut_one(index: int, start: float, end: float) -> str | None:
        if end - start <= EPS:
            return None

        start_tag = seconds_to_hms(start).replace(":", "-")
        end_tag = seconds_to_hms(end).replace(":", "-")
        output_path = os.path.join(
            clip_dir,
            f"clip_{index}_{start_tag}_to_{end_tag}.mp4",
        )

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(end - start),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "128k",
            "-avoid_negative_ts", "make_zero",
        ]
        if overwrite:
            cmd.append("-y")
        cmd.append(output_path)

        subprocess.run(cmd, check=True)
        return output_path

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i in range(expected_count):
            start = i * clip_duration
            end = min((i + 1) * clip_duration, duration)
            if end - start > EPS:
                futures.append(executor.submit(cut_one, i, start, end))

        outputs = [future.result() for future in as_completed(futures)]

    clips = clean_invalid_clips([p for p in outputs if p])
    return sorted(clips, key=lambda p: clip_index(Path(p).name)), duration


def ensure_video_clips(
    video_path: str,
    clip_save_folder: str,
    clip_duration: int = 60,
    workers: int = 8,
) -> tuple[str, list[str], float]:
    video_id = Path(video_path).stem
    clip_dir = os.path.join(clip_save_folder, video_id)
    Path(clip_dir).mkdir(parents=True, exist_ok=True)

    clip_paths = list_clip_files(clip_dir)
    if clip_paths:
        print(f"[Info] clip folder already exists: {clip_dir}")
        return clip_dir, clip_paths, probe_duration(video_path)

    print(f"[Info] clip folder is empty, splitting video: {video_path}")
    clip_paths, duration = split_video_to_clips(
        video_path=video_path,
        clip_dir=clip_dir,
        clip_duration=clip_duration,
        workers=workers,
        tolerate_missing=2,
        overwrite=True,
    )
    return clip_dir, clip_paths, duration


def build_runtime(clip_save_folder: str) -> dict:
    retrieval = Retrieval_Manager(
        clip_save_folder=clip_save_folder,
        dataset_folder="",
    )
    retrieval.load_model_to_gpu(0)

    analysis = AnalysisManager(retrieval=retrieval)

    runtime = dict(globals())
    runtime.update(
        retrieval=retrieval,
        analysis=analysis,
    )
    return runtime


def extract_code_block(code_string: str) -> str:
    match = re.search(r"<code>(.*?)</code>", code_string, flags=re.DOTALL)
    return match.group(1).strip() if match else code_string.strip()


def compile_execute_function(
    code_string: str,
    runtime_globals: dict,
) -> tuple[Callable, str]:
    raw_code = extract_code_block(code_string)
    processed_code = process_code(raw_code)

    local_env: dict = {}
    exec(processed_code, runtime_globals, local_env)

    execute_command = local_env.get("execute_command") or runtime_globals.get("execute_command")
    if execute_command is None:
        raise ValueError("No function named execute_command found in input code.")

    return execute_command, processed_code


def run_execute_command(
    code_string: str,
    video_path: str,
    question: str,
    choices,
    duration,
    clip_save_folder: str,
    clip_duration: int = 60,
    workers: int = 8,
) -> dict:
    clip_dir, clip_paths, real_duration = ensure_video_clips(
        video_path=video_path,
        clip_save_folder=clip_save_folder,
        clip_duration=clip_duration,
        workers=workers,
    )

    runtime_globals = build_runtime(clip_save_folder)
    runtime_globals.update(
        current_clip_dir=clip_dir,
        current_clip_paths=clip_paths,
    )

    execute_command, processed_code = compile_execute_function(
        code_string=code_string,
        runtime_globals=runtime_globals,
    )

    duration = int(real_duration) if not duration or duration <= 0 else duration
    result = execute_command(video_path, question, choices, duration)

    return {
        "success": True,
        "result": result,
        "processed_code": processed_code,
        "clip_dir": clip_dir,
        "num_clips": len(clip_paths),
        "error": "",
        "traceback": "",
    }


def safe_run_execute_command(
    code_string: str,
    video_path: str,
    question: str,
    choices,
    duration,
    clip_save_folder: str,
    clip_duration: int = 60,
    workers: int = 8,
) -> dict:
    try:
        return run_execute_command(
            code_string=code_string,
            video_path=video_path,
            question=question,
            choices=choices,
            duration=duration,
            clip_save_folder=clip_save_folder,
            clip_duration=clip_duration,
            workers=workers,
        )
    except Exception as exc:
        return {
            "success": False,
            "result": "",
            "processed_code": "",
            "clip_dir": "",
            "num_clips": 0,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    code_string = """
<code>
def execute_command(video_path, question, choices, duration):
    try:
        intervals, clip_paths = get_informative_clips(
            video_path,
            "person doing an activity",
            top_k=2,
            total_duration=duration,
        )
    except Exception:
        intervals, clip_paths = [], []

    frames = []
    for clip in clip_paths:
        try:
            frames.extend(extract_frames(clip, num_frames=16))
        except Exception:
            continue

    if not frames:
        frames = extract_frames(video_path, num_frames=32)

    person_frames = []
    for frame in frames:
        try:
            boxes = detect_object(frame, "person")
        except Exception:
            boxes = []
        if boxes:
            person_frames.append(frame)

    prompt = "What is the person doing in the video?"
    return query_mc(person_frames or frames, prompt, choices)
</code>
"""
    video_path = "/inspire/hdd/global_user/lichenglin-253208540324/VideoPro/__Bchxr3ejw.mp4"
    question = "What is the person doing in the video?"
    choices = [
        "Cooking in the kitchen",
        "Playing guitar",
        "Riding a bicycle",
        "Swimming in a pool",
    ]
    duration = 120
    clip_save_folder = "/inspire/hdd/global_user/lichenglin-253208540324/VidePro/clips"

    output = safe_run_execute_command(
        code_string=code_string,
        video_path=video_path,
        question=question,
        choices=choices,
        duration=duration,
        clip_save_folder=clip_save_folder,
        clip_duration=10,
        workers=8,
    )

    print("=== Execution Output ===")
    print(output)
