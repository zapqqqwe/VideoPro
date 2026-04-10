# VideoPro: Adaptive Program Reasoning for Long Video Understanding

Official codebase for **VideoPro**, a unified framework for long-video understanding with **adaptive reasoning** and **self-refinement**.

VideoPro dynamically chooses between:

1. **Native VideoLLM reasoning** for simple/high-confidence questions
2. **Multi-step visual program reasoning** for complex long-range queries

When execution fails or the prediction confidence is low, VideoPro performs **self-refinement** to repair or improve the generated reasoning program.

> Paper: **VideoPro: Adaptive Program Reasoning for Long Video Understanding**

---

## Overview

Long-video understanding is difficult because query-relevant evidence is often sparse and distributed across distant temporal segments. VideoPro addresses this by combining:

- **Adaptive Reasoning**  
  Route each query to either:
  - native VideoLLM reasoning
  - multi-step visual program reasoning

- **Self-Refinement**
  - fix failed programs using runtime error logs
  - refine low-confidence native/program reasoning outputs

- **General Video Module Library**
  - multimodal retrieval
  - temporal localization
  - fine-grained visual extraction
  - global context summarization
  - reasoning and answer generation

The framework is trained in two stages:
- **Stage 1:** Supervised Fine-Tuning (SFT)
- **Stage 2:** Group Relative Policy Optimization (GRPO)

This design improves both accuracy and efficiency on long-video benchmarks. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

---

## Highlights

- **Adaptive query-level routing** between native reasoning and program reasoning
- **Executable visual programs** for long-range temporal and semantic grounding
- **Execution-driven and confidence-driven refinement**
- **Coarse-to-fine video analysis pipeline** rather than naive frame-by-frame scanning
- Strong results on:
  - LongVideoBench
  - VideoMME (long subset)
  - LVBench
  - MLVU :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}

---

## Framework

### 1. Adaptive Reasoning

Given a video `V` and question `Q`, VideoPro first decides whether the query can be solved directly using the native VideoLLM.

- **Native mode**
  - uses a single `query_native(video_path, question, choices)` call
  - efficient for simple or high-confidence questions

- **Program mode**
  - synthesizes a multi-step executable program
  - uses retrieval, subtitle grounding, localization, OCR, detection, and frame-level reasoning when needed

This adaptive decision is central to balancing performance and runtime. :contentReference[oaicite:7]{index=7}

### 2. Self-Refinement

VideoPro refines predictions in two cases:

- **Failed execution refinement**
  - when a program crashes or produces invalid execution traces
  - the model uses runtime logs to repair the program

- **Low-confidence refinement**
  - when a prediction is uncertain
  - the model revises the reasoning strategy and re-executes

This refinement mechanism improves robustness beyond one-shot static programs. :contentReference[oaicite:8]{index=8}

### 3. Video Module Library

VideoPro uses a modular tool library tailored for long-form video understanding:

1. **Multimodal Retrieval**
2. **Temporal Localization**
3. **Fine-grained Visual Extraction**
4. **Global Context Summarization**
5. **Reasoning and Answer Generation**
