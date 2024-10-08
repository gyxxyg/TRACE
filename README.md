<h2 align="center"> <a href="https://arxiv.org/abs/2405.13382">TRACE: Temporal Grounding Video LLM via Casual Event Modeling</a></h2>

<h5 align="center"> If our project helps you, please give us a star ⭐ and cite our <a href="#bibliography">paper</a>!</h2>
<h5 align="center">

<!-- [![hf_space](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/papers/2405.13382) -->
[![hf_checkpoint](https://img.shields.io/badge/🤗-Checkpoints-9C276A.svg)](https://huggingface.co/Yongxin-Guo/TRACE)
<!-- [![hf_data](https://img.shields.io/badge/🤗-Datasets-9C276A.svg)](https://huggingface.co/Yongxin-Guo/VTG-LLM) -->
<!-- [![arxiv](https://img.shields.io/badge/Arxiv-2405.13382-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2405.13382) -->
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fgyxxyg%2FTRACE&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)


## News
- 10/10/2024, 🔥 Our model checkpoints and inference code are released!

TODO
- [x] Release the model checkpoints
- [x] Release the inference and evaluation code
- [] Release the training and fine-tuning code
- [] Release the training data

## Overview

In this work
- We model the videos by a series of events, and propose causal event modeling framework to capture videos' inherent structure.
- We present a novel task-interleaved video LLM model, TRACE, tailored to implement the causal event modeling framework through the sequential encoding/decoding of timestamps, salient scores, and textual captions.

<div align="center">
    <img src="assets/trace-overview.png" alt="Overview of TRACE" width="700"/>
    <br/>
    <figcaption>Overview of TRACE.</figcaption>
</div>

## Enviroments

We use NPU environments for training and fine-tuning, and use V100 GPUs for evaluation. The environment we use can be found in [npu-requirements](./install_requirements-npu.sh) and [gpu-requirements](./requirements.txt).

## Model Zoo

| Checkpoints | Description | URL |
| ----------- | ----------- | ----------- |
| Initialization      | Weights initialized from VideoLLaMA2 | [trace-init](https://huggingface.co/Yongxin-Guo/trace-init) |
| Stage-1      | Model checkpoints trained after stage-1 | [trace-stage1](https://huggingface.co/Yongxin-Guo/trace-stage1) |
| Stage-2   | Model checkpoints trained after stage-2 | [trace](https://huggingface.co/Yongxin-Guo/trace) |
| FT-Charades      | Fine-tuned on Charades-STA dataset | [trace-ft-charades](https://huggingface.co/Yongxin-Guo/trace-ft-charades) |
| FT-Youcook2      | Fine-tuned on Youcook2 dataset | [trace-ft-youcook2](https://huggingface.co/Yongxin-Guo/trace-ft-youcook2) |
| FT-QVHighlights   | Fine-tuned on QVHighlights dataset | [trace-ft-qvhighlights](https://huggingface.co/Yongxin-Guo/trace-ft-qvhighlights) |

## Inference and Evaluation

Please make sure the model and video paths are correct before running the code.
- Inference codes are provided in [inference.py](./scripts/inference/inference.py).
- Evaluation codes are provided in [eval.sh](./trace/eval/eval.sh)

#### Results

| Youcook2 (Zero-Shot) | CIDER | METEOR | SODA_c | F1 |
| --- | --- | --- | --- | --- |
| TRACE | 8.1 | 2.8 | 2.2 | 22.4 |

| Charades-STA (Zero-Shot) | 0.3 | 0.5 | 0.7 | mIOU |
| --- | --- | --- | --- | --- |
| TRACE | 58.6 | 40.3 | 19.4 | 38.7 |

| QVHighlights (Zero-Shot) | mAP | Hit@1 |
| --- | --- | --- |
| TRACE | 26.8 | 42.7

| ActivityNet-DVC | CIDER | METEOR | SODA_c | F1 |
| --- | --- | --- | --- | --- |
| TRACE | 25.9 | 6.0 | 6.4 | 39.3 |

| ActivityNet-MR | 0.3 | 0.5 | 0.7 | mIOU |
| --- | --- | --- | --- | --- |
| TRACE | 53.0 | 37.7 | 24.0 | 39.0 |

#### Demo

<div align="center">
    <img src="assets/trace-demo.png" alt="Demo of TRACE" width="700"/>
    <br/>
    <figcaption>Demo of TRACE.</figcaption>
</div>


## Acknowledgement
We are grateful for the following awesome projects:
* [VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2)
* [VTG-LLM](https://github.com/gyxxyg/VTG-LLM)
* [TimeChat](https://github.com/RenShuhuai-Andy/TimeChat)
* [Ask-Anything](https://github.com/OpenGVLab/Ask-Anything)
* [InternVid](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid)
