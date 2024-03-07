<h1 align="center">
RefuteBench: Evaluating Refuting Instruction-Following for Large Language Models
</h1>

## Introduction
RefuteBench evaluates how LLMs react to refuting instructions, demonstrating how stubborn are LLMs against user feedback. 
For more details, please check our [paper](https://arxiv.org/abs/2402.13463).


## Dependency


## Tutorial
Bash scripts for reproduce our results can be found in "scripts/". 
Our implementation is mostly based on OPENAI api and VLLM. 
For open-source models like LLaMA-2 family, Mistral and Alpaca, we use VLLM to first serve the model and then send queries to the server. 
For close-source models like ChatGPT, GPT4 and Claude-2, we directly send queries to Offical API endpoints. 


## 💡 Note
Data and code will be uploaded within the next few days. Stay tuned!
[Update 2024.3.7] We upload code and scripts to reproduce our benchmark.

[Update 2024.2.29] The data is uploaded in data/*. Our benchmark contains three tasks, namely machine translation, question answering and email writing. 
Then, we have two scenarios: single-feedback and multi-feedback.
The single-feedback scenario is a clean setting to study problems like memory loss and instruction following, and the multi-feedback setting mimics the real-world application. 
We recommend to using the multi-feedback directly to test your LLMs. 

## ☕️ Citation
If you find this repository helpful, please consider citing our paper:

```
@misc{yan2024refutebench,
      title={RefuteBench: Evaluating Refuting Instruction-Following for Large Language Models}, 
      author={Jianhao Yan and Yun Luo and Yue Zhang},
      year={2024},
      eprint={2402.13463},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
