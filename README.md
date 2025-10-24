# xbench-evals <img src="./images/xbench_logo.png" width="40" height="25" />

🌐 [Website](https://xbench.org) | 📄 [Paper](https://xbench.org/files/xbench_profession_v2.4.pdf) | 🤗 [Dataset](https://huggingface.co/datasets/xbench)

Evergreen, contamination-free, real-world, domain-specific AI evaluation framework

xbench is more than just a scoreboard — it's a new evaluation framework with two complementary tracks, designed to measure both the intelligence frontier and real-world utility of AI systems:
- AGI Tracking: Measures core model capabilities like reasoning, tool-use, and memory
- Profession Aligned: A new class of evals grounded in workflows, environments, and business KPIs, co-designed with domain experts

We open source the dataset and evaluation code for two of our AGI Tracking benchmarks: ScienceQA and DeepSearch.

## xbench-ScienceQA
ScienceQA is part of xbench's AGI Tracking series, focused on evaluating fundamental knowledge capabilities across scientific domains. For detailed evaluation procedures and further information, please refer to the [website](https://xbench.org/#/agi/scienceqa) and Eval Card [xbench-ScienceQA.pdf](https://xbench.org/files/Eval%20Card%20xbench-ScienceQA.pdf) (Chinese version)

| Rank |                Model                |   Company    | Score | BoN (N=5) | Time cost (s) |
|------|:-----------------------------------:|:------------:|:-----:|:---------:|:-------------:|
| 1    |               o3-high               |    OpenAI    | 60.8  |   78.0    |     87.7      |
| 2    |           Gemini 2.5 Pro            |    Google    | 57.2  |   74.0    |     63.7      |
| 3    |       Doubao-1.5-thinking-pro       |  ByteDance   | 53.6	 |   69.0    |     116.9     |
| 4    |             DeepSeek-R1             |   DeepSeek   | 50.4  |   71.0    |     161.6     |
| 5    |            o4-mini-high             |    OpenAI    | 50.4	 |   67.0    |     48.2      |
| 6    |  Claude Opus 4 - Extended Thinking  |  Anthropic   | 46.6	 |   69.0    |     30.8      |
| 7    |          Gemini 2.5 flash           |    Google    | 46.2  |   70.0    |     24.1      |
| 8    |            Qwen3 - Think            |   Alibaba    | 45.4  |   66.0    |     105.9     |
| 9    |     Grok 3 Mini (with Thinking)     |     xAI      | 42.6	 |   53.0    |     193.1     |
| 10   | Claude Sonnet 4 - Extended Thinking |  Anthropic   | 39.4	 |   61.0    |     28.3      |

## xbench-DeepSearch
DeepSearch is part of xbench's AGI Tracking series, focused on evaluating tool usage capabilities in search and information retrieval scenarios. For detailed evaluation procedures and further information, please refer to the [website](https://xbench.org/#/agi/aisearch) and Eval Card [xbench-DeepSearch](https://xbench.org/files/Eval%20Card%20xbench-DeepSearch.pdf) (Chinese version)

### xbench-DeepSearch-2510
| Rank |           Product          |  Company  | Accuracy |  Cost/Task  |  Time Cost/Task  | Evaluation Date |
|------|:--------------------------:|:---------:|:--------:|:-----------:|:----------------:|:---------------:|
| 1    |        ChatGPT-5-Pro       |  OpenAI   |   75+    |   ~$0.085   |     5-8 min      |     2025.08     |
| 2    |       SuperGrok Expert     |    xAI    |   40+    |   ~$0.08    |     3-5 min      |     2025.08     |
| 3    |           Fellou           |  Fellou   |   35+    |   ~$2       |     5-8 min      |     2025.08     |
| 3    |        Minimax Agent       |  Minimax  |   35+    |   ~$1.1     |     8-15 min     |     2025.08     |
| 3    |      StepFun Research      |  StepFun  |   35+    |   Free      |     8-15 min     |     2025.08     |
| 3    |           Flowith          |  Flowith  |   35+    |   ~$0.1     |     8-15 min     |     2025.08     |
| 3    |           Skywork          |  Kunlun   |   35+    |   ~$0.55    |     3-5 min      |     2025.08     |
| 3    |         Manus Agent        |  Manus    |   35+    |   ~$0.63    |     3-5 min      |     2025.08     |
| 3    |    Doubao Deep Research    | ByteDance |   35+    |   Free      |     5-8 min      |     2025.08     |
| 10   |    Genspark Super Agent    |  MainFunc |   30+    |   ~$0.15    |     3-5 min      |     2025.08     |
| 10   |          Coze Space        | ByteDance |   30+    |   Free      |     2-3 min      |     2025.08     |


### xbench-DeepSearch-2505
| Rank |           Model            |      Mode       |  Company  | Accuracy | Evaluation Date |
|------|:--------------------------:|:---------------:|:---------:|:--------:|:---------------:|
| 1    |             o3             |     Search      |  OpenAI   |   65+    |     2025.05     |
| 2    |        o4-mini-high        |     Search      |  OpenAI   |   60+    |     2025.05     |
| 3    |           Doubao           |   Deep Think	   | ByteDance |   50+    |     2025.05     |
| 3    |           Grok-3           |  DeeperSearch   |    xAI    |   50+    |     2025.05     |
| 3    |  Gemini 2.5 Pro (preview)  |  Deep Research  |  Google   |   50+    |     2025.05     |




## Setup

Make sure to set the `*_API_KEY` environment variables before using these APIs.
By default, we use gemini-2.0-flash as the judge model, so you need to set the `GOOGLE_API_KEY` environment variable as well.

For the [OpenAI API](https://pypi.org/project/openai/):
```bash
pip install openai
```

## Running the evals
To run the evaluations, you can use the following command:
```bash
python xbench_evals.py --model <model_name> --dataset <data/ScienceQA.csv or data/DeepSearch-2510.csv> --n-repeats <num_repeats>
```

Model name and API endpoint mappings are maintained in language_models.py, you can add your own models there.

## Notes
Benchmark data is encrypted to prevent search engine crawling and contamination, you can use the decrypt code in xbench_evals.py to get the plain text data. Please don't upload the plain text online. 

## Submit your agent
If you are developing an AI agent and would like to evaluate it using the latest version of xbench, we welcome you to contact us. Please submit a public access link of your agent, and we will complete the evaluation within an agreed timeframe and share the results with you promptly.

Contact: [team@xbench.org]()