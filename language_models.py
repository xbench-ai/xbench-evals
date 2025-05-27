import os
from openai import OpenAI

API_ENDPOINTS = {
    "anthropic": "https://api.anthropic.com/v1/",
    "deepseek": "https://api.deepseek.com",
    "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "xai": "https://api.x.ai/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "bytedance": "https://ark.cn-beijing.volces.com/api/v3/",
    "moonshotai": "https://api.moonshot.cn/v1",
    "zhipuai": "https://open.bigmodel.cn/api/paas/v4/",
    "minimax": "https://api.minimax.chat/v1",
    "stepfun": "https://api.stepfun.com/v1",
    "tencent": "https://api.hunyuan.cloud.tencent.com/v1",
    "baidu": "https://qianfan.baidubce.com/v2"
}

# API rates per million tokens
USD_EXCHANGE_RATE = 7.3
API_PRICE = {
    # OpenAI https://openai.com/api/pricing/
    "gpt-4o-mini": (0.30, 1.20, USD_EXCHANGE_RATE),
    "gpt-4.1": (2.00, 8.00, USD_EXCHANGE_RATE),
    "o3": (10.00, 40.00, USD_EXCHANGE_RATE),
    "o4-mini": (1.10, 4.40, USD_EXCHANGE_RATE),

    # Doubao https://www.volcengine.com/docs/82379/1099320#深度思考模型
    "doubao-1.5-pro-32k-250115": (0.80, 2.00, 1.0),
    "doubao-1-5-thinking-pro-250415": (4.00, 16.00, 1.0),

    # Anthropic https://www.anthropic.com/pricing#api
    "claude-3-7-sonnet-20250219": (3.00, 15.00, USD_EXCHANGE_RATE),
    "claude-sonnet-4-20250514": (3.00, 15.00, USD_EXCHANGE_RATE),
    "claude-opus-4-20250514": (15.00, 75.00, USD_EXCHANGE_RATE),

    # Google https://ai.google.dev/gemini-api/docs/pricing
    "gemini-2.0-flash": (0.10, 0.40, USD_EXCHANGE_RATE),
    "gemini-2.5-flash-preview-05-20": (0.15, 0.60, USD_EXCHANGE_RATE),
    "gemini-2.5-pro-preview-05-06": (1.25, 10.00, USD_EXCHANGE_RATE),

    # xAI https://docs.x.ai/docs/models#models-and-pricing
    "grok-3-latest": (3.00, 15.00, USD_EXCHANGE_RATE),
    "grok-3-mini-latest": (0.30, 0.50, USD_EXCHANGE_RATE),

    # Tencent https://cloud.tencent.com/document/product/1729/97731
    "hunyuan-turbos-latest": (0.8, 2.00, 1.0),
    "hunyuan-t1-latest": (0.8, 2.00, 1.0),

    # Aliyun https://www.alibabacloud.com/help/en/model-studio/models
    "qwen-max-latest": (1.60, 6.40, USD_EXCHANGE_RATE),
    "qwq-plus": (0.80, 2.40, USD_EXCHANGE_RATE),
    "qwen-plus-latest": (0.40, 1.20, USD_EXCHANGE_RATE),  # Qwen3
    "qwen-3-think": (0.40, 8.00, USD_EXCHANGE_RATE),

    # Baidu Wenxin https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Qm9cw2s7m
    "ernie-4.5-turbo-128k": (3.00, 9.00, 1.0),
    "ernie-x1-turbo-32k": (1.00, 4.00, 1.0),

    # DeepSeek
    "deepseek-chat": (2.00, 8.00, 1.0),
    "deepseek-reasoner": (4.00, 16.00, 1.0),

    # Moonshot AI
    "kimi-latest": (2.00, 10.00, 1.0),
    "kimi-thinking-preview": (200.00, 200.00, 1.0),

    # Zhipu
    "glm-4-air-250414": (0.5, 0.5, 1.0),
    "glm-z1-air": (5.00, 5.00, 1.0),

    # Stepfun
    "step-2-16k": (38.00, 120.00, 1.0),
    "step-r1-v-mini": (2.50, 8.00, 1.0),

    # Minimax
    "MiniMax-Text-01": (1.00, 8.00, 1.0),
}


# Function sourcing request to the corresponding LLM
def get_llm_response(prompt, model, judge=False):
    response = ""

    match model:
        # OpenAI
        # API: https://platform.openai.com/docs/overview
        case "gpt-4.5":
            response = get_api_response(prompt, model="gpt-4.5-preview-2025-02-27", api_key=os.getenv("OPENAI_API_KEY"))
        case "gpt-o1":
            response = get_api_response(prompt, model="o1-2024-12-17", api_key=os.getenv("OPENAI_API_KEY"))
        case "gpt-o3-mini":
            response = get_api_response(prompt, model="o3-mini-2025-01-31", api_key=os.getenv("OPENAI_API_KEY"))
        case "gpt-4.1":
            response = get_api_response(prompt, model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"))
        case "gpt-o3":
            response = get_api_response(prompt, model="o3", api_key=os.getenv("OPENAI_API_KEY"))
        case "gpt-o3-high":
            response = get_api_response(prompt, model="o3", api_key=os.getenv("OPENAI_API_KEY"), think_mode=True)
        case "gpt-o4-mini":
            response = get_api_response(prompt, model="o4-mini", api_key=os.getenv("OPENAI_API_KEY"))
        case "gpt-o4-mini-high":
            response = get_api_response(prompt, model="o4-mini", api_key=os.getenv("OPENAI_API_KEY"), think_mode=True)

        # Anthropic
        # API: https://docs.anthropic.com/en/api/openai-sdk
        case "claude-3.7-sonnet":
            response = get_api_response(prompt, model="claude-3-7-sonnet-20250219", api_key=os.getenv("ANTHROPIC_API_KEY"), base_url=API_ENDPOINTS.get("anthropic"))
        case "claude-3.7-sonnet-think":
            response = get_api_response(prompt, model="claude-3-7-sonnet-20250219", api_key=os.getenv("ANTHROPIC_API_KEY"), base_url=API_ENDPOINTS.get("anthropic"), think_mode=True)
        case "claude-sonnet-4":
            response = get_api_response(prompt, model="claude-sonnet-4-20250514", api_key=os.getenv("ANTHROPIC_API_KEY"), base_url=API_ENDPOINTS.get("anthropic"))
        case "claude-sonnet-4-think":
            response = get_api_response(prompt, model="claude-sonnet-4-20250514", api_key=os.getenv("ANTHROPIC_API_KEY"), base_url=API_ENDPOINTS.get("anthropic"), think_mode=True)
        case "claude-opus-4":
            response = get_api_response(prompt, model="claude-opus-4-20250514", api_key=os.getenv("ANTHROPIC_API_KEY"), base_url=API_ENDPOINTS.get("anthropic"))
        case "claude-opus-4-think":
            response = get_api_response(prompt, model="claude-opus-4-20250514", api_key=os.getenv("ANTHROPIC_API_KEY"), base_url=API_ENDPOINTS.get("anthropic"), think_mode=True)

        # DeepSeek
        # API: https://api-docs.deepseek.com
        case "deepseek-v3":
            response = get_api_response(prompt, model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=API_ENDPOINTS.get("deepseek"))
        case "deepseek-r1":
            response = get_api_response(prompt, model="deepseek-reasoner", api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=API_ENDPOINTS.get("deepseek"))

        # Alibaba
        # API: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
        case "qwen":
            response = get_api_response(prompt, model="qwen-max-latest", api_key=os.getenv("ALIYUN_API_KEY"), base_url=API_ENDPOINTS.get("aliyun"))
        case "qwen-qwq":
            response = get_stream_response(prompt, model="qwq-plus", api_key=os.getenv("ALIYUN_API_KEY"), base_url=API_ENDPOINTS.get("aliyun"))
        case "qwen-3":
            response = get_api_response(prompt, model="qwen-plus-latest", api_key=os.getenv("ALIYUN_API_KEY"), base_url=API_ENDPOINTS.get("aliyun"))
        case "qwen-3-think":
            response = get_stream_response(prompt, model="qwen-3-think", api_key=os.getenv("ALIYUN_API_KEY"), base_url=API_ENDPOINTS.get("aliyun"))

        # xAI - Grok 3
        # API: https://docs.x.ai/docs/overview#featured-models
        case "grok-3":
            response = get_api_response(prompt, model="grok-3-latest", api_key=os.getenv("XAI_API_KEY"), base_url=API_ENDPOINTS.get("xai"))
        case "grok-3-think":
            response = get_api_response(prompt, model="grok-3-mini-latest", api_key=os.getenv("XAI_API_KEY"), base_url=API_ENDPOINTS.get("xai"), think_mode=True)

        # Google
        # API: https://ai.google.dev/gemini-api/docs/openai
        case "gemini-2.0":
            response = get_api_response(prompt, model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"), base_url=API_ENDPOINTS.get("google"), judge=judge)
        case "gemini-2.5-flash":
            response = get_api_response(prompt, model="gemini-2.5-flash-preview-05-20", api_key=os.getenv("GOOGLE_API_KEY"), base_url=API_ENDPOINTS.get("google"))
        case "gemini-2.5-pro":
            response = get_api_response(prompt, model="gemini-2.5-pro-preview-05-06", api_key=os.getenv("GOOGLE_API_KEY"), base_url=API_ENDPOINTS.get("google"))

        # ByteDance
        # API: https://www.volcengine.com/docs/82379/1099455
        case "doubao-1.5":
            response = get_api_response(prompt, model="doubao-1.5-pro-32k-250115", api_key=os.getenv("BYTEDANCE_API_KEY"), base_url=API_ENDPOINTS.get("bytedance"))
        case "doubao-1.5-think":
            response = get_api_response(prompt, model="doubao-1-5-thinking-pro-250415", api_key=os.getenv("BYTEDANCE_API_KEY"), base_url=API_ENDPOINTS.get("bytedance"))

        # Moonshot AI
        # API: https://platform.moonshot.cn/docs/intro
        case "kimi":
            response = get_api_response(prompt, model="kimi-latest", api_key=os.getenv("MOONSHOT_API_KEY"), base_url=API_ENDPOINTS.get("moonshotai"))
        case "kimi-thinking":
            response = get_api_response(prompt, model="kimi-thinking-preview", api_key=os.getenv("MOONSHOT_API_KEY"), base_url=API_ENDPOINTS.get("moonshotai"), think_mode=True)

        # Zhipu AI
        # API: https://open.bigmodel.cn/dev/api/normal-model/glm-4
        case "glm-4-air":
            response = get_api_response(prompt, model="glm-4-air-250414", api_key=os.getenv("ZHIPU_API_KEY"), base_url=API_ENDPOINTS.get("zhipuai"))
        case "glm-z1-air":
            response = get_api_response(prompt, model="glm-z1-air", api_key=os.getenv("ZHIPU_API_KEY"), base_url=API_ENDPOINTS.get("zhipuai"))

        # MiniMax
        # API: https://platform.minimaxi.com/document/ChatCompletion
        case "minimax":
            response = get_api_response(prompt, model="MiniMax-Text-01", api_key=os.getenv("MINIMAX_API_KEY"), base_url=API_ENDPOINTS.get("minimax"))

        # StepFun
        # API: https://platform.stepfun.com/docs/overview/concept
        case "step-2":
            response = get_api_response(prompt, model="step-2-16k", api_key=os.getenv("STEPFUN_API_KEY"), base_url=API_ENDPOINTS.get("stepfun"))
        case "step-r1":
            response = get_api_response(prompt, model="step-r1-v-mini", api_key=os.getenv("STEPFUN_API_KEY"), base_url=API_ENDPOINTS.get("stepfun"))

        # Tencent
        # API: https://cloud.tencent.com/document/product/1729/111007
        case "hunyuan-turbos":
            response = get_api_response(prompt, model="hunyuan-turbos-latest", api_key=os.getenv("TENCENT_API_KEY"), base_url=API_ENDPOINTS.get("tencent"))
        case "hunyuan-t1":
            response = get_api_response(prompt, model="hunyuan-t1-latest", api_key=os.getenv("TENCENT_API_KEY"), base_url=API_ENDPOINTS.get("tencent"))

        # Baidu
        # API: https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Zm2ycv77m
        case "ernie-4.5":
            response = get_api_response(prompt, model="ernie-4.5-turbo-128k", api_key=os.getenv("BAIDU_API_KEY"), base_url=API_ENDPOINTS.get("baidu"))
        case "ernie-x1":
            response = get_api_response(prompt, model="ernie-x1-turbo-32k", api_key=os.getenv("BAIDU_API_KEY"), base_url=API_ENDPOINTS.get("baidu"))

        case _:
            print("Model Not Found!")

    return response


# Non stream api call
def get_api_response(prompt, model, api_key, base_url=None, judge=False, think_mode=False):
    # Setup API Client via OpenAI API format
    try:
        if ("claude" in model) and think_mode:
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                extra_body={
                    "thinking": {"type": "enabled", "budget_tokens": 8000}
                }
            )
        elif ("grok" in model) and think_mode:
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort="high",
            )
        elif ("kimi" in model) and think_mode:
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024 * 32
            )
        elif ("o4-mini" in model or "o3" in model) and think_mode:
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                reasoning_effort="high",
                messages=[{"role": "user", "content": prompt}]
            )
        else:
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

        # Skip error and cost analysis for judge
        if judge:
            return response.choices[0].message.content

        # Error and cost analysis
        finish_reason = response.choices[0].finish_reason

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        input_rate, output_rate, exc_rate = API_PRICE[model]
        query_cost = round((input_tokens * input_rate * exc_rate +
                            output_tokens * output_rate * exc_rate) / 1000000, 6)

        return {"response": response.choices[0].message.content,
                "cost": query_cost,
                "length_cutoff": (finish_reason == "length"),
                "safety_cutoff": (finish_reason == "content_filter"),
                "api_error": False}

    except Exception as e:
        print(f"API error occurred: {e}")

        return {"response": None,
                "cost": None,
                "length_cutoff": False,
                "safety_cutoff": False,
                "api_error": True}


# QwQ and Qwen3 only supports streaming output
def get_stream_response(prompt, model, api_key, base_url):
    try:

        client = OpenAI(api_key=api_key, base_url=base_url)

        if model == "qwen-3-think":
            model = "qwen3-235b-a22b"
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": True},
                stream=True
            )
        else:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )

        answer_content = ""
        for chunk in completion:
            if not chunk.choices:
                continue
            else:
                delta = chunk.choices[0].delta.content
                if delta is None:
                    continue
                elif hasattr(delta, 'reasoning_content'):  # Bypass the thinking content
                    continue
                else:
                    answer_content += delta

        return {"response": answer_content,
                "cost": None,
                "length_cutoff": False,
                "safety_cutoff": False,
                "api_error": False}

    except Exception as e:

        print(f"Streaming API error occurred: {e}")

        return {"response": None,
                "cost": None,
                "length_cutoff": False,
                "safety_cutoff": False,
                "api_error": True}
