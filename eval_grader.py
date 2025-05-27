import re
import time
import random
import numpy as np
from language_models import get_llm_response
from collections import Counter

###################################################
# Prompt Templates
###################################################

# System and answer format prompts
QUESTION_PROMPT = """
你是一个通用人工智能助手。我将向你提出一个学术问题, 请尽可能简洁地给出解题思路, 并用以下模版作为回答的结尾:

最终答案:[你的答案]

不要在最终答案周围添加任何多余的符号, 不要使用换行（在同一行中完成回答）
""".strip()

OBJECTIVE_PROMPT = "对于本题, 你的答案必须是尽可能简洁的数值, 短语, 或者数学表达式; 如果答案有多个, 使用逗号将它们隔开。"
CHOICE_PROMPT = "对于本题, 选出所有符合的选项, 少选、多选或错选都不得分; 如果选项有多个, 连续列出所有选项, 不要使用逗号或空格分隔。"

# LLM as a Judge prompt
LLM_JUDGE_PROMPT = """
你是一个通用人工智能助手。根据下面给出的[正确答案], 判断以下对[原问题]的[回答]的回答是否正确。

[原问题]: {question}

[正确答案]: {correct_answer}

[回答]:{response}

你的判断必须按照以下格式和标准进行:

最终答案: 从[回答]中提取出的最终准确答案。如果[回答]中没有明确的最终答案, 则填写'无'。

解释: 根据[正确]解释为什么[最终答案]是正确的或错误的。只关注[最终答案]与[正确答案]之间是否存在实质性差异, 不要评论题目的背景, 不要尝试重新解题, 不要为任何不同于[正确答案]的答案辩护, 只专注于判断答案是否一致。

结论: 如果[最终答案]与上方给出的[正确答案]一致, 或者在数值题目中处于可接受的微小误差范围内, 则填写'正确'; 否则（即存在任何不一致、歧义、不等价或提取出的答案错误的情况）填写'错误'。
""".strip()


###################################################
# Functions used to prompt and response for LLMs
###################################################

# Generate full question prompt for LLM
def get_question_prompt(question, question_type):
    full_prompt = QUESTION_PROMPT

    match question_type:
        case "问答题":
            full_prompt += OBJECTIVE_PROMPT + "\n\n"
        case "选择题":
            full_prompt += CHOICE_PROMPT + "\n\n"
        case _:
            pass

    full_prompt += "[问题]: " + question

    return full_prompt


# Majority vote function
def majority_vote(answers):
    if not answers:
        return None  # Return None if the input list is empty

    count = Counter(answers)
    max_votes = max(count.values())

    # Find all answers with the maximum number of votes
    candidates = [answer for answer, votes in count.items() if votes == max_votes]

    # If a tie, randomly choose one among the candidates
    vote_result = random.choice(candidates)

    return vote_result


# 解析 Match 结果
def parse_match_result(match):
    if match is None:
        return match

    match = match.group(0)

    try:
        target = match.split(':')[1].strip()
        return target
    except Exception as e:
        return match  # return naive result in case of failed


# Grade one question with LLM judge
def grade_question(question_text, correct_answer, llm_response):
    if llm_response is None:
        return 0, "", ""

    # If there's direct match, do not need LLM judge
    simple_match = re.search(r'最终答案:*(.*)', llm_response)
    simple_match = parse_match_result(simple_match)
    if simple_match == correct_answer:
        return 1, simple_match, "答案完全正确, 无需调用LLM Judge"

    # Otherwise, use LLM Judge
    judge_prompt = LLM_JUDGE_PROMPT.format(
        question=question_text,
        correct_answer=correct_answer,
        response=llm_response,
    )

    judge_response = get_llm_response(judge_prompt, model='gemini-2.0', judge=True)
    if judge_response is None:
        return 0, "", "Judge Response error"

    # Extract grader conclusions
    extract_match = re.search(r'最终答案:*(.*)', judge_response)
    extract_match = parse_match_result(extract_match)

    correct_match = re.search(r"结论:*.(正确|错误)", judge_response)
    correct_match = parse_match_result(correct_match)

    explain_match = re.search(r"解释:*(.*)", judge_response)
    explain_match = parse_match_result(explain_match)

    score = 1 if (correct_match == "正确") else 0

    return score, extract_match, explain_match


###################################################
# Integrated function to grade one question
###################################################

def eval_and_grade_question(question, model, n_repeats=5):
    # Extract the question contents
    question_id = question["id"]
    question_text = question["prompt"]
    question_type = question["type"] if "type" in question else "问答题"
    correct_answer = question["answer"]
    result = [question_id, question_text, question_type, correct_answer]

    # Get decorated prompt for ScienceQA, keep the original prompt for DeepSearch
    question_prompt = get_question_prompt(question=question_text, question_type=question_type) if "type" in question else question_text

    # Run multiple repeats and record answers
    score_list = []
    extracted_answer_list = []
    cost_list = []
    time_list = []

    for i in range(n_repeats):
        start_time = time.time()
        response = get_llm_response(question_prompt, model=model)
        end_time = time.time()

        response_time = round(end_time - start_time, 3)
        llm_response = response["response"]
        api_cost = response["cost"] if (response["cost"] is not None) else 0
        is_length_cutoff = "Y" if response["length_cutoff"] else ""
        is_safety_cutoff = "Y" if response["safety_cutoff"] else ""
        is_api_error = "Y" if response["api_error"] else ""

        score, extracted_answer, grader_explanation = grade_question(question_text, correct_answer, llm_response)

        result.append(llm_response)
        result.append(extracted_answer)
        result.append(str(score))
        result.append(grader_explanation)

        result.append(is_length_cutoff)
        result.append(is_safety_cutoff)
        result.append(is_api_error)

        score_list.append(score)
        extracted_answer_list.append(extracted_answer)
        cost_list.append(api_cost)
        time_list.append(response_time)

    # Get summary metrics
    average_score = np.average(score_list)
    best_score = np.max(score_list)
    majority_vote_answer = majority_vote(extracted_answer_list)
    majority_vote_score = 1 if (majority_vote_answer == correct_answer) else 0
    average_cost = round(np.average(cost_list), 6)
    average_time = round(np.average(time_list), 3)

    result.append(str(average_score))
    result.append(str(best_score))
    result.append(majority_vote_answer)
    result.append(str(majority_vote_score))
    result.append(str(average_cost))
    result.append(str(average_time))

    return result
