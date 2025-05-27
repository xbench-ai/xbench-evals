import csv
import argparse
from tqdm import tqdm
from eval_grader import eval_and_grade_question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-v3")
    parser.add_argument("--dataset", type=str, default="data/ScienceQA.csv")
    parser.add_argument("--n-repeats", type=int, default=5)
    args = parser.parse_args()
    n_repeats = args.n_repeats

    with open(args.dataset, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        questions = [_ for _ in reader]

    header = ["id", "prompt", "type", "answer"]
    for n_repeat in range(n_repeats):
        n = str(n_repeat + 1)
        header.append("response-" + n)
        header.append("extracted-answer-" + n)
        header.append("score-" + n)
        header.append("score-reason-" + n)
        header.append("exceed-length-" + n)
        header.append("content-filter-" + n)
        header.append("error-" + n)
    header.append("avg_score")
    header.append("best_of_n")
    header.append("majority_vote_answer")
    header.append("majority_vote_score")
    header.append("avg_cost (RMB)")
    header.append("avg_time（s)")

    csv_filename = f"{args.model}_results.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for question in tqdm(questions):
            result = eval_and_grade_question(question, model=args.model, n_repeats=n_repeats)
            writer.writerow(result)


if __name__ == "__main__":
    main()
