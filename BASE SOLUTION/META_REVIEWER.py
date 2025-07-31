import ollama
from ollama import Client
from pydantic import BaseModel
import json
import os
from dotenv import dotenv_values

# Load environment variables from the .env file (if present)
config = dotenv_values(".env")

# Access environment variables as if they came from the actual environment
META_REVIEWER = config['META_REVIEWER']
REVIEWERS = config['REVIEWERS'].split(" ")
MODEL = config['MODEL']

def sanitize_file_name(name: str):
    _forbidden_chars = "<>:\"/\\|?* "
    for _c in _forbidden_chars:
        name = name.replace(_c, "_")
    return name


# OUTPUT_FILE = f"./review_of_{sanitize_file_name(MODEL)}_by_{sanitize_file_name(REVIEWER)}.jsonl"
output_file = f'./REVIEWS/META_REVIEW_OF_{sanitize_file_name(MODEL)}_BY_{sanitize_file_name(META_REVIEWER)}_FOR_{"_AND_".join([sanitize_file_name(i) for i in REVIEWERS])}.jsonl'


class Review(BaseModel):
  mistakes: list[str]

chat = Client(timeout=600).chat

MODEL_INDEX = 0
while True:
    ERROR_COUNT = 0
    MODEL = MODELS[MODEL_INDEX]
    INPUT_FILE = f"./proposed_solution_by_{sanitize_file_name(MODEL)}.jsonl"
    REVIEWER = "qwen3:32b"
    print("Review by", REVIEWER)
    with open(f"./review_of_{sanitize_file_name(MODEL)}_by_deepseek-r1_14b.jsonl", "r", encoding='utf-8') as f:
        deepseek_reviews = {}
        i = 0
        for line in f:
            try:
                DATA = json.loads(line)
                ID = DATA['Problem_ID']
                del DATA['Problem_ID']
                deepseek_reviews[ID] =  DATA
            except:
                print(i)
            i += 1

    with open(f"./review_of_{sanitize_file_name(MODEL)}_by_phi4-reasoning_plus.jsonl", "r", encoding='utf-8') as f:
        phi4_reviews = {}
        i = 0
        for line in f:
            try:
                DATA = json.loads(line)
                ID = DATA['Problem_ID']
                del DATA['Problem_ID']
                phi4_reviews[ID] =  DATA
            except:
                print(i)
            i += 1

    with open(f"./review_of_{sanitize_file_name(MODEL)}_by_qwen3_14b.jsonl", "r", encoding='utf-8') as f:
        qwen_reviews = {}
        i = 0
        for line in f:
            try:
                DATA = json.loads(line)
                ID = DATA['Problem_ID']
                del DATA['Problem_ID']
                qwen_reviews[ID] =  DATA
            except:
                print(i)
            i += 1
    OUTPUT_FILE = f"./MAR_of_{sanitize_file_name(MODEL)}_by_{sanitize_file_name(REVIEWER)}.jsonl"
    COMPLETED_PROBLEMS = []
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                COMPLETED_PROBLEMS.append(json.loads(line)['Problem_ID'])
    except Exception as e:
        pass

    
    with open(INPUT_FILE, "r", encoding='utf-8') as f:
        PROBLEMS = [json.loads(line) for line in f]
        
    for i, problem in enumerate(PROBLEMS, start=1):
        ID = problem['Problem_ID']
        if ID in COMPLETED_PROBLEMS:
            continue
        print(f"Problem {i}/{len(PROBLEMS)}")

        PROMPT = (f"Problem: {problem['problem']} \n\n I had an LLM generate a solution to this. Solution: {problem['ai_solution']}  \n\n I had three other LLMs review this solution and point out any mistakes."
                        "Are there any mistakes in the solution? If there are, list them down. Consider the following:"
                        """Accuracy of calculations: Are the numbers correct based on the formulas used?

Correctness of formulas and principles: Are the right physics and engineering concepts being applied?

Logical consistency: Does the reasoning flow correctly from one step to the next?

Completeness: Does it address all parts of the question?

Assumptions made: Are any new assumptions introduced, and are they reasonable?

Clarty and coherence: Is the explanation clear and easy to understand?

Each score must be between 0 and 10.

Also, the mistakes made in each of the categories have been mentioned.
"""
            "deepseek-r1:14b had the following review:"
            f"{json.dumps(deepseek_reviews[ID])}"
            "qwen3:14b had the following review:"
            f"{json.dumps(qwen_reviews[ID])}"
            "phi4-reasoning:plus had the following review:"
            f"{json.dumps(phi4_reviews[ID])}" 

            "Now, from these lists of mistakes, based on the problem and solution, finalize a list of mistakes which you think are actually mistakes."
            )
        try:
            response = chat(
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are an expert on Physics. You are tasked to review the solutions to some problems.',
                    },
                    {
                        'role': 'user',
                        'content': PROMPT,
                    }
                ],
                model=REVIEWER,
                format=Review.model_json_schema(),
            )

            review = Review.model_validate_json(response.message.content)
            review = review.model_dump()
            review['Problem_ID'] = ID
            print("FOund errors:", len(review['mistakes']))
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
                out_f.write(json.dumps(review, ensure_ascii=False) + '\n')
        except Exception as e:
            print(e)
            ERROR_COUNT += 1

    if ERROR_COUNT:
        print(f"There were {ERROR_COUNT} errors. Running again.")
    else:
        MODEL_INDEX += 1
        if MODEL_INDEX == len(MODELS):
            print("All reviews complete.")
            break