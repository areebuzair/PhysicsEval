from ollama import Client
from pydantic import BaseModel
import json
import os
from dotenv import dotenv_values

MAX_TIME_LIMIT = 180 # seconds

# Load environment variables from the .env file (if present)
config = dotenv_values(".env")

# Access environment variables as if they came from the actual environment
REVIEWER = config['META_REVIEWER']
MODEL = config['MODEL']

def sanitize_file_name(name: str):
    _forbidden_chars = "<>:\"/\\|?* "
    for _c in _forbidden_chars:
        name = name.replace(_c, "_")
    return name


os.makedirs("./REVIEWS", exist_ok=True)
OUTPUT_FILE = f'./REVIEWS/sar_of_{sanitize_file_name(MODEL)}_by_{sanitize_file_name(REVIEWER)}.jsonl'


class Review(BaseModel):
  mistakes: list[str]

chat = Client(timeout=MAX_TIME_LIMIT).chat


ERROR_COUNT = 0
INPUT_FILE = f"./SOLUTIONS/proposed_solution_by_{sanitize_file_name(MODEL)}.jsonl"
print("Review by", REVIEWER)
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

    PROMPT = (f"Problem: {problem['problem']} \n\n I had an LLM generate a solution to this. Solution: {problem['ai_solution']}  \n"
                    "Are there any mistakes in the solution? If there are, list them down. Consider the following:"
                    """Accuracy of calculations: Are the numbers correct based on the formulas used?

Correctness of formulas and principles: Are the right physics and engineering concepts being applied?

Logical consistency: Does the reasoning flow correctly from one step to the next?

Completeness: Does it address all parts of the question?

Assumptions made: Are any new assumptions introduced, and are they reasonable?

Clarty and coherence: Is the explanation clear and easy to understand?

"""
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
        print("Found errors:", len(review['mistakes']))
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
            out_f.write(json.dumps(review, ensure_ascii=False) + '\n')
    except Exception as e:
        print(e)
        ERROR_COUNT += 1

if ERROR_COUNT:
    print(f"There were {ERROR_COUNT} errors. Please run again.")
else:
    print(f"All problems reviewed successfully")