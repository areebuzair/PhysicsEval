from ollama import Client
from pydantic import BaseModel
import json
import os
from dotenv import dotenv_values

# Load environment variables from the .env file (if present)
config = dotenv_values(".env")

# Access environment variables as if they came from the actual environment
REVIEWERS = config['REVIEWERS'].split(" ")
MODEL = config['MODEL']


def sanitize_file_name(name: str):
    _forbidden_chars = "<>:\"/\\|?* "
    for _c in _forbidden_chars:
        name = name.replace(_c, "_")
    return name

INPUT_FILE = f"./SOLUTIONS/proposed_solution_by_{sanitize_file_name(MODEL)}.jsonl"
os.makedirs("./REVIEWS", exist_ok=True)

MAX_TIME_LIMIT = 180 # seconds

class Review(BaseModel):
  calculation_accuracy_score: float
  calculation_mistakes: list[str]
  formula_correctness_score: float
  formula_mistakes: list[str]
  logical_consistency_score: float
  logical_mistakes: list[str]
  completeness_score: float
  incomplete_requirements: list[str]
  assumption_validity_score: float
  mistaken_assumptions: list[str]
  clarity_and_coherence_score: float
  incoherent_statements: list[str]

chat = Client(timeout=MAX_TIME_LIMIT).chat

REVIEWER_INDEX = 0
while True:
    ERROR_COUNT = 0
    REVIEWER = REVIEWERS[REVIEWER_INDEX]
    print("Review by", REVIEWER)
    OUTPUT_FILE = f"./REVIEWS/review_of_{sanitize_file_name(MODEL)}_by_{sanitize_file_name(REVIEWER)}.jsonl"
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

        PROMPT = (f"Problem: {problem['problem']} \n\n Solution: {problem['ai_solution']} \n\n Is this solution correct? If there are any mathematical or logical mistakes, point out the mistakes briefly."
    """ 
    Score the solution on the following criteria:
    Accuracy of calculations (calculation_accuracy_score): Are the numbers correct based on the formulas used?

    Correctness of formulas and principles (formula_correctness_score): Are the right physics and engineering concepts being applied?

    Logical consistency (logical_consistency_score): Does the reasoning flow correctly from one step to the next?

    Completeness (completeness_score): Does it address all parts of the question?

    Assumptions made (assumption_validity_score): Are any new assumptions introduced, and are they reasonable?

    Clarty and coherence (clarity_and_coherence_score): Is the explanation clear and easy to understand?

    Each score must be between 0 and 10.

    Also, point out the mistakes made in each of the categories mentioned.
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
            review['final_score'] = (
                review['calculation_accuracy_score'] * 0.3 
                + review['formula_correctness_score'] * 0.25 
                + review['logical_consistency_score'] * 0.25 
                + review['completeness_score'] * 0.1 
                + review['assumption_validity_score'] * 0.05 
                + review['clarity_and_coherence_score'] * 0.05
            )
            print("Final Score:", review['final_score'])
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
                out_f.write(json.dumps(review, ensure_ascii=False) + '\n')
        except Exception as e:
            print(e)
            ERROR_COUNT += 1

    if ERROR_COUNT:
        print(f"There were {ERROR_COUNT} errors. Running again.")
    else:
        REVIEWER_INDEX += 1
        if REVIEWER_INDEX == len(REVIEWERS):
            print("All reviews complete.")
            break