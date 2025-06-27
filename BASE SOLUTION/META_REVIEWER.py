import ollama

from ollama import chat
from pydantic import BaseModel
import json

MODEL = "qwen3:32b"


def SanitizeFileName(name: str) -> str:
    """
    Sanitize a string to be used as a filename.
    """
    name = name.replace(" ", "_")
    name = name.replace(":", "_")
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace("?", "_")
    name = name.replace("*", "_")
    return name
output_file = f'META_REVIEW_{SanitizeFileName(MODEL)}.jsonl'


class Review(BaseModel):
  mistakes: list[str]

COMPLETED_PROBLEMS = []
try:
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            COMPLETED_PROBLEMS.append(json.loads(line)['Problem_ID'])
except Exception as e:
    pass

 
with open("Test Set.json", "r", encoding='utf-8') as f:
    PROBLEMS = json.load(f)
    PROBLEMS = {i['Problem_ID']: i['problem'] for i in PROBLEMS}

with open("gemini_2.5_pro_review_by_deepseek-r1_14b.jsonl", "r", encoding='utf-8') as f:
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

with open("gemini_2.5_pro_review_by_phi-4-physics_latest.jsonl", "r", encoding='utf-8') as f:
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

with open("gemini_2.5_pro_review_by_qwen3_14b.jsonl", "r", encoding='utf-8') as f:
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


   
COUNT = 0
with open('openai2.jsonl', 'r', encoding='utf-8') as in_f:
    for line in in_f:
        COUNT += 1
        print(f"Problem {COUNT}")
        try:
            DATA = json.loads(line)
            ID = DATA['custom_id']
            if ID in COMPLETED_PROBLEMS:
                continue
            SOLUTION = DATA['response']['body']['choices'][0]['message']['content']
            PROMPT = (f"Problem: {PROBLEMS[ID]} \n\n I had an LLM generate a solution to this. Solution: {SOLUTION} \n\n I had three other LLMs review this solution and point out any mistakes."
"""
The scoring was done based on the following criteria:
Accuracy of calculations (calculation_accuracy_score): Are the numbers correct based on the formulas used?

Correctness of formulas and principles (formula_correctness_score): Are the right physics and engineering concepts being applied?

Logical consistency (logical_consistency_score): Does the reasoning flow correctly from one step to the next?

Completeness (completeness_score): Does it address all parts of the question?

Assumptions made (assumption_validity_score): Are any new assumptions introduced, and are they reasonable?

Clarty and coherence (clarity_and_coherence_score): Is the explanation clear and easy to understand?

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
            response = chat(
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are an expert on Physics. You are tasked to review the solutions to some problems. You will be provided the reviews already made by other LLMs',
                    },
                    {
                        'role': 'user',
                        'content': PROMPT,
                    }
                ],
                model=MODEL,
                format=Review.model_json_schema(),
            )

            review = Review.model_validate_json(response.message.content)
            review = review.model_dump()
            review['Problem_ID'] = ID
            review['final_score'] = 0.5 * phi4_reviews[ID]['final_score'] + 0.3 * deepseek_reviews[ID]['final_score'] + 0.2 * qwen_reviews[ID]['final_score']
            review['calculation_accuracy_score'] = 0.5 * phi4_reviews[ID]['calculation_accuracy_score'] + 0.3 * deepseek_reviews[ID]['calculation_accuracy_score'] + 0.2 * qwen_reviews[ID]['calculation_accuracy_score']
            review['formula_correctness_score'] = 0.5 * phi4_reviews[ID]['formula_correctness_score'] + 0.3 * deepseek_reviews[ID]['formula_correctness_score'] + 0.2 * qwen_reviews[ID]['formula_correctness_score']
            review['logical_consistency_score'] = 0.5 * phi4_reviews[ID]['logical_consistency_score'] + 0.3 * deepseek_reviews[ID]['logical_consistency_score'] + 0.2 * qwen_reviews[ID]['logical_consistency_score']
            review['completeness_score'] = 0.5 * phi4_reviews[ID]['completeness_score'] + 0.3 * deepseek_reviews[ID]['completeness_score'] + 0.2 * qwen_reviews[ID]['completeness_score']
            review['assumption_validity_score'] = 0.5 * phi4_reviews[ID]['assumption_validity_score'] + 0.3 * deepseek_reviews[ID]['assumption_validity_score'] + 0.2 * qwen_reviews[ID]['assumption_validity_score']
            review['clarity_and_coherence_score'] = 0.5 * phi4_reviews[ID]['clarity_and_coherence_score'] + 0.3 * deepseek_reviews[ID]['clarity_and_coherence_score'] + 0.2 * qwen_reviews[ID]['clarity_and_coherence_score']
            print("Mistakes:", review['mistakes'])
            with open(output_file, 'a', encoding='utf-8') as out_f:
                out_f.write(json.dumps(review, ensure_ascii=False) + '\n')
        except Exception as e:
            print(e)

#   calculation_accuracy_score: float
#   calculation_mistakes: list[str]
#   formula_correctness_score: float
#   formula_mistakes: list[str]
#   logical_consistency_score: float
#   logical_mistakes: list[str]
#   completeness_score: float
#   incomplete_requirements: list[str]
#   assumption_validity_score: float
#   mistaken_assumptions: list[str]
#   clarity_and_coherence_score: float
#   incoherent_statements: list[str]