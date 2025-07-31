from openai import OpenAI
import json
import os
from dotenv import load_dotenv

# Load environment variables from the .env file (if present)
load_dotenv()

# Access environment variables as if they came from the actual environment
API_KEY = os.getenv('API_KEY')
BASE_URL = os.getenv('BASE_URL')
MODEL = os.getenv('MODEL')
META_REVIEWER = os.getenv('META_REVIEWER')

# Can be used with openai, ollama, gemini, openrouter etc.
client = OpenAI(
  base_url=BASE_URL,
  api_key=API_KEY,
)
MAX_TIME_LIMIT = 180 # seconds

def sanitize_file_name(name: str):
    _forbidden_chars = "<>:\"/\\|?* "
    for _c in _forbidden_chars:
        name = name.replace(_c, "_")
    return name

INPUT_FILE = f"./SOLUTIONS/proposed_solution_by_{sanitize_file_name(MODEL)}.jsonl"
OUTPUT_FILE = f"./SOLUTIONS/solution_by_{sanitize_file_name(MODEL)}_after_single_agent_review_by_{sanitize_file_name(META_REVIEWER)}.jsonl"
REVIEW_FILE = f'./REVIEWS/sar_of_{sanitize_file_name(MODEL)}_by_{sanitize_file_name(META_REVIEWER)}.jsonl'

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    PROBLEMS = [json.loads(line) for line in f]

with open(REVIEW_FILE, "r", encoding="utf-8") as f:
    REVIEWS = [json.loads(line) for line in f]
    REVIEWS = {i['Problem_ID']: i["mistakes"] for i in REVIEWS}

def get_solution(problem: str, ai_solution: str, feedback: list[str]):
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (f"You are an expert on Physics. You solve problems step by step while maintaining logical consistency. Solve the following Physics problem: {problem}"

                    "Finally, write the final answers in brief. Make sure you write all equations in LaTeX.")
                },
                {
                    "role": "assistant",
                    "content": f"{ai_solution}"
                },
                {
                    "role": "user",
                    "content": f"I have some feedback. {" ".join(feedback)} After taking this into account, please generate the solution once again. Remember to write all equations in LaTeX" 
                }
            ],
            timeout=MAX_TIME_LIMIT
        )

        return completion.choices[0].message.content
    except Exception as e:
        print(e)
        return None

while True:
    COMPLETED_PROBLEMS = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            COMPLETED_PROBLEMS = set(json.loads(line)['Problem_ID'] for line in f)
    ERROR_COUNT = 0
    for i, problem in enumerate(PROBLEMS, start=1):
        ID = problem['Problem_ID']
        NO_MISTAKES = False
        if ID in COMPLETED_PROBLEMS:
            continue
        print(f"Problem {i}/{len(PROBLEMS)}")
        solution = ""
        if len(REVIEWS[ID]) != 0:
            solution = get_solution(problem['problem'], problem['ai_solution'], REVIEWS[ID])
        else:
            solution = problem['ai_solution']
            NO_MISTAKES = True
        if not solution:
            ERROR_COUNT += 1
            print("Failed to solve:", ID)
            continue

        DATA = {}
        DATA['Problem_ID'] = ID
        DATA['problem'] = problem['problem']
        DATA['ai_solution'] = solution
        DATA['elaborated_solution_steps'] = problem['elaborated_solution_steps']
        if NO_MISTAKES:
            DATA['no_mistakes'] = True

        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(DATA) + '\n')
    if ERROR_COUNT:
        print(f"There were {ERROR_COUNT} error/s: Please run the code again")
    else:
        print("All problems solved successfully")
        break