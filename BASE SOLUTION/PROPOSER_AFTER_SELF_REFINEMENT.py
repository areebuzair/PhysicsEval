from openai import OpenAI
import json
import os
from dotenv import dotenv_values

# Load environment variables from the .env file (if present)
config = dotenv_values(".env")

# Access environment variables as if they came from the actual environment
BASE_URL = config['BASE_URL']
MODEL = config['MODEL']
API_KEY = config['API_KEY']

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
OUTPUT_FILE = f"./SOLUTIONS/self_refined_solution_by_{sanitize_file_name(MODEL)}.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    PROBLEMS = [json.loads(line) for line in f]

def get_solution(problem: str, ai_solution: str):
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
                    "content": "You are a Physics Professor. Outline physics principles of given problem and please check your own answers for any mistakes, then answer again." 
                }
            ],
            timeout=MAX_TIME_LIMIT
        )

        return completion.choices[0].message.content
    except Exception as e:
        print(e)
        return None



COMPLETED_PROBLEMS = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        COMPLETED_PROBLEMS = set(json.loads(line)['Problem_ID'] for line in f)
ERROR_COUNT = 0
for i, problem in enumerate(PROBLEMS, start=1):
    ID = problem['Problem_ID']
    if ID in COMPLETED_PROBLEMS:
        continue
    print(f"Problem {i}/{len(PROBLEMS)}")

    solution = get_solution(problem['problem'], problem['ai_solution'])
    if not solution:
        ERROR_COUNT += 1
        print("Failed to solve:", ID)
        continue

    DATA = {}
    DATA['Problem_ID'] = ID
    DATA['problem'] = problem['problem']
    DATA['ai_solution'] = solution
    DATA['elaborated_solution_steps'] = problem['elaborated_solution_steps']

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(DATA) + '\n')
if ERROR_COUNT:
    print(f"There were {ERROR_COUNT} error/s: Please run the code again")
else:
    print("All problems solved successfully")
