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

# Can be used with openai, ollama, gemini, openrouter etc.
client = OpenAI(
  base_url=BASE_URL,
  api_key=API_KEY,
)
MAX_TIME_LIMIT = 60 # seconds

def sanitize_file_name(name: str):
    _forbidden_chars = "<>:\"/\\|?* "
    for _c in _forbidden_chars:
        name = name.replace(_c, "_")
    return name

OUTPUT_FILE = f"./proposed_solution_by_{sanitize_file_name(MODEL)}.jsonl"

# Replace with API call to Huggingface dataset when dataset is made public "https://huggingface.co/datasets/IUTVanguard/PhysicsEval"
with open("test set.json", "r", encoding="utf-8") as f:
    PROBLEMS = json.load(f)

def get_solution(problem: str):
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (f"You are an expert on Physics. You solve problems step by step while maintaining logical consistency. Solve the following Physics problem: {problem}"

                    "Finally, write the final answers in brief. Make sure you write all equations in LaTeX.")
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
    print(f"Problem {i}/{len(PROBLEMS)}")
    ID = problem['Problem_ID']
    if ID in COMPLETED_PROBLEMS:
        continue

    solution = get_solution(problem['problem'])
    if not solution:
        ERROR_COUNT += 1
        continue

    DATA = {}
    DATA['Problem_ID'] = ID
    DATA['problem'] = problem['problem']
    DATA['ai_solution'] = solution
    DATA['elaborated_solution_steps'] = problem['elaborated_solution_steps']

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(DATA) + '\n')
if ERROR_COUNT:
    print("There were errors: Please run the code again")
else:
    print("All problems solved successfully")

    



