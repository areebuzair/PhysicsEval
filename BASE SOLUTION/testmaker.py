import json
import random

minimum = int(input("Enter minimum difficulty (1-10): ") or 1)
maximum = int(input("Enter maximum difficulty (1-10): ") or 10)
with open("test.json", 'r', encoding='utf-8') as f:
    problems = json.load(f)
    problems = [i for i in problems if i['problem_difficulty'] >= minimum and i["problem_difficulty"] <= maximum]
    print(len(problems), "problems found")

n = int(input("How many problems do you want? "))
problems = random.sample(problems, min(n, len(problems)))

with open("test set.json", 'w', encoding='utf-8') as f:
    json.dump(problems, f, indent=4, ensure_ascii=False)