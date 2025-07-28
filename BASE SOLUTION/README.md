# Base solution

To run the PROPOSER, after cloning the repository run:
```
pip install -r requirements.txt
```

Then, create a ```.env``` file with the following keys:
```
API_KEY=<Your API Key>
BASE_URL=<Base URL: Omit if using openai models directly>
MODEL=<Model Name>
REVIEWERS=<Model Names, space seperated>
META_REVIEWER=<Model Name>
```

Example:
```
API_KEY=api-key
BASE_URL=https://openrouter.ai/api/v1
MODEL=microsoft/phi-4-reasoning-plus
REVIEWERS=qwen2.5:3b llama3.2:1b
META_REVIEWER=qwen3:32b
```

Finally, run PROPOSER.py
```
python PROPOSER.py
```