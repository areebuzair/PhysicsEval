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
```

Finally, run PROPOSER.py
```
python PROPOSER.py
```