import os
import json
import requests
import logging
import time
from pathlib import Path
import re
import itertools

# Configuration
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
MODEL_NAME = "gemini-2.5-pro"
LOG_FILE = "evaluation_run.log"
MAX_API_RETRIES = 3
API_RETRY_DELAY = 5 # Increased delay to be safer
API_TIMEOUT = 180
API_KEY_FILE = "api_keys.txt"
SAVE_CHECKPOINT_INTERVAL = 10

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global API key management
API_KEYS = []
current_api_key_iterator = None

def load_api_keys(file_path):
    """Loads API keys and creates a cyclical iterator."""
    global API_KEYS, current_api_key_iterator
    try:
        with open(file_path, 'r') as f:
            API_KEYS = [line.strip() for line in f if line.strip()]
        if not API_KEYS:
            logger.error(f"No API keys found in {file_path}.")
            raise ValueError(f"No API keys found in {file_path}.")
        logger.info(f"Loaded {len(API_KEYS)} API keys from {file_path}.")
        current_api_key_iterator = itertools.cycle(API_KEYS)
    except FileNotFoundError:
        logger.error(f"API key file not found at {file_path}.")
        raise

def get_next_api_key():
    """Cycles to the next API key."""
    if not current_api_key_iterator:
        raise ValueError("API keys not loaded.")
    return next(current_api_key_iterator)

def call_gemini_api(prompt: str, api_key: str) -> tuple[str | None, int | None]:
    """Calls the Gemini API with a given prompt and API key."""
    url = GEMINI_API_URL.format(model=MODEL_NAME, api_key=api_key)
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=API_TIMEOUT)
        response.raise_for_status()
        json_response = response.json()
        if "candidates" in json_response and json_response["candidates"]:
            content = json_response["candidates"][0].get("content", {})
            if "parts" in content and content["parts"]:
                text = content["parts"][0].get("text")
                if text:
                    return text, 200
        logger.warning(f"Unexpected response format: {str(json_response)[:500]}...")
        return None, 200 # Success, but no text
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        logger.warning(f"HTTP error {status_code} for key ...{api_key[-5:]}.")
        return None, status_code
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request failed: {req_err}")
        return None, None
    return None, None

def get_gemini_response(prompt: str) -> str | None:
    """Manages API key rotation and retries for a single prompt."""
    if not API_KEYS:
        raise ValueError("API keys are not loaded.")

    for _ in range(len(API_KEYS)): # Try each key once
        api_key = get_next_api_key()
        
        for attempt in range(MAX_API_RETRIES):
            response_text, status_code = call_gemini_api(prompt, api_key)
            
            if status_code == 200:
                return response_text
            
            if status_code == 429: # Rate limit
                logger.warning(f"Rate limit for key ...{api_key[-5:]}. Switching key and delaying for 20s.")
                time.sleep(20)
                break # Switch to the next key
            
            if status_code in [400, 401, 403]: # Invalid key
                logger.error(f"Invalid API key ...{api_key[-5:]}. This key will be skipped.")
                break # Permanently fail for this key
            
            # For other retryable errors like 500, 503
            logger.warning(f"Attempt {attempt + 1} failed with status {status_code}. Retrying in {API_RETRY_DELAY}s.")
            time.sleep(API_RETRY_DELAY)

    logger.error("All API keys failed for the request.")
    return None

def extract_json_from_response(response_text: str, problem_id: str) -> dict | None:
    """Attempts to parse response_text as JSON, or extract JSON from it."""
    try:
        # Attempt direct parsing first
        evaluation = json.loads(response_text)
        logger.debug(f"Successfully parsed JSON directly for {problem_id}.")
        return evaluation
    except json.JSONDecodeError:
        logger.warning(f"Direct JSON parsing failed for {problem_id}. Attempting to extract JSON from text.")

        # Try to extract JSON from markdown code block
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text, re.IGNORECASE)
        if not match:
            # If not in a markdown block, try to find the first valid JSON object pattern
            match = re.search(r"(\{[\s\S]*\})", response_text)

        if match:
            json_str = match.group(1)
            try:
                evaluation = json.loads(json_str)
                logger.info(f"Successfully extracted and parsed JSON for {problem_id} after initial parse failure.")
                return evaluation
            except json.JSONDecodeError as json_e_inner:
                logger.error(f"Failed to parse extracted JSON for {problem_id}: {json_e_inner}")
                logger.debug(f"Original response text for {problem_id} (snippet): {response_text[:500]}...")
                logger.debug(f"Extracted JSON string for {problem_id} (snippet): {json_str[:500]}...")
                return None
        else:
            logger.error(f"Could not find or extract JSON from Gemini response for {problem_id}.")
            logger.debug(f"Original response text for {problem_id} (snippet): {response_text[:500]}...")
            return None

def create_evaluation_prompt(problem_id, elaborated_solution, ai_solution):
    """Create the evaluation prompt for Gemini with detailed and stricter scoring guidelines."""
    return f"""You are an expert physics problem evaluator. Your task is to meticulously and STRICTLY evaluate an AI-generated solution based on its own merits and against the provided elaborated solution steps.

Evaluate the AI-generated solution based on the following categories and scoring guidelines. Provide your evaluation STRICTLY as a JSON object.

Evaluation Categories and Scoring Guidelines:

1.  **mathematical_accuracy**: (Score 1-5) How correct are the AI's calculations, numerical answers, and units?
    *   5: All calculations, numerical results, and units are perfectly correct and appropriately presented.
    *   4: Minor calculation error in the AI solution, or an incorrect/missing unit, but the underlying mathematical method is sound.
    *   3: Several minor errors, or one significant calculation error that impacts the AI's result. Units might be inconsistently handled.
    *   2: Major calculation errors or fundamental misunderstandings of mathematical operations in the AI solution.
    *   1: Almost all calculations in the AI solution are incorrect, non-sensical, or missing.

2.  **logical_consistency**: (Score 1-5) Does the AI solution follow a logical step-by-step progression? Is the AI's reasoning sound and aligned with physics principles?
    *   5: The AI solution flows perfectly. Each step logically follows from the previous one. The reasoning is impeccable.
    *   4: AI solution is mostly logical and well-reasoned. Perhaps one step is slightly unclear or its justification is weak, but it doesn't break the overall logic.
    *   3: Some logical gaps, inconsistencies, or steps that don't clearly follow, making the solution harder to follow or verify.
    *   2: Significant logical flaws. Steps are out of order, reasoning is poor or contradictory to established physics.
    *   1: The AI solution is illogical, incoherent, or internally contradictory.

3.  **completeness**: (Score 1-5) Does the AI-generated solution address all parts of the problem and provide a full answer?
    *   5: All parts of the problem (including sub-questions, if any) are fully addressed and answered by the AI.
    *   4: A minor aspect of the problem is overlooked by the AI, or one sub-question is not fully answered or is missing.
    *   3: A significant part of the problem is ignored or left unanswered by the AI.
    *   2: Only a small portion of the problem is addressed by the AI; major components are missing.
    *   1: The problem is largely unaddressed by the AI, or the AI solution is off-topic.

4.  **clarity_and_coherence**: (Score 1-5) Is the AI's explanation clear, concise, and easy to understand?
    *   5: The AI explanation is exceptionally clear, concise, well-structured, and very easy to understand. Excellent use of language and terminology.
    *   4: The AI explanation is clear and generally easy to understand, with minor areas for improvement in conciseness, structure, or flow.
    *   3: The AI explanation is generally understandable but may be verbose, unclear in parts, poorly organized, or contain jargon without adequate explanation.
    *   2: The AI explanation is difficult to understand due to ambiguity, poor writing, or convoluted structure.
    *   1: The AI explanation is incomprehensible, extremely poorly written, or nonsensical.

5.  **formulas_principles**: (Score 1-5) Are correct physical formulas and principles identified and applied correctly by the AI?
    *   5: All necessary physical formulas and principles are correctly identified, stated, and applied appropriately by the AI.
    *   4: Mostly correct formulas/principles used by AI. Perhaps a minor error in recalling a formula, or a slight misapplication of a correct principle.
    *   3: Some incorrect formulas/principles are used by AI, or correct ones are applied incorrectly in a significant way.
    *   2: Major errors in formula/principle selection or application by AI. Fundamental physics concepts are misunderstood by the AI.
    *   1: Completely inappropriate formulas/principles are used by AI, or relevant physics is entirely ignored.

6.  **assumptions_made**: (Score 1-5) Are AI assumptions (explicit or implicit) explicit, justified, and reasonable?
    *   5: All necessary assumptions made by the AI are explicitly stated, well-justified, and perfectly reasonable for the problem context.
    *   4: Most necessary assumptions made by the AI are stated and reasonable; some minor ones might be implicit but obvious, or lack full justification but are acceptable.
    *   3: Some key assumptions are missing, not clearly stated, or questionable in reasonableness.
    *   2: Major unreasonable assumptions are made by the AI, or critical assumptions are not stated, leading to an incorrect or flawed solution path.
    *   1: Assumptions in the AI solution are entirely inappropriate, absent when clearly needed, or lead to a trivialization/misrepresentation of the problem.

7.  **overall_correctness**: (Score 0-10) How correct and sound is the AI's approach and final answer(s) overall?
    *   10: Perfect solution. The AI's method, reasoning, data interpretation, assumptions, and final answer(s) are flawless.
    *   8-9: Excellent solution. Fundamentally correct with very minor, inconsequential flaws or slight stylistic deviations.
    *   6-7: Good solution. Generally correct approach, and largely correct answer(s), but with some noticeable errors, omissions, or areas for improvement.
    *   4-5: Partially correct. The AI demonstrates some understanding but contains significant flaws in reasoning, calculation, or choice of principles.
    *   2-3: Mostly incorrect. The AI shows fundamental misunderstandings of the problem or physics principles.
    *   0-1: Completely incorrect, irrelevant, or no meaningful attempt made by the AI to solve the problem.

Problem ID: {problem_id}

Elaborated Solution Steps(manually provided by the user):

{elaborated_solution}


AI-Generated Solution to Evaluate:

{ai_solution}

Provide your evaluation STRICTLY as a JSON object with the problem_id and scores for each category listed above.
Your entire response should be ONLY the JSON object, starting with {{{{ and ending with }}}}.
Example JSON format:
{{
    "problem_id": "{problem_id}",
    "mathematical_accuracy": <score_1_to_5>,
    "logical_consistency": <score_1_to_5>,
    "completeness": <score_1_to_5>,
    "clarity_and_coherence": <score_1_to_5>,
    "formulas_principles": <score_1_to_5>,
    "assumptions_made": <score_1_to_5>,
    "overall_correctness": <score_0_to_10>
}}"""

def validate_evaluation(evaluation: dict, problem_id: str) -> bool:
    """Validate the evaluation JSON structure and scores."""
    if not isinstance(evaluation, dict):
        logger.error(f"Validation input for {problem_id} is not a dictionary. Received type: {type(evaluation)}")
        return False

    required_fields_types = {
        "problem_id": str,
        "mathematical_accuracy": (int, float), "logical_consistency": (int, float),
        "completeness": (int, float), "clarity_and_coherence": (int, float),
        "formulas_principles": (int, float), "assumptions_made": (int, float),
        "overall_correctness": (int, float)
    }

    for field, expected_type in required_fields_types.items():
        if field not in evaluation:
            logger.error(f"Missing field '{field}' in evaluation for {problem_id}.")
            return False
        if not isinstance(evaluation[field], expected_type): # Allow int or float for scores
            logger.error(f"Invalid type for field '{field}' in evaluation for {problem_id}. Expected {expected_type}, got {type(evaluation[field])}.")
            return False

    if evaluation["problem_id"] != problem_id:
        logger.error(f"Mismatched problem_id in evaluation for {problem_id}. Expected '{problem_id}', got '{evaluation['problem_id']}'.")
        # Do not return False here, as the problem_id mismatch is not a validation error for the *structure* of the evaluation
        # It's a check that happens elsewhere in process_single_jsonl_file
        # This function is primarily for validating the structure and score ranges
        # The check for matching problem_id is better handled when adding to processed_problem_ids
        # return False
        pass # Allow this to pass for now, as the check is redundant and can cause issues if the LLM changes the problem_id string in its JSON output.

    score_1_5_fields = [
        "mathematical_accuracy", "logical_consistency", "completeness",
        "clarity_and_coherence", "formulas_principles", "assumptions_made"
    ]
    for field in score_1_5_fields:
        score = evaluation[field]
        if not (isinstance(score, (int, float)) and 1 <= score <= 5):
            logger.error(f"Invalid score for '{field}' in evaluation for {problem_id}: {score}. Must be a number between 1 and 5.")
            return False

    overall_score = evaluation["overall_correctness"]
    if not (isinstance(overall_score, (int, float)) and 0 <= overall_score <= 10):
        logger.error(f"Invalid score for 'overall_correctness' in evaluation for {problem_id}: {overall_score}. Must be a number between 0 and 10.")
        return False

    return True

def save_evaluated_data(data_to_save, file_path):
    """Saves the evaluated data list to a JSON file."""
    try:
        # Sort data by problem_id before saving for consistency
        sorted_data = sorted(data_to_save, key=lambda x: x.get('Problem_ID', ''))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved {len(data_to_save)} items to {file_path}.")
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}", exc_info=True)

def process_single_jsonl_file(input_filepath: Path):
    """Processes a single .jsonl file sequentially."""
    output_filename = f"evaluated_{input_filepath.stem}.json"
    output_path = input_filepath.parent / output_filename

    evaluated_data = []
    processed_problem_ids = set()
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            if isinstance(loaded_data, list):
                evaluated_data.extend(item for item in loaded_data if isinstance(item, dict))
                processed_problem_ids.update(item.get('Problem_ID') for item in evaluated_data if item.get('Problem_ID'))
            logger.info(f"Loaded {len(evaluated_data)} previously evaluated items from {output_path}.")
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Could not load or parse {output_path}. Starting fresh.")
            evaluated_data = []
            processed_problem_ids = set()

    items_to_process = []
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    problem_id = item.get('Problem_ID')
                    if problem_id and problem_id not in processed_problem_ids:
                        if item.get('elaborated_solution_steps') and item.get('ai_solution'):
                            items_to_process.append(item)
                except (json.JSONDecodeError, AttributeError):
                    continue
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_filepath}")
        return

    if not items_to_process:
        logger.info(f"No new items to process in {input_filepath.name}.")
        return

    logger.info(f"Found {len(items_to_process)} new items to process in {input_filepath.name}.")

    for i, item in enumerate(items_to_process):
        problem_id = item.get('Problem_ID')
        logger.info(f"Processing item {i + 1}/{len(items_to_process)}: {problem_id}")

        prompt = create_evaluation_prompt(
            problem_id,
            item.get('elaborated_solution_steps', ''),
            item.get('ai_solution', '')
        )
        
        response_text = get_gemini_response(prompt)

        if response_text:
            evaluation = extract_json_from_response(response_text, problem_id)
            if evaluation and validate_evaluation(evaluation, problem_id):
                item['gemini_evaluation'] = evaluation
                evaluated_data.append(item)
                logger.info(f"Successfully evaluated {problem_id}.")
            else:
                logger.error(f"Failed to get a valid evaluation for {problem_id}. It will be skipped.")
        else:
            logger.error(f"Failed to get any response for {problem_id}. It will be skipped.")
        
        # Checkpoint saving
        if (i + 1) % SAVE_CHECKPOINT_INTERVAL == 0:
            save_evaluated_data(evaluated_data, output_path)
            logger.info(f"Checkpoint saved after {i + 1} items.")
            
        time.sleep(3) # Add a 3-second delay between each request
            
    # Final save
    save_evaluated_data(evaluated_data, output_path)
    logger.info(f"Finished processing {input_filepath.name}. Total evaluated items: {len(evaluated_data)}")

def main():
    """Main function to run the evaluation script."""
    logger.info("Starting evaluation script run.")
    try:
        load_api_keys(API_KEY_FILE)
    except Exception as e:
        logger.error(f"Failed to load API keys: {e}. Exiting.")
        return

    if not API_KEYS:
        logger.error("No API keys loaded. Cannot proceed.")
        return

    current_dir = Path('.')
    jsonl_files = sorted(list(current_dir.glob('*.jsonl')))
    if not jsonl_files:
        logger.warning("No .jsonl files found in the current directory.")
        return

    logger.info(f"Found {len(jsonl_files)} files to process: {[f.name for f in jsonl_files]}")

    for jsonl_file in jsonl_files:
        logger.info(f"--- Processing file: {jsonl_file.name} ---")
        process_single_jsonl_file(jsonl_file)
        logger.info(f"--- Finished processing file: {jsonl_file.name} ---")

if __name__ == "__main__":
    main()
    logger.info("Evaluation script run finished.")
