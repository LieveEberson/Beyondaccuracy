!pip install --upgrade datasets huggingface_hub
from datasets import load_dataset
import random
import torch
import transformers
from transformers import pipeline, StoppingCriteria, StoppingCriteriaList
import re
import json
import os
os.environ["HF_TOKEN"] = "your_hf_token"
from datetime import datetime
from huggingface_hub import login, HfFolder

# Check if token is set in environment
if not os.getenv("HF_TOKEN"):
    print("Warning: HF_TOKEN environment variable not set. Trying to use token from code...")
    try:
        login("your_hf_token")
    except Exception as e:
        print(f"Error during login: {e}")
        print("Please set your Hugging Face token as an environment variable named 'HF_TOKEN'")
        print("You can get your token from: https://huggingface.co/settings/tokens")
        exit(1)

# Load datasets and prepare examples
languages = ["en", "bn", "de", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
language = languages[5]

print("Loading datasets...")
train_dataset = None
test_dataset = None

train_dataset = load_dataset("juletxara/mgsm", language, split="train", token=os.environ["HF_TOKEN"])
test_dataset = load_dataset("juletxara/mgsm", language, split="test", token=os.environ["HF_TOKEN"])

if train_dataset is None or test_dataset is None:
    print("Failed to load datasets. Exiting...")
    exit(1)

print("Datasets loaded successfully!")

few_shot_examples = []
for example in train_dataset:
    few_shot_examples.append({
        "question": example["question"],
        "answer_text": example["answer"],
        "answer": str(example["answer_number"])
    })

test_examples = []
for example in test_dataset:
    test_examples.append({
        "question": example["question"],
        "answer": str(example["answer_number"])
    })

prompts = {
    "en" :"Let's think step by step.",
    "bn": "আসুন ধাপে ধাপে চিন্তা করি।",
    "de": "Denken wir Schritt für Schritt.",
    "es": "Pensemos paso a paso.",
    "fr": "Réfléchissons étape par étape.",
    "ja": "段階的に考えてみましょう。",
    "ru": "Давайте думать поэтапно.",
    "sw": "Hebu fikiria hatua kwa hatua.",
    "te": "అంచెలంచెలుగా ఆలోచిద్దాం.",
    "th": "ลองคิดทีละขั้นตอน",
    "zh": "让我们一步步思考。"
    }

questions_language = {
    "en" :"Question:",
    "de": "Frage:",
    "ja": "問題:",
    "ru": "Задача:",
    "sw": "Swali:",
    "th": "โจทย์:"
    }

# Format the few-shot prompt
def format_few_shot_prompt(examples, new_question, n_shots=8):
    """
    Create a few-shot prompt with step-by-step examples and explicit instruction
    """
    prompt = ""

    for ex in examples:
        prompt += f"{ex['question']}\n" # TBD: clean up
        prompt += f"{ex['answer_text']}\n\n"
        #prompt += f"Answer: {ex['answer']}\n\n"

    prompt += f"{questions_language[language]}{new_question}\n"
    prompt += f"{prompts[language]}\n"

    return prompt

# Load the model
pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

class StopOnSubstrings(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return any(decoded.strip().endswith(stop_string) for stop_string in self.stop_strings)

# Instantiate your custom stopping criteria
stop_strings = ["Question:", "Q:", "question:", "Frage:", "問題:", "Задача:", "Swali:", "โจทย์:", "問題"]
custom_stopping_criteria = StoppingCriteriaList([
    StopOnSubstrings(stop_strings, pipe.tokenizer)
])

# Generate solution
def generate_solution(question, n_shots=8):
    prompt = format_few_shot_prompt(few_shot_examples, question, n_shots)

    response = pipe(
        prompt,
        max_new_tokens=1024,
        temperature=0.0,       # Set temperature to 0 for deterministic generation
        top_p=1.0,            # Set top_p to 1.0 to disable nucleus sampling
        # top_k=0,              # Set top_k to 0 to disable top-k sampling
        do_sample=False,      # Disable sampling
        num_return_sequences=1,
        pad_token_id=pipe.tokenizer.eos_token_id,
        eos_token_id=pipe.tokenizer.eos_token_id,
        repetition_penalty=1.0,  # Reduce repetition penalty since we're not sampling
        length_penalty=1.0,      # Use default length penalty
        no_repeat_ngram_size=0,  # Disable n-gram repetition prevention
        early_stopping=True,
        num_beams=5,            # Keep beam search for better accuracy
        stopping_criteria=custom_stopping_criteria
    )[0]['generated_text']

    # Clean up the response
    response = response.replace(prompt, "").strip()

    return response

# Application
def evaluate_question(question, true_answer):
    print("\nQuestion:", question)
    print("\nGenerating solution...")
    solution = generate_solution(question)
    print("\nModel's solution:")
    print(solution)
    print("\nTrue answer:", true_answer)

def extract_final_answer(solution_text):
    """
    Extract the final numerical answer from the solution text using regex pattern matching.
    Returns the last number found in the text, or None if no number is found.
    """
    # Remove commas from the text
    solution_text = solution_text.replace(',', '')

    # Find all numbers in the text using regex
    # Pattern matches:
    # - Optional minus sign followed by digits and optional decimal
    # - OR digits followed by optional space and more digits
    numbers = re.findall(r'-?\d+\.?\d*|\d+(?:\s+\d+)?', solution_text)

    if not numbers:
        return None

    # Convert the last number found to a float
    try:
        return float(numbers[-1])
    except ValueError:
        return None

def save_results_to_json(results, filename=None):
    """
    Save evaluation results to a JSON file
    """
    if filename is None:
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_responses_{timestamp}.json"

    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {filepath}")


def evaluate_accuracy(test_cases, num_cases=None):
    """
    Evaluate the model's accuracy on a set of test cases and save results to JSON
    """
    if num_cases is not None:
        test_cases = random.sample(test_cases, min(num_cases, len(test_cases)))

    correct = 0
    total = len(test_cases)
    epsilon = 1e-10  # Small value for numerical comparison
    results = []

    print(f"\nEvaluating accuracy on {total} test cases...")
    print("-" * 80)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}/{total}:")
        print("Question:", test_case["question"])
        print("\nGenerating solution...")

        solution = generate_solution(test_case["question"])
        print("\nModel's solution:")
        print(solution)

        predicted_answer = extract_final_answer(solution)
        true_answer = test_case["answer"]

        print("\nPredicted answer:", predicted_answer)
        print("True answer:", true_answer)

        # Convert answers to floats and compare
        try:
            pred_float = float(predicted_answer) if predicted_answer else None
            true_float = float(true_answer) if true_answer else None

            is_correct = False
            if pred_float is not None and true_float is not None:
                if abs(pred_float - true_float) < epsilon:
                    correct += 1
                    is_correct = True
                    print("✓ Correct!")
                else:
                    print("✗ Incorrect")
            else:
                print("✗ Could not compare answers")
        except ValueError:
            print("✗ Invalid number format")
            is_correct = False

        # Save the result for this test case
        results.append({
            "question": test_case["question"],
            "model_solution": solution,
            "predicted_answer": predicted_answer,
            "true_answer": true_answer,
            "is_correct": is_correct,
        })

        print("-" * 80)

    accuracy = (correct / total) * 100

    print(f"\nFinal Results:")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Add statistics to results
    results.append({
        "statistics": {
            "accuracy": accuracy,
            "total_cases": total,
            "correct_cases": correct,
        }
    })

    # Save all results to JSON
    save_results_to_json(results)

    return accuracy

# Test with a few examples
def run_examples(num_examples):
    test_cases = random.sample(test_examples, num_examples)
    for test_case in test_cases:
        evaluate_question(test_case["question"], test_case["answer"])
        print("-" * 80)

# Run accuracy evaluation
def run_accuracy_evaluation():
    """
    Evaluate accuracy on all test examples in their original order
    """
    print(f"\nEvaluating all {len(test_examples)} test examples in order...")
    evaluate_accuracy(test_examples)

# Run evaluation on all test examples
if __name__ == "__main__":
    print("\nRunning accuracy evaluation on all test examples:")
    run_accuracy_evaluation()
