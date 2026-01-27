# WorldVQA Judge Prompt and Utils
# Ported from: https://github.com/MoonshotAI/WorldVQA/blob/master/eval/eval.py

WORLDVQA_JUDGE_PROMPT = """
### Role
You are an expert judge specialized in evaluating the correctness of answers. Your task is to assess whether a model-generated answer is correct based on a given question, the model's response, and the ground truth answer.

### Task: Evaluate Answer Correctness
Please classify the model's response into one of the following three categories. Ignore differences in formatting, punctuation, language (Chinese vs. English), or abbreviations/full names. Focus strictly on the **core semantics** and the **level of detail (granularity)**:

1. **Correct**:
    - The model answer contains the core information of the ground truth.
    - The model answer is semantically consistent with the ground truth and contains no contradictions.
    - **The granularity of the model answer is equal to or finer than the ground truth.**
    - Extra irrelevant information is allowed as long as it does not conflict with the ground truth.

2. **Incorrect**:
    - The model answer provides information that contradicts the ground truth.
    - The model answer provides the wrong specific entity, value, or description.
    - **The granularity of the model answer is coarser than the ground truth**, leading to incomplete or insufficiently specific information.
    - Even if the model expresses uncertainty but follows up with a wrong answer (e.g., "I'm not sure, maybe it's B" when the truth is A), it is considered Incorrect.

3. **Unattempted**:
    - The model explicitly states it does not know the answer (e.g., "I don't know," "I cannot answer this question").
    - The model suggests the user search elsewhere (e.g., "Please search the internet").
    - The model answer contains no information from the ground truth but provides no incorrect or contradictory information.

### Output Format
Please strictly follow this two-line format for your output:
1. **Evaluation**: [A brief explanation of your reasoning]
2. **Label**: [Final classification: "Correct", "Incorrect", or "Unattempted"]

---
### Examples

**Example 1 (Incorrect - Granularity Mismatch/Too Coarse)**
Input:
'''
Question: 图片中属于什么类型的田地？
Model Answer: 图片中展示的是梯田。梯田是在山坡地上开垦并修筑的阶梯状农田。
Ground Truth Answer: 龙脊梯田
'''
Evaluation: 标准答案特指"龙脊梯田"，模型只回答了通用的"梯田"。模型答案层级比答案层级更粗略，未能提供标准答案所需的特指信息，属于层级不一致导致的回答错误。
Label: Incorrect

**Example 2 (Correct - Finer Granularity)**
Input:
'''
Question: What weather phenomenon is in the image?
Model Answer: Based on the visual evidence in the image, the weather phenomenon shown is a **severe storm with extremely high winds**, most likely a **tornado** or a very powerful **hurricane/typhoon**.
Ground Truth Answer: High winds
'''
Evaluation: The ground truth is "high winds," and a "tornado" is a more specific and granular type of high wind. The semantics are correct and the detail is finer.
Label: Correct

**Example 3 (Correct)**
Input:
'''
Question: 图中内容是什么品牌的logo？
Model Answer: via浏览器
Ground Truth Answer: via
'''
Evaluation: 模型答案"via浏览器"包含了标准答案"via"，核心语义一致，且"via浏览器"是更具体的描述，层级上是匹配的。
Label: Correct

**Example 4 (Unattempted)**
Input:
'''
Question: Which athlete is in the image?
Model Answer: I cannot answer this question as I do not have relevant sports data.
Ground Truth Answer: Wout Weghorst
'''
Evaluation: The model explicitly states its inability to answer and provides no incorrect information.
Label: Unattempted

**Example 5 (Incorrect)**
Input:
'''
Question: 图片中展示的是什么苹果品种？
Model Answer: 我觉得可能是阿克苏苹果。
Ground Truth Answer: 烟台苹果
'''
Evaluation: 虽然模型用了"可能"等词汇，但它给出的具体答案"阿克苏苹果"与标准答案"烟台苹果"不符，提供了错误信息。
Label: Incorrect

**Example 6 (Unattempted)**
Input:
'''
Question: What is the name of the insect in this image?
Model Answer: This is a photo of an insect. To find the species, consult an entomologist or use recognition software.
Ground Truth Answer: Japanese rhinoceros beetle
'''
Evaluation: The model does not attempt to name the insect and suggests the user search elsewhere, providing no incorrect information.
Label: Unattempted

---
### Current Task
Input:
'''
Question: {question}
Model Answer: {model_answer}
Ground Truth Answer: {ground_truth_answer}
'''

Evaluation:
"""


def worldvqa_build_judge_prompt(question: str, prediction: str, answer: str) -> str:
    """Build the judge prompt for WorldVQA evaluation.
    
    Args:
        question: The question asked about the image
        prediction: The model's predicted answer
        answer: The ground truth answer
        
    Returns:
        The formatted judge prompt string
    """
    return WORLDVQA_JUDGE_PROMPT.format(
        question=question,
        model_answer=prediction,
        ground_truth_answer=answer
    )


def worldvqa_parse_judge_result(judge_response: str) -> dict:
    """Parse the judge model's response to extract the evaluation result.
    
    Args:
        judge_response: The raw response from the judge model
        
    Returns:
        A dict containing:
        - judge_result: 1 for correct, 0 for incorrect/unattempted, None for error
        - answer_category: "correct", "incorrect", "unattempted", or "error"
        - judge_reason: The full judge response
    """
    import re
    
    if judge_response is None:
        return {
            "judge_result": None,
            "answer_category": "error",
            "judge_reason": "No response from judge"
        }
    
    judge_result_str = str(judge_response).strip()
    
    # First try to find explicit "Label: XXX" pattern (most reliable)
    label_match = re.search(r'Label:\s*(Correct|Incorrect|Unattempted)', judge_result_str, re.IGNORECASE)
    if label_match:
        label = label_match.group(1).lower()
        if label == "correct":
            return {"judge_result": 1, "answer_category": "correct", "judge_reason": judge_result_str}
        elif label == "unattempted":
            return {"judge_result": 0, "answer_category": "unattempted", "judge_reason": judge_result_str}
        else:
            return {"judge_result": 0, "answer_category": "incorrect", "judge_reason": judge_result_str}
    
    # Fallback: Check for keywords in the response
    # Note: "Incorrect" contains "Correct" as substring, so check Incorrect first
    if "Incorrect" in judge_result_str:
        return {
            "judge_result": 0,
            "answer_category": "incorrect",
            "judge_reason": judge_result_str
        }
    elif "Unattempted" in judge_result_str:
        return {
            "judge_result": 0,
            "answer_category": "unattempted",
            "judge_reason": judge_result_str
        }
    elif "Correct" in judge_result_str:
        return {
            "judge_result": 1,
            "answer_category": "correct",
            "judge_reason": judge_result_str
        }
    
    # Default to incorrect if no clear label found
    return {
        "judge_result": 0,
        "answer_category": "incorrect",
        "judge_reason": judge_result_str
    }


def worldvqa_judge_single(question, answer, prediction, idx, judge_model):
    """Judge a single WorldVQA sample. Used for parallel evaluation.
    
    Args:
        question: The question text
        answer: The ground truth answer
        prediction: The model's predicted answer
        idx: The sample index
        judge_model: The judge model instance
        
    Returns:
        A dict containing judge_result, answer_category, judge_reason
    """
    prompt = worldvqa_build_judge_prompt(
        question=str(question),
        prediction=str(prediction),
        answer=str(answer)
    )
    
    try:
        judge_response = judge_model.generate(prompt)
        result = worldvqa_parse_judge_result(judge_response)
    except Exception as e:
        result = {
            "judge_result": None,
            "answer_category": "error",
            "judge_reason": str(e)
        }
    
    return result


def _is_valid_value(val):
    """Check if a value is valid (not None, not NaN)."""
    import pandas as pd
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    return True


def worldvqa_calculate_scores(results: list) -> dict:
    """Calculate overall scores for WorldVQA evaluation.
    
    Args:
        results: List of evaluation result dicts, each containing:
            - judge_result: 1 or 0
            - answer_category: category string
            - difficulty: optional difficulty level
            - category: optional category
            
    Returns:
        Dict containing accuracy scores broken down by difficulty and category
    """
    import numpy as np
    import pandas as pd
    
    # Filter out failed judgments (None or NaN)
    valid_results = [r for r in results if _is_valid_value(r.get("judge_result"))]
    failed_count = len(results) - len(valid_results)
    
    if not valid_results:
        return {"accuracy": 0.0}
    
    accuracies = {}
    
    # Difficulty-based scores
    diff_scores = {"easy": [], "medium": [], "hard": []}
    for result in valid_results:
        difficulty = result.get("difficulty")
        # Handle both string and potential NaN values
        if _is_valid_value(difficulty) and str(difficulty).lower() in diff_scores:
            diff_scores[str(difficulty).lower()].append(int(result["judge_result"]))
    
    for difficulty, scores in diff_scores.items():
        if scores:
            accuracies[f"accuracy_{difficulty}"] = float(np.mean(scores))
    
    # Overall accuracy
    judge_scores = [int(r["judge_result"]) for r in valid_results]
    tot_correct_count = sum(judge_scores)
    tot_accuracy = tot_correct_count / len(valid_results)
    accuracies["accuracy"] = float(tot_accuracy)
    
    # Answer category breakdown
    categories = {}
    for r in valid_results:
        cat = r.get("answer_category", "unknown")
        if not _is_valid_value(cat):
            cat = "unknown"
        categories[cat] = categories.get(cat, 0) + 1
    
    if categories:
        total = len(valid_results)
        for cat, count in categories.items():
            rate = count / total
            accuracies[f"tot_{cat}"] = count
            accuracies[f"tot_{cat}_rate"] = float(rate)
    
    # Category-wise scores (if category column exists)
    category_scores = {}
    for result in valid_results:
        category = result.get("category")
        if _is_valid_value(category):
            category = str(category)
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(int(result["judge_result"]))
    
    if category_scores:
        for category, scores in category_scores.items():
            if scores:
                accuracies[f"category_{category}"] = float(np.mean(scores))
    
    accuracies["valid_samples"] = len(valid_results)
    accuracies["failed_samples"] = failed_count
    
    return accuracies
