from dotenv import load_dotenv
import yaml
import json
import os
import ollama
import re

load_dotenv()

# Prompt templates
QUIZ_SYSTEM_PROMPT = """\
You are crafting quiz questions from pieces of text. Alternate between binary and multiple-choice formats:

<Binary>:
{
    "type": "binary",
    "question": "Question text?",
    "correct_answer": "Yes",
    "choices": ["Yes", "No"]
}

<Multiple>:
{
    "type": "multiple_choice",
    "question": "Question text?",
    "correct_answer": "Correct option",
    "choices": ["Option 1", "Option 2", "Correct option", "Option 4"]
}

Final output should have this form and alternate between binary and multiple: 

{{
    "q1": <Binary>,
    "q2": <Multiple>,
    ...
    "q{num_questions}": <Binary>
}}


Only use fields: "type", "question", "correct_answer", "choices"
""".strip()

ALIGNMENT_SYSTEM_PROMPT = """\
You are a strict JSON-only evaluator scoring the alignment of a system's answer to an expected answer.
Return:
{
    "alignment_scores": {
        "flexible_binary_score": 0 or 1,
        "neutral_binary_score": 0 or 1,
        "conservative_binary_score": 0 or 1,
        "detailed_score": float (0.0 to 1.0)
    }
}
""".strip()

# --- Validator Class ---
class Validator:
    @staticmethod
    def validate_fields(data, required_fields):
        for q in data.values():
            if not all(field in q for field in required_fields):
                raise ValueError("Missing required fields in data item.")
        return list(data.values())

# --- ModelInteraction Class ---
class ModelInteraction:
    @staticmethod
    def generate_model_response(system, prompt, required_fields=[]):
        response = ollama.generate(
            model='llama3:8b',
            system=system.strip(),
            prompt=prompt.strip(),
            format='json',
            options={
                "mirostat": 2,
                "mirostat_tau": 7,
                "mirostat_eta": 0.1,
                "num_ctx": 8192,
                "repeat_penalty": 1.2,
                "presence_penalty": 1.5,
                "frequency_penalty": 0.0,
                "stop": ["<|endofjson|>"]
            }
        )

        while True:
            try:
                result = json.loads(response['response'])
                return Validator.validate_fields(result, required_fields)
            except Exception as e:
                print("Retrying due to parse error:", e)
                print("Raw Response:", response['response'])

# --- QuestionGenerator Class ---
class QuestionGenerator:
    @staticmethod
    def generate_content_questions(content: str, num_questions=10):
        prompt = f"""
        Format your response as a valid JSON. Generate exactly {num_questions} questions about the following document:

        {content}
        """.strip()
        return ModelInteraction.generate_model_response(
            QUIZ_SYSTEM_PROMPT,
            prompt,
            required_fields=['type', 'question', 'correct_answer', 'choices']
        )

# --- AnswerEvaluator Class ---
class AnswerEvaluator:
    @staticmethod
    def evaluate_qa(questions: list, responses: list):
        total_score = total_detailed = 0

        for i, q in enumerate(questions):
            expected = q['correct_answer']
            response = responses[i] if i < len(responses) else ""
            scores = Prober.score_answer_alignment(q['question'], expected, response)[0]

            if sum([
                scores['flexible_binary_score'],
                scores['neutral_binary_score'],
                scores['conservative_binary_score']
            ]) >= 2:
                total_score += 1

            total_detailed += scores['detailed_score']

        return {
            "total_score": total_score,
            "total_detailed_score": total_detailed,
            "max_score": len(questions)
        }

    @staticmethod
    def score_answer_alignment(question, expected, system_answer):
        prompt = f"""
        Question: "{question}"
        Expected Answer: "{expected}"
        System Answer: "{system_answer}"
        """.strip()
        return ModelInteraction.generate_model_response(
            ALIGNMENT_SYSTEM_PROMPT,
            prompt
        )

# --- AxisEvaluator Class ---
class AxisEvaluator:
    @staticmethod
    def create_single_axis_prompt(item_type: str, axis_name: str, axis_value: str, content: str):
        system = f"""
        You are a JSON-only evaluator. 
        
        Score the relevancy of a {item_type} according to "{axis_name}" from 0 (irrelevant) to 3 (strongly relevant).
        
        Return **JSON format**:
        {{"relevancy_score": {{"{axis_name}": <0-3>}}}} 
        """.strip()

        prompt = f"""
        You must evaluate relevancy against **{axis_name}**
        
        Read the following {axis_name} thorouly: {axis_value}
        
        Here is what the document the person wrote as **{item_type.capitalize()}**: {content}
        
        Reason about how relevant this {item_type} regarding what you read for {axis_name}.
        
        Now proceed to evaluate **{item_type.capitalize()}** relevancy according to **{axis_name}**
        
        Make sure to respect the **JSON format**: {{"relevancy_score": {{"{axis_name}": <0-3>}}}}
        """.strip()
        return system, prompt

    @staticmethod
    def evaluate_axis(item_type: str, axis_name: str, axis_value: str, content: str):
        system, prompt = AxisEvaluator.create_single_axis_prompt(item_type, axis_name, axis_value, content)
        response = ModelInteraction.generate_model_response(system, prompt, required_fields=[axis_name])
        return int(response[0][axis_name])

    @staticmethod
    def score_relevancy(item_type: str, axes: dict, content: str):
        return {
            axis: AxisEvaluator.evaluate_axis(item_type, axis, value, content)
            for axis, value in axes.items()
        }

    @staticmethod
    def make_scaled_relevancy_poll(item_type: str, axes: dict, content_value: str):
        return AxisEvaluator.score_relevancy(item_type, axes, content_value)

# --- Prober Class ---
class Prober:
    @staticmethod
    def generate_content_questions(content: str, num_questions=10):
        return QuestionGenerator.generate_content_questions(content, num_questions)

    @staticmethod
    def evaluate_qa(questions: list, responses: list):
        return AnswerEvaluator.evaluate_qa(questions, responses)

    @staticmethod
    def evaluate_scales(score_list):
        total_scores = {}
        for scores in score_list:
            for k, v in scores.items():
                total_scores[k] = total_scores.get(k, 0) + v

        avg_scores = {k: round(v / len(score_list), 4) for k, v in total_scores.items()}
        overall_avg = round(sum(avg_scores.values()) / len(avg_scores), 4) if avg_scores else 0.0

        return {
            "total_scores": total_scores,
            "average_scores": avg_scores,
            "overall_average_score": overall_avg
        }

    @staticmethod
    def score_answer_alignment(question, expected, system_answer):
        return AnswerEvaluator.score_answer_alignment(question, expected, system_answer)

    @staticmethod
    def score_relevancy(item_type: str, axes: dict, content: str):
        return AxisEvaluator.score_relevancy(item_type, axes, content)

    @staticmethod
    def make_scaled_relevancy_poll(item_type: str, axes: dict, content_label: str):
        return AxisEvaluator.make_scaled_relevancy_poll(item_type, axes, content_label)
