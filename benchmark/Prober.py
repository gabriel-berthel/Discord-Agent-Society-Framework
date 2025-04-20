from openai import OpenAI
from dotenv import load_dotenv
import yaml
import json
import os
import ollama
import re


load_dotenv()


class Prober:
    @staticmethod
    def generate_model_response(system, prompt, required_fields=[]):
        """
        Generate the content based on the prompt

        Returns: json 
        """
        
        response = ollama.generate(
            model='llama3:8b',
            system=system,
            prompt=prompt,
            format='json',
            options = {
                "mirostat": 2,
                "mirostat_tau": 7, 
                "mirostat_eta": 0.1, 
                "num_ctx": 8000,
                "repeat_penalty": 1.2,
                "presence_penalty": 1.5,
                "frequency_penalty": 0.0,
                "stop": ["<|endofjson|>"]
            }
        )

        answer = response['response']

        while True:
            try:
                questions = json.loads(answer) 
                for q in questions.values():
                    if not all(field in q for field in required_fields):
                        print(questions)
                        raise ValueError('Missing required fields in question.')
                return list(questions.values())
            except Exception as e:
                print('Error while parsi<ng! Retrying:', e)

    @staticmethod
    def generate_content_questions(content: str, num_questions=10):
        """"
        Generate questions from content (reflection, summary, conversation, history)

        Returns: json of questions (type, question, correct_answer, choices)"""
        
        system = f"""
        You are crafting quizz questions from pieces of text that you read. Here is the format that you should follow:
        
        <Binary> : For binary questions (yes/no or true/false)
        {{
            "type": "binary",
            "question": "Question text goes here?",
            "correct_answer": "Yes",
            "choices": ["Yes", "No"]
        }}
        
        <Multiple> : For multiple choice questions with exactly 4 options each.
        {{
            "type": "multiple_choice",
            "question": "Question text goes here?",
            "correct_answer": "Correct option",
            "choices": ["Option 1", "Option 2", "Correct option", "Option 4"]
        }}
        
        Final output should have this form and alternate between binary and multiple: 
        
        {{
          "q1": <Binary>,
          "q2": <Multiple>,
          ...
          "q{num_questions}": <Binary>
        }}
        
        Fields should **ONLY** be "type", "question", "correct_answer" and "choices"
        """
        
        print('Gnerating responses')
        prompt = f"Format your response as a valid JSON. You should make the question ONLY about the following document Make sure to generate {num_questions} as asked and not more. Document:\n{content}"
        while True:
            try: 
                return Prober.generate_model_response(system, prompt, required_fields=['type', 'question', 'correct_answer', 'choices'])
            except ValueError as e:
                print('Error while parsing! Retrying:', e)
        

    @staticmethod
    def evaluate(questions:list,  responses:list):
        
        total_score = 0
        total_detailed = 0
        for i, q in enumerate(questions):
            expected_answer = q['correct_answer']
            system_answer = responses[i] if i < len(responses) else ""

            print('Generating aligmentment.')
            scores = Prober.score_answer_alignment(q.get("question", ""), expected_answer, system_answer)[0]     
            binary_votes = [
                scores["flexible_binary_score"],
                scores["neutral_binary_score"],
                scores["conservative_binary_score"]
            ]
            
            total_score += 1 if sum(binary_votes) >= 2 else 0
            total_detailed += scores["detailed_score"]
        
        print(total_score, total_detailed, len(questions))
        return {
            "total_score": total_score,
            "total_detailed_score": total_detailed,
            "max_score": len(questions)
        }
        
    @staticmethod
    def score_reflection_relevancy(transcript, reflection, personality):
        system = f"""
        You are a JSON-only evaluator. Rate how relevant the reflection is to the dialogue and personality.

        Instructions:
        - Score each axis (personality, dialogue) from 0 (irrelevant) to 3 (strongly relevant).
        
        Respond ONLY with a JSON object in this format:
        
        {{
            "relevancy_scores": {{
                "personality": <0-3>,
                "dialogue": <0-3>
            }}
        }}
        """
        
        prompt = f"""
        Format your response as a valid JSON. 
        You should score only considering reflection against dialogue and personality and be objective.
        
        Input reflection: "{reflection}"

        Dialogue:
        {transcript}

        Personality:
        {personality}
        """

        return Prober.generate_model_response(system, prompt, required_fields=['personality', 'dialogue'])[0]
    
    
    @staticmethod
    def evaluate_scales(score_list):
        total_scores = {}
        num_items = len(score_list)
        print('Evaluating scale', score_list)
        for scores in score_list:
            for key, value in scores.items():
                if key not in total_scores:
                    total_scores[key] = 0
                total_scores[key] += value

        final_avg_scores = {
            key: round(total / num_items, 4) for key, total in total_scores.items()
        }

        overall_average_score = round(
            sum(final_avg_scores.values()) / len(final_avg_scores), 4
        ) if final_avg_scores else 0.0

        return {
            "total_scores": total_scores,
            "average_scores": final_avg_scores,
            "overall_average_score": overall_average_score
        }


    @staticmethod
    def score_message_relevancy(personality, plan, memories, context, dialogue, response):
        
        system = f"""
        You are a JSON-only evaluator. You are evaluating response relevancy against: personality, plan, memories, context and dialogue

        Instructions:
        - Score each axis (personality, dialogue) from 0 (irrelevant) to 3 (strongly relevant).

        Respond ONLY with a JSON object in this format:
        
        {{
            "relevancy_scores": {{
                "personality": <0-3>,
                "plan": <0-3>,
                "memories": <0-3>,
                "context": <0-3>,
                "dialogue": <0-3>
            }}
        }}
        """
        
        prompt = f"""
        Format your response as a valid JSON. 
        You should score only considering reflection against relevancy personality, plan, memories, context and dialogue and be objective.
        
        Personality:
        {personality}
        
        Plan:
        {plan}
        
        Memories:
        {memories}
        
        Context:
        {context}
        
        Dialogue:
        {dialogue}
        
        
        Here is the response to evaluate: 
        Response: {response}
        """

        return Prober.generate_model_response(system, prompt, required_fields=['personality', 'plan', 'memories', 'context', 'dialogue'])[0]

    @staticmethod
    def score_plan_relevancy(former_plan, context, personality, memories, new_plan):
        
        system = f"""
        You are a JSON-only evaluator. You are evaluating response notebook entries against: personality, former plan and context

        Instructions:
        - Score each axis (personality, dialogue) from 0 (irrelevant) to 3 (strongly relevant).

        Respond ONLY with a JSON object in this format:
        
        {{
            "relevancy_scores": {{
                "personality": <0-3>,
                "former_plan": <0-3>,
                "memories": <0-3>,
                "context": <0-3>
            }}
        }}
        """
        
        prompt = f"""
        Format your response as a valid JSON. 
        You should score only considering notebook entries relevancy against personality, former plan and context, and be objective.
        
        Personality:
        {personality}
        
        Former plan:
        {former_plan}
        
        Memories:
        {memories}
        
        Context:
        {context}
        
        
        Here is the response to notebook entry to evaluated: 
        Response: {new_plan}
        """

        return Prober.generate_model_response(system, prompt, required_fields=['personality', 'former_plan', 'memories', 'context'])[0]
    
    
    @staticmethod
    def score_answer_alignment(question, expected_answer, system_answer):
        system = """
        You are a strict JSON-only evaluator.

        Task:
        Evaluate whether the `system_answer` matches the `expected_answer` under three levels of strictness.

        Scoring Criteria:

        - flexible_binary_score (0 or 1):
        Assign 1 if the `system_answer` is sufficiently equivalent to the `expected_answer`,
        allowing for minor variations in phrasing or structure.
        Focus is on overall meaning, not exact wording.

        - neutral_binary_score (0 or 1):
        Assign 1 if the `system_answer` conveys the intended meaning of the `expected_answer`
        with minimal ambiguity. The answer should be clear and understandable,
        without requiring subjective interpretation.

        - conservative_binary_score (0 or 1):
        Assign 1 only if the `system_answer` is clearly and unambiguously equivalent to the `expected_answer`.
        This is a strict match in meaning, structure, and clarity—no room for uncertainty.

        Also return:

        - detailed_score (0.0 to 1.0):
        A fine-grained score that reflects how well the `system_answer` expresses the `expected_answer`.
        1.0 means fully aligned, 0.0 means completely unrelated.
        Intermediate values indicate partial correctness.

        Important:
        Do not consider verbosity. Longer or shorter answers should not affect scoring.
        Focus only on content accuracy and semantic equivalence.

                Format:
        {
            "alignment_scores": {
                "flexible_binary_score": 0 or 1,
                "neutral_binary_score": 0 or 1,
                "conservative_binary_score": 0 or 1,
                "detailed_score": float (0.0 to 1.0)
            }
        }
        """

        prompt = f"""
        Question: "{question}"
        Expected Answer: "{expected_answer}"
        System Answer: "{system_answer}"

        Decide if the system answer *is the expected answer* under each perspective, and score depth of match.

        Respond only in JSON.
        """

        return Prober.generate_model_response(system, prompt)
