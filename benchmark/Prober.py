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
    def generate_model_response(prompt):
        """
        Generate the content based on the prompt

        Returns: json 
        """
        
        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            answer = response.choices[0].message.content
        except Exception as e:
            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {"role": "system", "content": "Do not make any comment. Make sure to follow the instructions."},
                    {"role": "user", "content": prompt}
                ],
            )

            answer = response['message']['content']
        
        try: 
            questions = json.loads(answer)
            return questions
        except json.JSONDecodeError as e:
            raise ValueError(e)

    @staticmethod
    def generate_content_questions(content: str, num_questions=10):
        """"
        Generate questions from content (reflection, summary, conversation, history)

        Returns: json of questions (type, question, correct_answer, choices)
        """
        prompt = f"""
        Generate {num_questions//2} binary (yes/no or true/false) questions and {num_questions - num_questions//2} multiple choice questions with exactly 4 options each.

        Format your response as a valid JSON array of question objects with the following structure:

        [
            {{
                "type": "binary",
                "question": "Question text goes here?",
                "correct_answer": "Yes",
                "choices": ["Yes", "No"]
            }},
            {{
                "type": "multiple_choice",
                "question": "Question text goes here?",
                "correct_answer": "Correct option",
                "choices": ["Option 1", "Option 2", "Correct option", "Option 4"]
            }}
        ]
        Content:
        {content}
        """
        # generate questions
        questions = Prober.generate_model_response(prompt)
        return questions[0]

    @staticmethod
    def generate_sk_questions(peronality_prompt, num_questions=10):
        """
        Generate self-knowledge questions evaluation for an archetype

        Returns: json of questions (type, question, correct_answer, choices)
        """

        prompt =  f"""
        Generate {num_questions//2} binary (yes/no or true/false) questions and {num_questions - num_questions//2} multiple choice questions with exactly 4 options each.
        The questions should help evaluate how well the archetype understands its own traits and characteristics.

        Format your response as a valid JSON array of question objects with the following structure:

        [
            {{
                "type": "binary",
                "question": "Question text goes here?",
                "correct_answer": "Yes",
                "choices": ["Yes", "No"]
            }},
            {{
                "type": "multiple_choice",
                "question": "Question text goes here?",
                "correct_answer": "Correct option",
                "choices": ["Option 1", "Option 2", "Correct option", "Option 4"]
            }}
        ]
        Content:
        {peronality_prompt}
        """
        # generate questions
        questions = Prober.generate_model_response(prompt)
        return questions[0]

    @staticmethod
    def evaluate(questions:list,  responses:list):
        """ 
        Evaluate responses based on questions 

        Returns: dict of results and scor
        """
        
        results = []
        score = 0
        total_score = 0
        for i, q in enumerate(questions):
            expected_answer = q[0]
            system_answer = responses[i] if i < len(responses) else ""
            
            score = 1 if system_answer.strip().lower() == expected_answer.strip().lower() else 0
            total_score += score
            
            results.append({
                "question": q.get("question", ""),
                "expected_answer": expected_answer,
                "system_answer": system_answer,
                "score": score
            })
        
        return {
            "results": results,
            "total_score": total_score,
            "max_score": len(questions)
        }
        
    @staticmethod
    def classify_reflection_relevancy(transcript, reflection, personality):
        prompt = f"""
        You are a JSON-only evaluator. Rate how relevant the reflection is to the dialogue and personality.

        Instructions:
        - Score each axis (personality, dialogue) from 0 (irrelevant) to 3 (strongly relevant).
        - Justify each score briefly.

        Input:
        Reflection: "{reflection}"

        Dialogue:
        {transcript}

        Personality:
        {personality}

        Respond ONLY with a JSON object in this format:
        {{
        "relevancy_scores": {{
            "personality": <0-3>,
            "dialogue": <0-3>
        }},
        "justification": {{
            "personality": "<short explanation>",
            "dialogue": "<short explanation>"
        }}
        }}
        """

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {'role': 'system', 'content': 'Do not make any comment. Make sure to follow the instructions and reply in json and json only.'},
                {'role': 'user', 'content': prompt},
            ],
            format='json'
        )

        try:
            return json.loads(response['message']['content'])
        except json.JSONDecodeError as e:
            raise ValueError(e)

if __name__=='__main__':

    # test
    general_content = "Je pense que Jean est fiable. Je crois que Marie n'est pas ambitieuse. Le projet avance rapidement."
    try:
        prober = Prober()
        sk_questions = prober.generate_sk_questions()
        content_questions = prober.generate_content_questions(general_content)

        print(sk_questions)
        print("------------------------------------------------------")
        print(content_questions)

        
    except Exception as e:
        print(e)