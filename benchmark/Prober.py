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
    def generate_model_response(system, prompt):
        """
        Generate the content based on the prompt

        Returns: json 
        """

        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            answer = response.choices[0].message.content
        except Exception as e:
            
            response = ollama.generate(
                model='llama3:8b',
                system=system,
                prompt=prompt,
                format='json',
                options = {
                    "mirostat": 2,
                    "mirostat_tau": 7, 
                    "mirostat_eta": 0.1, 
                    "num_ctx": 2048,
                    "repeat_penalty": 1.3,
                    "presence_penalty": 1.4,
                    "frequency_penalty": 0.2,
                    "stop": ["<|endofjson|>"]
                }
            )

            answer = response['response']
        
        try: 
            questions = json.loads(answer)
            return list(questions.values())
            
        except json.JSONDecodeError as e:
            raise ValueError(e)

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
        
        """
        
        prompt = f"Format your response as a valid JSON. You should make the question ONLY about the following document Make sure to generate {num_questions} as asked and not more. Document:\n{content}"
        
        return Prober.generate_model_response(system, prompt)
        

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
            expected_answer = q['correct_answer']
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
        system = f"""
        You are a JSON-only evaluator. Rate how relevant the reflection is to the dialogue and personality.

        Instructions:
        - Score each axis (personality, dialogue) from 0 (irrelevant) to 3 (strongly relevant).
        - Justify each score briefly.

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
        
        prompt = f"""
        Format your response as a valid JSON. 
        You should score only considering reflection against dialogue and personality and be objective.
        
        Input reflection: "{reflection}"

        Dialogue:
        {transcript}

        Personality:
        {personality}
        """

        return Prober.generate_model_response(system, prompt)

if __name__=='__main__':

    # test
    general_content = "Je pense que Jean est fiable. Je crois que Marie n'est pas ambitieuse. Le projet avance rapidement."
    try:
        prober = Prober()
        # sk_questions = prober.generate_sk_questions()
        content_questions = prober.generate_content_questions(general_content)

        print(content_questions)

        
    except Exception as e:
        print(e)