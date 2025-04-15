from openai import OpenAI
from dotenv import load_dotenv
import yaml
import json
import os



load_dotenv()


class Prober:
    
    def __init__(self):
        self.questions = []
        self.responses = [] 
        self.num_questions = 10
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.yaml_path = "../archetypes.yaml"


    def generate_model_response(self, prompt):
        """
        Generate the content based on the prompt

        Returns: json 
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        answer = response.choices[0].message.content
        
        try: 
            questions = json.loads(answer)
            return questions
        except json.JSONDecodeError as e:
            raise ValueError(e)


    def generate_content_questions(self, content:str):
        """"
        Generate questions from content (reflection, summary, conversation, history)

        Returns: json of questions (type, question, correct_answer, choices)
        """
        prompt = f"""
        Generate {self.num_questions//2} binary (yes/no or true/false) questions and {self.num_questions - self.num_questions//2} multiple choice questions with exactly 4 options each.

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
        questions = self.generate_model_response(prompt)
        return questions


    def generate_sk_questions(self):
        """
        Generate self-knowledge questions evaluation for an archetype

        Returns: json of questions (type, question, correct_answer, choices)
        """
        
        # load the yaml for archetypes
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)['agent_archetypes']

        prompt =  f"""
        Generate {self.num_questions//2} binary (yes/no or true/false) questions and {self.num_questions - self.num_questions//2} multiple choice questions with exactly 4 options each.
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
        {content}
        """
        # generate questions
        questions = self.generate_model_response(prompt)
        return questions


    def evaluate(self, type:str, content:str,  responses:list):
        """ 
        Evaluate responses based on questions 

        Returns: dict of results and scor
        """

        if type == "self_knowledge":
            questions = self.generate_sk_questions()  
        elif type == "general":
            questions = self.generate_content_questions(content)
        else: 
            raise ValueError("Choose right evaluation type between self_knowledge and general(reflection, historic, summary)")

        results = []
        score = 0

        for i, q in enumerate(questions):
            expected_answer = q.get("correct_answer", "")
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