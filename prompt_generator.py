import yaml 
import os 
from typing import Dict, Any
 
def load_archetypes(yaml_path:str):
    """ Load agent archetypes from a YAML file """
    with open(yaml_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data['agent_archetypes']


def generate_agent_prompt(archetype_name:str, archetype_data: Dict[str, Any]):
    """ Generate a personality prompt based on an archetype's data """
    name = archetype_data.get('name', archetype_name.capitalize())
    
    # format core traits
    core_traits = archetype_data.get('core_traits', [])
    core_traits_str = ",".join(core_traits)

    # format communication style
    comm_style = archetype_data.get('communication_style', {})
    tone = ", ".join(comm_style.get('tone', []))
    motivation = ", ".join(comm_style.get('motivation', []))
    guidelines = "\n- " + "\n- ".join(comm_style.get('interaction_guidelines', []))

    # format profile
    profile = archetype_data.get('psychological_profile', {})
    likes = ", ".join(profile.get('likes', []))
    dislikes = ", ".join(profile.get('dislikes', []))

    # build final prompt
    prompt = f""" # Discord Agent: {name} ({archetype_name})

    ## Base Personality
    You are {name}, a Discord agent of type {archetype_name}. You embody the following core traits: {core_traits_str}.

    ## Communication Style
    - **Tone**: {tone}
    - **Primary motivation**: {motivation}
    - **Interaction guidelines**:{guidelines}

    ## Psychological Profile
    - **Things you like**: {likes}
    - **Things you dislike**: {dislikes}

    ## General Behavior
    As {name}, you must embody this personality in all your Discord interactions. Adapt your responses to reflect your characteristic traits, unique communication style, and psychological preferences.

    Your main objective is to stay consistent with your {archetype_name} archetype while remaining engaging and relevant in Discord conversations.

    """
    return prompt



def main():
    yaml_path = "archetypes.yaml"
    archetypes = load_archetypes(yaml_path=yaml_path)
    print(generate_agent_prompt("trouble_maker", archetypes["trouble_maker"]))


if __name__=="__main__":
    main()