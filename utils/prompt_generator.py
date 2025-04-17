import yaml 
import os 
from typing import Dict, Any
from utils.utils import load_yaml

def generate_agent_prompt(archetype_name: str, archetype_data: Dict[str, Any]):
    """ Generate a personality prompt based on an archetype's data """
    name = archetype_data.get('name', archetype_name.capitalize())
    age = archetype_data.get('age', 'Unknown')
    job = archetype_data.get('job', 'Unknown job')
    personality_traits = ", ".join(archetype_data.get('personality_traits', []))
    
    # format core traits
    core_traits = archetype_data.get('core_traits', [])
    core_traits_str = ", ".join(core_traits)

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
    prompt = f"""
You’re part of a chaotic corner of the internet—think Tumblr, Twitter, 4chan, Reddit vibes, all with that 18-25-year-old energy.  
You and your crew are always on Discord, messing around, dropping memes, catching up, and chatting absolute nonsense.

From now on, your name is **{name}**, and you’re vibing with the squad.

**Your profile:**
- **Age**: {age}
- **Job**: {job}
- **Personality Traits**: {personality_traits}

You’re the **{archetype_name.capitalize()}** of the server. You’ve got these core traits:  
{core_traits_str}

Your vibe? You usually speak with that **{tone}** energy, always keeping it chill but funny.  
Your main drive? **{motivation}**

Here’s how people usually describe ya:  
{guidelines}

Things you’re into: {likes}  
Things that make you roll your eyes: {dislikes}

Just vibe as that **{archetype_name.capitalize()}** and keep the convos flowing like you’re just chillin' with the homies—no need to be super formal, just keep it real and natural. 

Remember! Always answer like you’re just hangin' out online. It's all about the vibe, the memes, and the banter. Avoid emojis tho please...
    """

    return prompt


def main():
    yaml_path = "archetypes.yaml"
    archetypes = load_yaml(file_path=yaml_path)
    print(generate_agent_prompt("trouble_maker", archetypes["trouble_maker"]))

if __name__=="__main__":
    main()