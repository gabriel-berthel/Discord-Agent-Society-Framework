import asyncio
import json
from prompt_client import PromptClient
from modules.DiscordServer import DiscordServer
from dotenv import load_dotenv

# Liste des archétypes à tester
ARCHETYPES = ['baseline', 'trouble_maker', 'fact_checker', 'activist', 'moderator']

PROMPTS = [
    "Salut girl",
    "Comment vas tu ?",
    "Quelle est la difference entre la science de donnee et la valorisation des donnees",
    "Est ce que tu connais Bang Liu"
    "Que pensez-vous de l'IA dans l'éducation ?",
    "Pouvez-vous expliquer le changement climatique en termes simples ?",
    "L'exploration spatiale vaut-elle l'investissement ?",
    "Quelle est votre opinion sur le télétravail ?",
    "Comment la société devrait-elle gérer la désinformation ?",
    "Pouvez-vous décrire l'avenir des énergies renouvelables ?",
    "Quels sont les risques de l'IA ?",
    "Les gouvernements devraient-ils réglementer les réseaux sociaux ?",
    "Parlez-moi de l'impact de la technologie sur la santé mentale.",
    "Que pensez-vous du revenu universel de base ?"
]

# Fonction principale
async def run_benchmark():
    # Initialiser le serveur Discord simulé
    server = DiscordServer(1, 'Benchmarking', 1)
    server.update_user(1, 'User')
    server.add_channel(1, 'General')

    results = []

    # Pour chaque archétype, créer un agent et faire les tests
    for archetype in ARCHETYPES:
        print(f"\nTesting archetype: {archetype}")

        agent = PromptClient('benchmark_config.yaml', archetype, f"{archetype}_agent", 1, server)

        # Override get_bot_context pour forcer la réponse
        def forced_context():
            return "Toujours répondre quoi qu'il arrive.."

        agent.agent.get_bot_context = forced_context

        await agent.start()

        # Pour chaque prompt, envoyer le message et collecter la réponse
        for prompt in PROMPTS:
            print(f"Prompt: {prompt}")

            # Envoyer le message comme un utilisateur
            server.add_message(1, 1, 'User', prompt)
            response = await agent.prompt(prompt, 1, 'User')

            print(f"Response: {response}")

            results.append({
                'archetype': archetype,
                'input': prompt,
                'response': response
            })

            await asyncio.sleep(0.5)  # petite pause pour la stabilité

    # Sauvegarder les résultats dans un fichier JSON
    with open('benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nBenchmark terminé. Résultats sauvegardés dans 'benchmark_results.json'.")

if __name__ == '__main__':
    asyncio.run(run_benchmark())
