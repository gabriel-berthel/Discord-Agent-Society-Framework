import os 
import asyncio
import aiohttp
import json 
import logging
import ollama
from dotenv import load_dotenv


load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WebBrowser:
    """
    Class for online research 
    """

    def __init__(self, search_api_key, search_engine_id, llm_api_key, use_ollama=True):
        self.search_api_key = search_api_key 
        self.search_engine_id = search_engine_id
        self.llm_api_key = llm_api_key
        self.use_ollama=use_ollama
        self.search_endpoint = "https://www.googleapis.com/customsearch/v1"
        self.llm_endpoint = "https://api.openai.com/v1/chat/completions"


    # fetch the web browser search results
    async def _fetch_search_results(self, query):
        params = {
            "key": self.search_api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": 5
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.search_endpoint, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Erreur lors de la recherche Google: {response.status}")
                        return {"error": f"Statut HTTP: {response.status}"}
        except Exception as e:
            logger.error(f"Exception lors de la recherche: {str(e)}")
            return {"error": str(e)} 



    # process the research results with llm
    async def _process_with_llm(self,query , search_results):
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        }

        websites = []
        if "items" in search_results:
            for item in search_results["items"]:
                websites.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })

        # llm prompt 
        prompt = f"""
        En te basant sur les résultats de recherche suivants, réponds à la question: "{query}"
        
        Résultats de recherche:
        {json.dumps(websites, ensure_ascii=False, indent=2)}
        
        Réponds de manière concise mais complète. Cite tes sources quand c'est pertinent.
        """

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Tu es un assistant de recherche web qui analyse les résultats et fournit des réponses précises."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.llm_endpoint, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "answer": result["choices"][0]["message"]["content"],
                            "sources": websites
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Erreur LLM: {response.status} - {error_text}")
                        return {"error": f"Erreur LLM: {response.status}"}
        except Exception as e:
            logger.error(f"Exception lors du traitement LLM: {str(e)}")
            return {"error": f"Exception LLM: {str(e)}"}
        

    # process the research results with llm
    async def _process_with_ollama(self, query, search_results):
        ollama_endpoint = "http://localhost:11434/api/generate"
    
        websites = []
        if "items" in search_results:
            for item in search_results["items"]:
                websites.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                }) 
                
        prompt = f"""
        En te basant sur les résultats de recherche suivants, réponds à la question: "{query}"
        
        Résultats de recherche:
        {json.dumps(websites, ensure_ascii=False, indent=2)}
        
        Réponds de manière concise mais complète. Cite tes sources quand c'est pertinent.
        """
        payload = {
            "model": "llama3", # or another ollama model
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3
            }
        }


        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(ollama_endpoint, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "answer": result.get("response", "Pas de réponse obtenue."),
                            "sources": websites
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Erreur Ollama: {response.status} - {error_text}")
                        return {"error": f"Erreur Ollama: {response.status}"}
        except Exception as e:
            logger.error(f"Exception lors du traitement Ollama: {str(e)}")
            return {"error": f"Exception Ollama: {str(e)}"}






    # combine the two functions for searching and treating using the llm
    async def _search(self, query):
        logger.info(f"Recherche Google démarrée pour: '{query}'")
    
        search_results = await self._fetch_search_results(query)
        
        if "error" in search_results:
            return {"error": search_results["error"]}
        
        logger.info(f"Résultats de recherche Google obtenus, traitement par LLM en cours...")
        
        if self.use_ollama:
            processed_results = await self._process_with_ollama(query, search_results)
        else:
            processed_results = await self._process_with_llm(query, search_results)
        
        if "error" in processed_results:
            return {"error": processed_results["error"]}
        
        logger.info(f"Recherche complétée avec succès")
        return processed_results
        
    
    # perform search and gives a brieve summary of the results
    async def summarize_search(self, query, max_length=150):
        logger.info(f"Génération d'un résumé concis pour la requête: '{query}'")

        full_results = await self._search(query)
        if "error" in full_results:
            return {"error": full_results["error"]}

        summarize_prompt = f"""
        Tu dois générer un résumé EXTRÊMEMENT CONCIS (maximum {max_length} caractères) de l'information suivante.
        Le résumé doit contenir uniquement les informations les plus essentielles et pertinentes.
        
        Information à résumer:
        {full_results["answer"]}
        
        IMPORTANT: Ton résumé ne doit pas dépasser {max_length} caractères.
        """

        try:
            if self.use_ollama:
                 response = ollama.chat(
                model="llama3",
                messages=[
                    {"role": "system", "content": "Tu es un assistant spécialisé dans la création de résumés concis."},
                    {"role": "user", "content": summarize_prompt}
                ],
                options={"temperature": 0.3}
            )
            summary = response['message']['content']
        
            logger.info(f"Résumé généré avec succès ({len(summary)} caractères)")
        
            return {
                "summary": summary,
                "full_answer": full_results["answer"],
                "sources": full_results["sources"]
            }
        
        except Exception as e:
            logger.error(f"Exception lors de la génération du résumé: {str(e)}")
            return {"error": f"Exception lors du résumé: {str(e)}"}



async def main():
    search_api_key=os.getenv("GOOGLE_API_KEY")
    search_engine_id=os.getenv("GOOGLE_CSE_ID")
    llm_api_key=os.getenv("LLM_API_KEY")


    browser = WebBrowser(search_api_key, search_engine_id, llm_api_key)

    # test 
    summary_results = await browser.summarize_search("Quelles sont les dernières avancées en intelligence artificielle?", max_length=150)
    print("\n=== RÉSUMÉ ===")
    print(summary_results["summary"])
    print("\n=== RÉPONSE COMPLÈTE ===")
    print(summary_results["full_answer"])


if __name__ == "__main__":
    asyncio.run(main())