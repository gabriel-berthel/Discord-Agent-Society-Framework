import os 
import asyncio
import aiohttp
import json 
import logging
import ollama

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebBrowser:
    """
    Class for online research 
    """

    def __init__(self, use_ollama=True):
        self.search_api_key = os.getenv("GOOGLE_API_KEY") 
        self.search_engine_id = os.getenv("GOOGLE_CSE_ID")
        self.llm_api_key = os.getenv("LLM_API_KEY")
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
        
        print(query, params)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.search_endpoint, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Error during Google search: {response.status}")
                        return {"error": f"HTTP Status: {response.status}"}
        except Exception as e:
            logger.error(f"Exception during search: {str(e)}")
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
        Based on the following search results, answer the question: "{query}"
        
        Search results:
        {json.dumps(websites, ensure_ascii=False, indent=2)}
        
        Answer concisely but completely. Cite your sources when relevant.
        """

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a web research assistant who analyzes results and provides accurate answers."},
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
                        logger.error(f"LLM Error: {response.status} - {error_text}")
                        return {"error": f"LLM Error: {response.status}"}
        except Exception as e:
            logger.error(f"Exception during LLM processing: {str(e)}")
            return {"error": f"LLM Exception: {str(e)}"}
        

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
        Based on the following search results, answer the question: "{query}"
        
        Search results:
        {json.dumps(websites, ensure_ascii=False, indent=2)}
        
        Answer concisely but completely. Cite your sources when relevant.
        """
        payload = {
            "model": "llama3.2", # or another ollama model
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
                            "answer": result.get("response", "No response obtained."),
                            "sources": websites
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Ollama Error: {response.status} - {error_text}")
                        return {"error": f"Ollama Error: {response.status}"}
        except Exception as e:
            logger.error(f"Exception during Ollama processing: {str(e)}")
            return {"error": f"Ollama Exception: {str(e)}"}



    # combine the two functions for searching and treating using the llm
    async def _search(self, query):
        logger.info(f"Google search started for: '{query}'")
    
        search_results = await self._fetch_search_results(query)
        
        if "error" in search_results:
            return {"error": search_results["error"]}
        
        logger.info(f"Google search results obtained, LLM processing in progress...")
        
        if self.use_ollama:
            processed_results = await self._process_with_ollama(query, search_results)
        else:
            processed_results = await self._process_with_llm(query, search_results)
        
        if "error" in processed_results:
            return {"error": processed_results["error"]}
        
        logger.info(f"Search completed successfully")
        return processed_results
        
    
    # perform search and gives a brieve summary of the results
    async def summarize_search(self, queries, max_length=300):
        if isinstance(queries, str):
            queries = [queries]
            
        logger.info(f"Generating a combined summary for {len(queries)} queries")
        
        search_tasks = [self._search(query) for query in queries]
        search_results = await asyncio.gather(*search_tasks)
        
        errors = [result["error"] for result in search_results if "error" in result]
        if errors:
            return {"error": f"Errors in {len(errors)} queries: {', '.join(errors[:3])}" + 
                    ("..." if len(errors) > 3 else "")}
        
        combined_answers = []
        all_sources = []
        
        for i, (query, result) in enumerate(zip(queries, search_results)):
            combined_answers.append(f"Query: {query}\nAnswer: {result['answer']}")
            all_sources.extend(result["sources"])
            
        combined_text = "\n\n".join(combined_answers)
        
        summarize_prompt = f"""
        You must generate an EXTREMELY CONCISE summary (maximum {max_length} characters) of the following information 
        from multiple search queries.
        
        The summary should synthesize the key findings across all queries and highlight common themes or contradictions.
        Only include the most essential and relevant information.
        
        Information to summarize:
        {combined_text}
        
        IMPORTANT: Your summary must not exceed {max_length} characters.
        """

        try:
            if self.use_ollama:
                response = ollama.chat(
                    model="llama3.2",
                    messages=[
                        {"role": "system", "content": "You are an assistant specialized in creating concise summaries of multiple research findings."},
                        {"role": "user", "content": summarize_prompt}
                    ],
                    options={"temperature": 0.3}
                )
                summary = response['message']['content']
            
                logger.info(f"Combined summary generated successfully ({len(summary)} characters)")
            
                return {
                    "summary": summary,
                    "full_answers": {queries[i]: result["answer"] for i, result in enumerate(search_results)},
                    "sources": all_sources
                }
            
        except Exception as e:
            logger.error(f"Exception during summary generation: {str(e)}")
            return {"error": f"Summary Exception: {str(e)}"}


async def main():
    search_api_key=os.getenv("GOOGLE_API_KEY")
    search_engine_id=os.getenv("GOOGLE_CSE_ID")
    llm_api_key=os.getenv("LLM_API_KEY")


    browser = WebBrowser(search_api_key, search_engine_id, llm_api_key)

    # Test with multiple queries
    queries = [
        "What are the latest advances in artificial intelligence?",
        "How is AI being used in healthcare?",
        "What are the ethical concerns with AI development?"
    ]
    
    summary_results = await browser.summarize_search(queries, max_length=300)

    # test 
    summary_results = await browser.summarize_search("What are the latest advances in artificial intelligence?", max_length=150)
    print("\n=== SUMMARY ===")
    print(summary_results["summary"])
    print("\n=== FULL ANSWER ===")
    print(summary_results["full_answer"])


if __name__ == "__main__":
    asyncio.run(main())