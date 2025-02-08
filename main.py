import requests
from RAG_DB import RAGDB
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import time
import re

CODDER_MODEL = "qwen2.5-coder:3b"
DEEP_SEEK_MODEL = "deepseek-r1:1.5b"
CODDER_MODEL_BIG = "qwen2.5-coder:14b"
CODDER_MODEL_SMALL = "qwen2.5-coder:1.5b"
CODDER_MODEL_SUPPER_SMALL = "qwen2.5-coder:0.5b"
DEEP_SEEK_MODEL_BIG = "deepseek-r1:32b"

class OllamaRAG:
    def __init__(self, model_name: str = CODDER_MODEL, db_path: str = "ragV2.db", performance: bool = True , web_search: bool = True, context_search: bool = True):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        self.db = RAGDB(db_path)
        self.number_of_previous_conversations = 8
        self.generate_tags_for_resource()
        self.performance = performance
        self.web_search = web_search
        self.context_search = context_search

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        response = requests.post(self.api_url, json=payload)
        return response.json()['response']
    
    def add_resource(self, name: str, content: str) -> dict:
        """
        Adds a new resource to the database with an AI-generated description.

        - Generates a **concise and informative** description of the resource content.
        - Ensures the description is **clear, relevant, and useful** for future retrieval.

        Returns a dictionary containing the resource name, description, and tags.
        """

        # **Optimized Description Prompt**
        description_prompt = f"""
        You are an AI assistant responsible for summarizing resources for a knowledge database.

        ### Resource Content:
        {content}

        ### Instructions:
        - Provide a **concise and informative** summary of the resource.
        - The description should be **2-3 sentences long**.
        - Clearly convey the **main topic** and **key points** of the content.
        - Avoid unnecessary details but ensure the description is useful for search.

        ### Generated Description:
        """
        start = time.time()
        description = self._call_ollama(description_prompt)
        print("Average time for word generation: ", (time.time() - start) / len(description.split()))
        print(f"-"*15)

        tags_prompt = f"""
        You are an AI assistant responsible for generating relevant tags for a given resource.

        ### Resource Content:
        {content}

        ### Instructions:
        - Provide **3-7 highly relevant** tags that best describe the resource.
        - Tags should be **short, precise, and meaningful** (e.g., 'Machine Learning', 'Cybersecurity').
        - If the resource covers multiple topics, include diverse yet related tags.
        - **Avoid generic words** like "information", "article", or "document".

        ### Generated Tags (comma-separated):
        """

        start = time.time()
        tags = self._call_ollama(tags_prompt)
        print("Average time for word generation: ", (time.time() - start) / len(tags.split()))
        print(f"-"*15)


        # **Add resource to the database**
        self.db.add_resource(name, content, description, tags)

        return {"name": name, "description": description, "tags": tags}


    def _get_relevant_context(self, query: str, n_results: int = 3) -> str:
        """
        Retrieves the most relevant resources based on a query by:
        1. Generating a refined search description.
        2. Searching for resources in the database using the description and extracted keywords.
        3. Ranking the resources based on relevance.
        4. Returning the top `n_results` as a formatted string.
        """

        context = ""

        # **Step 1: Generate an Optimized Search Description**
        search_query_prompt = f"""
        You are an expert in information retrieval. Given the following user query, generate a concise and optimized search description that captures its key intent and meaning.

        ### User Query:
        {query}

        ### Requirements:
        - The description should be **1-2 sentences long**.
        - Avoid unnecessary words but retain key context.
        - Make it **search-friendly** by using commonly used terms.
        
        ### Optimized Search Description:
        """
        description = self._call_ollama(search_query_prompt)

        if self.performance == False:
            # **Step 2: Search for Initial Resources in the Database**
            resources = self.db.search_resources(description)

            # Convert tuples to a dictionary format
            res_list = []
            for r in resources:
                res_list.append({
                    'id': r[0],
                    'name': r[1],
                    'description': r[2],
                    'content': r[3],
                    'score': 0
                })

        # **Step 3: Generate Additional Keywords for Better Resource Discovery**
        keywords_prompt = f"""
        Given the following user query, generate **5 highly relevant search keywords** that can be used to find related information.

        ### User Query:
        {query}

        ### Requirements:
        - Provide **only** 5 keywords separated by commas (e.g., "AI, machine learning, deep learning, neural networks, NLP").
        - Focus on **key terms** that enhance search accuracy.
        - Prioritize **broad but meaningful** keywords.

        ### Keywords:
        """
        keywords = self._call_ollama(keywords_prompt).strip()

        # Split keywords and search for additional resources
        keywords = [keyword.strip() for keyword in keywords.split(',')]

        if self.performance == False:
            for keyword in keywords:
                more_res = self.db.search_resources(keyword)
                for mr in more_res:
                    res_list.append({
                        'id': mr[0],
                        'name': mr[1],
                        'description': mr[2],
                        'content': mr[3],
                        'score': 0
                    })

            # **Step 4: Rank Resources by Relevance Using LLM**
            for resource in res_list:
                rank_prompt = f"""
                You are an AI ranking system. Given the **user query** and a **resource**, assign a **relevance score from 1 to 100** based on how useful the resource is in answering the query.

                ### User Query:
                {query}

                ### Resource Content:
                {resource['content']}

                ### Resource Description:
                {resource['description']}

                ### Instructions:
                - Score must be **between 1 and 100**.
                - A **higher score (closer to 100)** means the resource is very relevant.
                - A **lower score (closer to 1)** means the resource is mostly irrelevant.
                - Provide **only** the numerical score (no explanations).

                ### Relevance Score:
                """
                r = self._call_ollama(rank_prompt)

                print(f"Resource: {resource['name']} - Score: {r}")
                # print(f"Description: {resource['description']}")

                # Extract numeric score from response
                resource['score'] = next((word for word in r.split() if word.isdigit()), "0")

            # **Step 5: Select Top `n_results` Resources**
            res_list = sorted(res_list, key=lambda x: float(x['score']), reverse=True)[:n_results]

            # **Step 6: Format and Return Context**
            context = "\n".join([f"{r['name']}: {r['content']}" for r in res_list])

        resV2 = self.db._search_resources_new(query, n_results, 2048)
        resV2 += self.db._search_resources_new(description, n_results, 2048)
        for k in keywords:
            resV2 += self.db._search_resources_new(k, n_results, 2048)

        seen = set()
        if resV2:
            resV2 = [x for x in resV2 if x['name'] not in seen and not seen.add(x['name'])]
            resV2 = sorted(resV2, key=lambda x: float(x['score']), reverse=True)[:n_results]
            resV2 = "\n\n".join([f"{r['name']}: {r['content']} : {r['description']}" for r in resV2])
        else:
            resV2 = ""
        

        print("-"*15)
        print("V2 DB query: ",resV2)
        print("-"*15)
        context += resV2
        
        return context

    
    def _find_resources_on_web(self, query: str, num_results: int = 3):
        try:
            # Search web using DuckDuckGo
            cleaned_query = query.replace('"', '').replace("'", "").strip()
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(cleaned_query, max_results=num_results)]
            
            web_resources = []
            for result in results:
                try:
                    # Rate limiting
                    time.sleep(0.1)
                    
                    # Get webpage content
                    url = result['href']
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get(url, headers=headers, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Remove unwanted elements
                    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                        tag.decompose()

                    # Get relevant content with more semantic tags
                    content_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'article', 'section', 'main'])
                    content = " ".join(tag.get_text(strip=True) for tag in content_tags)
                    
                    # Clean and normalize text
                    content = re.sub(r'\s+', ' ', content)  # Remove extra whitespace
                    content = re.sub(r'[^\w\s.,!?-]', '', content)  # Keep basic punctuation
                    content = content[:8196]  # Limit content length
                    
                    # Basic relevance check
                    if not any(term.lower() in content.lower() for term in query.split()):
                        continue

                    name = result['title']
                    print(f"Web Resource: {name} - {url}")
                    
                    if content.__contains__('404') or content.__contains__('Page not found') or content.__contains__('denied'):
                        continue

                    # Add to web resources to db for future reference
                    r = self.add_resource(name, content)

                    print(f"Resource added: {r}")
                    
                    web_resources.append({
                        'name': name,
                        'content': content,
                        'url': url,
                        'tags': r['tags'],
                        'description': r['description']
                    })
                    
                except requests.RequestException as e:
                    print(f"Request error for {url}: {str(e)}")
                    continue
                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")
                    continue
                    
            # Format context from web resources
            if web_resources:
                context = "\n\n".join([
                    f"From {r['url']} ({r['name']}): {r['content']}" 
                    for r in web_resources
                ])
                
                return context , web_resources
            
            return "" , []
            
        except Exception as e:
            print(f"Error searching web: {str(e)}")
            return "" , []
    
    def _summarize_previous_conversations(self) -> str:
        """
        Retrieves and summarizes the last N conversations with the user.

        - Extracts key topics, questions, and AI responses.
        - Ensures the summary is **concise yet informative**.
        - Provides a structured, easy-to-understand overview.
        """

        # **Fetch the last N conversations**
        conversations = self.db.get_last_n_conversations(self.number_of_previous_conversations)

        # **Format conversations into a structured transcript**
        conversation_log = "\n".join([
            f"User: {c[1]}\nAssistant: {c[2]}\n" for c in conversations
        ])

        # **Optimized Summary Prompt**
        summary_prompt = f"""
        You are an AI summarization assistant responsible for condensing past conversations.

        ### Previous {self.number_of_previous_conversations} Conversations:
        {conversation_log}

        ### Instructions:
        - Provide a **concise yet detailed** summary of key topics discussed.
        - Extract **important questions, AI responses, and any unresolved issues**.
        - The summary should be **coherent and structured**.
        - Avoid unnecessary details but retain meaningful context.
        
        ### Summary of Conversations:
        """
        
        summary = self._call_ollama(summary_prompt)
        return summary
    
    def generate_tags_for_resource(self):
        # iterate over all resources in the database if tags are not present then generate tags
        resources = self.db.resources_with_empty_tags()
        for resource in resources:
            tags_prompt = f"""
            You are an AI assistant responsible for generating relevant tags for a given resource.

            ### Resource Content:
            {resource[3]}

            ### Instructions:
            - Provide **3-7 highly relevant** tags that best describe the resource.
            - Tags should be **short, precise, and meaningful** (e.g., 'Machine Learning', 'Cybersecurity').
            - If the resource covers multiple topics, include diverse yet related tags.
            - **Avoid generic words** like "information", "article", or "document".

            ### Generated Tags (comma-separated):
            """

            tags = self._call_ollama(tags_prompt)
            self.db.update_tags(resource[0], tags)


    def chat(self, user_input: str):
        # Initialize empty context
        context = ""
        
        # Get relevant context from database only if enabled
        if self.context_search:
            context = self._get_relevant_context(user_input)
            print(f"Context: {context}")
            print("-"*15)

        if self.web_search:
            # create query for web search
            query_for_web = f"""
            Generate an effective and precise search query that can be used to find high-quality, relevant, and up-to-date information from the web. 
            The query should be optimized for search engines like Google and focus on retrieving authoritative sources, blogs, research papers, or forums.

            Context: {user_input}

            Ensure the query:
            - Uses relevant keywords and phrases.
            - Avoids unnecessary words or ambiguity.
            - Targets reputable sources for the best information.
            - Is structured concisely for accurate search results.

            Provide only the search query without additional explanation.
            """

            start = time.time()
            query_for_web = self._call_ollama(query_for_web)
            print("Average time for word generation: ", (time.time() - start) / len(query_for_web.split()))
            print(f"-"*15)

            context_from_web , resources = self._find_resources_on_web(query_for_web, num_results=5)
            # summarize the context from web
            web_summary_prompt = f"""
            You are an advanced AI assistant designed to extract key information from provided sources.

            ## **Task:**
            Summarize the most relevant and important insights from the retrieved web content to help answer the following user query:

            ### **User Query:**
            {user_input}

            ## **Web Content:**
            {context_from_web}

            ## **Instructions:**
            1. **Extract Key Insights:** Identify the most valuable information relevant to answering the user’s query.
            2. **Filter Out Irrelevant Details:** Remove redundant or off-topic content.
            3. **Summarize Clearly and Concisely:** Provide a structured summary in a few paragraphs or bullet points.
            4. **Prioritize Recent & Reliable Sources:** Highlight the most authoritative and up-to-date information.
            5. **Maintain Neutrality:** Present the summary in an objective manner without adding opinions.

            ## **Output Format:**
            Provide a structured summary that is easy to understand and directly useful in answering the user’s query.
            """
            start = time.time()
            context_from_web = self._call_ollama(web_summary_prompt)
            print("Average time for word generation: ", (time.time() - start) / len(context_from_web.split()))
            print(f"-"*15)
        
            # Update RAG prompt to include context only if it exists
            rag_prompt = f"""
            You are an intelligent assistant designed to provide accurate and context-aware responses. 

            {f'## **Context Information:**\n{context}\n' if context else ''}

            ## **Web Search Results:**
            {context_from_web}

            ## **User Conversation History (Last {self.number_of_previous_conversations} interactions):**
            {self._summarize_previous_conversations()}

            ### **Instructions:**
            1. **Synthesize Information:** Combine relevant details from the provided context, web search results, and past conversations.
            2. **Prioritize Relevance:** Focus on the most important and up-to-date facts to answer the question accurately.
            3. **Ensure Clarity & Conciseness:** The response should be clear, well-structured, and concise while covering key points.
            4. **Avoid Redundancy:** If a previous conversation already addressed this, summarize the past response and add new insights if necessary.
            5. **Cite Sources If Applicable:** If information is derived from web search results, indicate it subtly.

            ### **User Query:**
            {user_input}

            ### **Your Response:**
            """
        else:
            # Create RAG prompt without web search results
            rag_prompt = f"""
            You are an intelligent assistant designed to provide accurate and context-aware responses. 

            {f'## **Context Information From Database:**\n{context}\n' if context else ''}

            ## **User Conversation History (Last {self.number_of_previous_conversations} interactions):**
            {self._summarize_previous_conversations()}

            ### **Instructions:**
            1. **Synthesize Information:** Combine relevant details from the provided context and past conversations.
            2. **Prioritize Relevance:** Focus on the most important facts to answer the question accurately.
            3. **Ensure Clarity & Conciseness:** The response should be clear, well-structured, and concise while covering key points.
            4. **Avoid Redundancy:** If a previous conversation already addressed this, summarize the past response and add new insights if necessary.

            ### **User Query:**
            {user_input}

            ### **Your Response:**
            """

            resources = []

        # Get response from Ollama
        start = time.time()
        response = self._call_ollama(rag_prompt)
        print("Average time for word generation: ", (time.time() - start) / len(response.split()))
        
        # Store conversation in database
        self.db.add_conversation(user_input, response)

        info = {
            "performance": self.performance,
            "web_search": self.web_search,
            "model": self.model_name,
            "context_search": self.context_search
        }
        return response , resources , info

def main():
    # Initialize RAG system
    rag = OllamaRAG()
    
    # Add some sample resources
    # rag.db.add_resource(
    #     name="AI Definition",
    #     description="Basic definition of AI",
    #     content="Artificial Intelligence (AI) is the simulation of human intelligence by machines."
    # )
    
    # Interactive chat loop
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        response = rag.chat(user_input)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()