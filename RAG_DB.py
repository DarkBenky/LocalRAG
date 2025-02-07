import sqlite3
from datetime import datetime
from typing import Optional, List, Tuple

class RAGDB:
    def __init__(self, db_path: str = "ragV2.db"):
        self.db_path = db_path
        self._create_tables()
        # self.migrateV1()

    def migrateV1(self):
        try:
            # load the old database rag.db
            with sqlite3.connect("rag.db") as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM resources')
                resources = cursor.fetchall()
                cursor.execute('SELECT * FROM conversations')
                conversations = cursor.fetchall()
            
            # migrate the resources
            for resource in resources:
                self.add_resource(resource[1], resource[2], resource[3], "")
            
            # migrate the conversations
            for conversation in conversations:
                self.add_conversation(conversation[1], conversation[2])
        except Exception as e:
            print(f"Error migrating database: {str(e)}")


    def _create_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create Resources table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS resources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT
                )
            ''')
            
            # Create Conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_input TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def add_resource(self, name: str, content: str, description: str , tags : str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO resources (name, description, content, tags) VALUES (?, ?, ?, ?)',
                (name, description, content, tags)
            )
            return cursor.lastrowid
        
    def update_tags(self, resource_id: int, tags: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE resources SET tags = ? WHERE id = ?',
                (tags, resource_id)
            )
    
    def resources_with_empty_tags(self) -> List[Tuple]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM resources WHERE tags IS NULL OR tags = ""')
            return cursor.fetchall()
        

    def add_conversation(self, user_input: str, assistant_response: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO conversations (user_input, assistant_response) VALUES (?, ?)',
                (user_input, assistant_response)
            )
            return cursor.lastrowid

    def get_resource(self, resource_id: int) -> Optional[Tuple]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM resources WHERE id = ?', (resource_id,))
            return cursor.fetchone()

    def get_conversation(self, conversation_id: int) -> Optional[Tuple]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM conversations WHERE id = ?', (conversation_id,))
            return cursor.fetchone()

    def get_all_resources(self) -> List[Tuple]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM resources')
            return cursor.fetchall()

    def get_all_conversations(self) -> List[Tuple]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM conversations')
            return cursor.fetchall()
        
    def get_last_n_conversations(self, n: int) -> List[Tuple]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM conversations ORDER BY created_at DESC LIMIT ?', (n,))
            return cursor.fetchall()
        

    def search_resources(self, query: str) -> List[Tuple]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM resources WHERE name LIKE ? OR description LIKE ? OR content LIKE ? OR tags LIKE ?',
                (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%')
            )
            return cursor.fetchall()
        
    def _search_resources_new(self, query: str, n_results: int = 3) -> str:
        try:
            # Tokenize and clean query
            query_terms = set(word.lower() for word in query.split())
            
            # Get resources from SQLite DB
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT name, content, tags FROM resources')
                resources = cursor.fetchall()
            
            res_list = []
            for name, content, tags in resources:
                # Calculate relevance scores
                content_terms = set(word.lower() for word in content.split())
                name_terms = set(word.lower() for word in name.split())
                tag_terms = set(word.lower() for word in (tags or '').split(','))
                
                # Weighted scoring
                name_matches = len(query_terms & name_terms) * 3.0  # Higher weight for name
                content_matches = len(query_terms & content_terms)
                tag_matches = len(query_terms & tag_terms) * 2.0    # Medium weight for tags
                
                total_score = name_matches + content_matches + tag_matches
                
                # Include if matches found
                if total_score > 0:
                    # Limit content length for display
                    display_content = content[:500] + '...' if len(content) > 500 else content
                    
                    res_list.append({
                        'name': name,
                        'content': display_content,
                        'score': total_score,
                        'tags': tags or ''
                    })

            # Sort by score and limit results
            res_list = sorted(res_list, key=lambda x: float(x['score']), reverse=True)[:n_results]

            # Format results with metadata
            context = "\n\n".join([
                f"Source: {r['name']} (Score: {r['score']:.1f})\n"
                f"Tags: {r['tags']}\n"
                f"Content: {r['content']}" 
                for r in res_list
            ])
            
            return context if res_list else ""

        except Exception as e:
            print(f"Search error: {str(e)}")
            return ""