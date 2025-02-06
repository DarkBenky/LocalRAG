import sqlite3
from datetime import datetime
from typing import Optional, List, Tuple

class RAGDB:
    def __init__(self, db_path: str = "rag.db"):
        self.db_path = db_path
        self._create_tables()

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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

    def add_resource(self, name: str, content: str, description: Optional[str] = None) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO resources (name, description, content) VALUES (?, ?, ?)',
                (name, description, content)
            )
            return cursor.lastrowid

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
                'SELECT * FROM resources WHERE name LIKE ? OR description LIKE ? OR content LIKE ?',
                (f'%{query}%', f'%{query}%', f'%{query}%')
            )
            return cursor.fetchall()