import sqlite3
from werkzeug.security import generate_password_hash

def init_db():
    # Connect to SQLite database (creates it if it doesn't exist)
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Create a test user (you can modify these credentials)
    test_email = "test@example.com"
    test_password = "password123"
    
    try:
        cursor.execute(
            'INSERT INTO users (email, password) VALUES (?, ?)',
            (test_email, generate_password_hash(test_password))
        )
        print("Test user created successfully!")
    except sqlite3.IntegrityError:
        print("Test user already exists!")

    # Commit changes and close connection
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully!") 