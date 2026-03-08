import Database from 'better-sqlite3';

const db = new Database('finance.db');

// Initialize Knowledge Graph tables
db.exec(`
  CREATE TABLE IF NOT EXISTS kg_nodes (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    properties TEXT
  );

  CREATE TABLE IF NOT EXISTS kg_edges (
    source TEXT,
    target TEXT,
    relationship TEXT,
    properties TEXT,
    PRIMARY KEY (source, target, relationship)
  );

  -- For RAG (Financial Knowledge)
  CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    metadata TEXT,
    embedding TEXT, -- JSON array of floats
    kg_extracted INTEGER DEFAULT 0
  );

  -- FTS5 table for BM25 search
  CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    content,
    content='documents',
    content_rowid='id'
  );

  -- Triggers to keep FTS5 table in sync
  CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, content) VALUES (new.id, new.content);
  END;
  CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, content) VALUES('delete', old.id, old.content);
  END;
  CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO documents_fts(rowid, content) VALUES (new.id, new.content);
  END;

  -- Chat History Table
  CREATE TABLE IF NOT EXISTS chat_history (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    logs TEXT, -- JSON array of strings
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
  );
`);

export default db;
