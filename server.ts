import dotenv from 'dotenv';
dotenv.config({ override: true });
import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { createServer as createViteServer } from 'vite';
import db from './src/db.js'; // Initialize DB
import fs from 'fs';
import path from 'path';
import cron from 'node-cron';
import { routeQuery } from './src/agents/router.js';
import { populateInitialKnowledge } from './src/agents/task2.js';

// Initialize DB knowledge (moved to startServer)

const app = express();
const httpServer = createServer(app);
const PORT = 3000;

app.use(cors());
app.use(express.json());

// API routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Chat History API
app.get('/api/history', (req, res) => {
  try {
    const stmt = db.prepare('SELECT * FROM chat_history ORDER BY timestamp ASC');
    const history = stmt.all();
    // Parse logs from JSON string
    const parsedHistory = history.map((msg: any) => ({
      ...msg,
      logs: msg.logs ? JSON.parse(msg.logs) : []
    }));
    res.json(parsedHistory);
  } catch (error) {
    console.error('Error fetching history:', error);
    res.status(500).json({ error: 'Failed to fetch history' });
  }
});

// Serve README.md
app.get('/api/readme', (req, res) => {
  try {
    const readmePath = path.join(process.cwd(), 'README.md');
    if (fs.existsSync(readmePath)) {
      const content = fs.readFileSync(readmePath, 'utf-8');
      res.send(content);
    } else {
      res.status(404).send('README.md not found');
    }
  } catch (error) {
    console.error('Error serving README:', error);
    res.status(500).send('Failed to serve README');
  }
});

app.post('/api/history', (req, res) => {
  const { id, role, content, logs } = req.body;
  try {
    const stmt = db.prepare('INSERT OR REPLACE INTO chat_history (id, role, content, logs) VALUES (?, ?, ?, ?)');
    stmt.run(id, role, content, JSON.stringify(logs || []));
    res.json({ status: 'ok' });
  } catch (error) {
    console.error('Error saving history:', error);
    res.status(500).json({ error: 'Failed to save history' });
  }
});

app.delete('/api/history', (req, res) => {
  try {
    const stmt = db.prepare('DELETE FROM chat_history');
    stmt.run();
    res.json({ status: 'ok' });
  } catch (error) {
    console.error('Error clearing history:', error);
    res.status(500).json({ error: 'Failed to clear history' });
  }
});

// Chat endpoint (Streaming)
app.post('/api/chat', async (req, res) => {
  const { messages } = req.body;
  
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  
  try {
    await routeQuery(messages, (type, text) => {
      res.write(`data: ${JSON.stringify({ type, text })}\n\n`);
    });
    res.write(`data: [DONE]\n\n`);
  } catch (error: any) {
    console.error('Error processing chat:', error);
    res.write(`data: ${JSON.stringify({ type: 'error', text: error.message })}\n\n`);
    res.end();
  }
});

// Scheduled task (Task 2 requirement)
cron.schedule('0 0 * * *', () => {
  console.log('Running scheduled task to fetch financial data...');
  // Implement data fetching logic here
});

async function startServer() {
  console.log('Starting server...');
  console.log('GEMINI_API_KEY present:', !!process.env.GEMINI_API_KEY);
  
  try {
    await populateInitialKnowledge();
  } catch (error) {
    console.error('Failed to populate initial knowledge:', error);
  }
  
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { 
        middlewareMode: true,
        hmr: {
          server: httpServer,
          clientPort: 443,
        }
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
