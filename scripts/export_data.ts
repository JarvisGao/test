import db from '../src/db.js';
import fs from 'fs';
import path from 'path';

const dataDir = path.join(process.cwd(), 'src', 'data');
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}

const nodes = db.prepare('SELECT * FROM kg_nodes').all();
const edges = db.prepare('SELECT * FROM kg_edges').all();
const documents = db.prepare('SELECT * FROM documents').all();

const data = {
  nodes,
  edges,
  documents
};

fs.writeFileSync(path.join(dataDir, 'initial_knowledge.json'), JSON.stringify(data, null, 2));

console.log(`Exported ${nodes.length} nodes, ${edges.length} edges, and ${documents.length} documents to src/data/initial_knowledge.json`);
