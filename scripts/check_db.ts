import db from '../src/db.js';

const nodes = db.prepare('SELECT COUNT(*) as count FROM kg_nodes').get() as { count: number };
const edges = db.prepare('SELECT COUNT(*) as count FROM kg_edges').get() as { count: number };
const docs = db.prepare('SELECT COUNT(*) as count FROM documents').get() as { count: number };

console.log(`KG Nodes: ${nodes.count}`);
console.log(`KG Edges: ${edges.count}`);
console.log(`Documents: ${docs.count}`);
