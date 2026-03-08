import db from './src/db.js';
const count = db.prepare('SELECT COUNT(*) as count FROM documents').get();
console.log(count);
