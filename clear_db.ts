import db from './src/db.js';
db.prepare('DELETE FROM documents').run();
console.log('Documents table cleared.');
