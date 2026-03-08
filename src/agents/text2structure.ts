import { GoogleGenAI, Type } from '@google/genai';
import db from '../db.js';

const getAi = () => new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

export async function extractKnowledgeGraph(text: string) {
  const ai = getAi();
  
  const prompt = `
  请从以下金融文本中提取知识图谱实体和关系。
  实体（Nodes）应该包含：id（唯一标识符，如"市盈率"）、label（实体类型，如"概念"、"指标"、"公司"等）。
  关系（Edges）应该包含：source（源实体id）、target（目标实体id）、relationship（关系类型，如"属于"、"包含"、"用于评估"等）。
  
  文本内容：
  ${text}
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3.1-flash-lite-preview',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            nodes: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  label: { type: Type.STRING },
                  properties: { type: Type.STRING, description: "JSON string of additional properties if any" }
                },
                required: ["id", "label"]
              }
            },
            edges: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  source: { type: Type.STRING },
                  target: { type: Type.STRING },
                  relationship: { type: Type.STRING },
                  properties: { type: Type.STRING, description: "JSON string of additional properties if any" }
                },
                required: ["source", "target", "relationship"]
              }
            }
          },
          required: ["nodes", "edges"]
        }
      }
    });

    const data = JSON.parse(response.text || '{}');
    return data;
  } catch (e) {
    console.error('Failed to extract KG:', e);
    return { nodes: [], edges: [] };
  }
}

export async function processDocumentsToKG() {
  console.log('Starting to process documents into Knowledge Graph...');
  
  const docs = db.prepare('SELECT id, content FROM documents WHERE kg_extracted = 0 LIMIT 15').all() as { id: number, content: string }[];
  
  const insertNode = db.prepare('INSERT OR IGNORE INTO kg_nodes (id, label, properties) VALUES (?, ?, ?)');
  const insertEdge = db.prepare('INSERT OR IGNORE INTO kg_edges (source, target, relationship, properties) VALUES (?, ?, ?, ?)');
  const updateDoc = db.prepare('UPDATE documents SET kg_extracted = 1 WHERE id = ?');

  for (const doc of docs) {
    console.log(`Processing document ID ${doc.id} for KG extraction...`);
    const kgData = await extractKnowledgeGraph(doc.content);
    
    if (kgData.nodes && kgData.nodes.length > 0) {
      for (const node of kgData.nodes) {
        insertNode.run(node.id, node.label, node.properties || '{}');
      }
    }
    
    if (kgData.edges && kgData.edges.length > 0) {
      for (const edge of kgData.edges) {
        insertEdge.run(edge.source, edge.target, edge.relationship, edge.properties || '{}');
      }
    }
    
    updateDoc.run(doc.id);
    
    // Add a 4-second delay to avoid hitting the 15 RPM Gemini API free tier rate limit
    await new Promise(resolve => setTimeout(resolve, 4000));
  }
  
  console.log('Finished processing documents into Knowledge Graph.');
}
