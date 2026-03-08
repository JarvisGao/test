import { GoogleGenAI, Type } from '@google/genai';
import { pipeline } from '@xenova/transformers';
import db from '../db.js';
import googlethis from 'googlethis';
import { HNSW } from 'hnsw';
import { processDocumentsToKG } from './text2structure.js';

const getAi = () => new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Singleton for the embedding pipeline
let extractor: any = null;
async function getExtractor() {
  if (!extractor) {
    // Use a lightweight embedding model
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  return extractor;
}

export async function getEmbedding(text: string): Promise<number[]> {
  const extract = await getExtractor();
  const output = await extract(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

// HNSW Index Singleton
let hnswIndex: HNSW | null = null;
let hnswIdToDocId = new Map<number, number>();

export async function initHNSW() {
  if (hnswIndex) return;
  console.log('Initializing HNSW Index...');
  hnswIndex = new HNSW(16, 200, 384, 'cosine'); // 384 is dim for all-MiniLM-L6-v2
  
  const stmtAll = db.prepare(`SELECT id, embedding FROM documents WHERE embedding IS NOT NULL`);
  const docs = stmtAll.all() as { id: number, embedding: string }[];
  
  let internalId = 0;
  for (const doc of docs) {
    const embedding = JSON.parse(doc.embedding);
    await hnswIndex.addPoint(internalId, embedding);
    hnswIdToDocId.set(internalId, doc.id);
    internalId++;
  }
  console.log(`HNSW Index built with ${docs.length} vectors.`);
}

function cosineSimilarity(vecA: number[], vecB: number[]) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// 1. Knowledge Graph Search
async function searchKG(query: string) {
  const ai = getAi();
  
  // Extract entities from query
  const prompt = `提取以下问题中的核心金融实体（如公司名、指标名、概念等）。只返回实体名称，用逗号分隔。问题：${query}`;
  const response = await ai.models.generateContent({
    model: 'gemini-3.1-flash-lite-preview',
    contents: prompt
  });
  
  const entities = response.text?.split(',').map(e => e.trim()) || [];
  if (entities.length === 0) return null;

  let kgContext = '';
  const stmt = db.prepare(`
    SELECT source, relationship, target 
    FROM kg_edges 
    WHERE source LIKE ? OR target LIKE ?
    LIMIT 10
  `);

  for (const entity of entities) {
    const edges = stmt.all(`%${entity}%`, `%${entity}%`) as { source: string, relationship: string, target: string }[];
    for (const edge of edges) {
      kgContext += `- ${edge.source} ${edge.relationship} ${edge.target}\n`;
    }
  }

  return kgContext.trim().length > 0 ? kgContext : null;
}

// 2. Hybrid Search (BM25 + Embedding)
async function hybridSearch(query: string) {
  // BM25 Search (FTS5)
  // Clean query for FTS5 (remove special chars)
  const cleanQuery = query.replace(/[^a-zA-Z0-9\u4e00-\u9fa5]/g, ' ').trim();
  let bm25Results: { id: number, rank: number }[] = [];
  
  if (cleanQuery) {
    try {
      // Create match query by splitting words and joining with OR
      const matchQuery = cleanQuery.split(/\s+/).join(' OR ');
      const stmt = db.prepare(`
        SELECT rowid as id, bm25(documents_fts) as rank 
        FROM documents_fts 
        WHERE documents_fts MATCH ? 
        ORDER BY rank LIMIT 10
      `);
      bm25Results = stmt.all(matchQuery) as { id: number, rank: number }[];
    } catch (e) {
      console.error("FTS5 search error:", e);
    }
  }

  // Embedding Search using HNSW
  if (!hnswIndex) {
    await initHNSW();
  }
  
  const queryEmbedding = await getEmbedding(query);
  let topVectorResults: { id: number, score: number, content: string }[] = [];
  
  if (hnswIndex) {
    const knn = hnswIndex.searchKNN(queryEmbedding, 10);
    
    // Fetch content for the results
    const stmtGetDoc = db.prepare(`SELECT content FROM documents WHERE id = ?`);
    topVectorResults = knn.map(res => {
      const docId = hnswIdToDocId.get(res.id)!;
      const doc = stmtGetDoc.get(docId) as { content: string };
      return { id: docId, score: res.score, content: doc.content };
    });
  }

  // Reciprocal Rank Fusion (RRF)
  const rrfScores = new Map<number, { score: number, content: string }>();
  const k = 60;

  // Add BM25 ranks
  bm25Results.forEach((res, index) => {
    const stmtGetDoc = db.prepare(`SELECT content FROM documents WHERE id = ?`);
    const doc = stmtGetDoc.get(res.id) as { content: string };
    if (doc) {
      rrfScores.set(res.id, { score: 1 / (k + index + 1), content: doc.content });
    }
  });

  // Add Vector ranks
  topVectorResults.forEach((res, index) => {
    if (rrfScores.has(res.id)) {
      rrfScores.get(res.id)!.score += 1 / (k + index + 1);
    } else {
      rrfScores.set(res.id, { score: 1 / (k + index + 1), content: res.content });
    }
  });

  // Apply Cosine Similarity Threshold to filter out low-relevance results
  const SIMILARITY_THRESHOLD = 0.55;
  const stmtGetEmbedding = db.prepare(`SELECT embedding FROM documents WHERE id = ?`);
  
  const validDocs = [];
  for (const [docId, data] of rrfScores.entries()) {
    try {
      const row = stmtGetEmbedding.get(docId) as { embedding: string };
      if (row && row.embedding) {
        const docEmbedding = JSON.parse(row.embedding);
        const sim = cosineSimilarity(queryEmbedding, docEmbedding);
        if (sim >= SIMILARITY_THRESHOLD) {
          validDocs.push(data);
        }
      }
    } catch (e) {
      console.error("Error checking similarity threshold:", e);
    }
  }

  // Sort by RRF score
  validDocs.sort((a, b) => b.score - a.score);
  
  if (validDocs.length === 0) {
    return null; // Trigger fallback to Google Search
  }
  
  // Return top 3
  return validDocs.slice(0, 3).map(d => d.content).join('\n\n');
}

// 3. Google Search Fallback
async function googleSearchFallback(query: string) {
  try {
    const options = {
      page: 0, 
      safe: false,
      additional_params: { hl: 'zh-CN' }
    };
    const response = await googlethis.search(query, options);
    
    // Extract top 5 snippets
    const topResults = response.results.slice(0, 5);
    let context = '来自Google搜索的补充信息：\n';
    topResults.forEach((res: any, index: number) => {
      context += `${index + 1}. [${res.title}] ${res.description}\n`;
    });
    return context;
  } catch (e) {
    console.error("Google search failed:", e);
    return null;
  }
}

export async function task2RAG(query: string, history: string, onData?: (type: 'system' | 'content', text: string) => void) {
  const ai = getAi();
  let context = '';
  let source = '';

  if (onData) onData('system', '📚 正在检索本地知识图谱...');
  // Step 1: Try Knowledge Graph
  const kgContext = await searchKG(query);
  if (kgContext) {
    context = kgContext;
    source = '知识图谱 (Knowledge Graph)';
    if (onData) onData('system', '✅ 成功从知识图谱获取上下文');
  } else {
    if (onData) onData('system', '🧠 知识图谱未命中，正在进行向量数据库混合检索...');
    // Step 2: Try Hybrid Search
    const hybridContext = await hybridSearch(query);
    if (hybridContext && hybridContext.trim().length > 0) {
      context = hybridContext;
      source = '本地向量数据库 (Hybrid Search)';
      if (onData) onData('system', '✅ 成功从向量数据库获取上下文');
    } else {
      if (onData) onData('system', '🌐 本地知识库未命中，正在使用 Google 搜索补充最新信息...');
      // Step 3: Fallback to Google Search
      const webContext = await googleSearchFallback(query);
      if (webContext) {
        context = webContext;
        source = 'Google Web Search';
        if (onData) onData('system', '✅ 成功从 Google 搜索获取上下文');
      }
    }
  }

  if (onData) onData('system', `📝 检索完成，来源: ${source || '无'}。正在生成专业回答...`);

  const prompt = `
你是一位顶级的华尔街金融分析师和财富管理专家。请基于以下提供的上下文，以专业、客观、严谨的口吻回答用户的问题。

【回答要求】
1. **专业性**：使用准确的金融术语，逻辑清晰，分析透彻。
2. **结构化**：使用 Markdown 格式（如加粗、列表、分段）让回答易于阅读。
3. **数据引用**：如果上下文中包含具体数据或指标，请务必在回答中引用。
4. **诚实性**：如果上下文中没有包含答案，请使用你的通用金融知识进行解答，但**必须**在开头明确声明：“*注：以下信息基于通用金融知识，非本地知识库检索结果。*”
5. **来源标注**：请在回答的结尾，附上信息来源（见下方“信息来源”字段）。

【信息来源】: ${source || '无'}

【检索到的上下文】:
${context || '没有找到相关上下文。'}

【聊天历史记录】:
${history || '无'}

【用户问题】: 
${query}

请开始你的专业解答：
`;

  const config: any = {
    systemInstruction: '你是一位专业的金融分析师，必须使用中文回答，保持客观、严谨的分析态度。',
  };

  const response = await ai.models.generateContentStream({
    model: 'gemini-3.1-flash-lite-preview',
    contents: prompt,
    config,
  });

  let fullText = '';
  for await (const chunk of response) {
    if (chunk.text) {
      fullText += chunk.text;
      if (onData) onData('content', chunk.text);
    }
  }

  return fullText || '无法生成回答。';
}

// Function to populate some initial knowledge
export async function populateInitialKnowledge() {
  const countStmt = db.prepare('SELECT COUNT(*) as count FROM documents');
  const { count } = countStmt.get() as { count: number };
  
  if (count === 0) {
    // Try to load from local JSON file first
    const fs = await import('fs');
    const path = await import('path');
    const dataPath = path.join(process.cwd(), 'src', 'data', 'initial_knowledge.json');
    
    if (fs.existsSync(dataPath)) {
      console.log('Loading initial knowledge from local JSON file...');
      const data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
      
      const insertDoc = db.prepare('INSERT INTO documents (id, content, metadata, embedding, kg_extracted) VALUES (?, ?, ?, ?, ?)');
      const insertNode = db.prepare('INSERT OR IGNORE INTO kg_nodes (id, label, properties) VALUES (?, ?, ?)');
      const insertEdge = db.prepare('INSERT OR IGNORE INTO kg_edges (source, target, relationship, properties) VALUES (?, ?, ?, ?)');
      
      const updateDoc = db.prepare('UPDATE documents SET kg_extracted = 1');

      db.transaction(() => {
        for (const doc of data.documents) {
          insertDoc.run(doc.id, doc.content, doc.metadata, doc.embedding, doc.kg_extracted);
        }
        for (const node of data.nodes) {
          insertNode.run(node.id, node.label, node.properties);
        }
        for (const edge of data.edges) {
          insertEdge.run(edge.source, edge.target, edge.relationship, edge.properties);
        }
        updateDoc.run();
      })();
      
      console.log(`Restored ${data.documents.length} documents, ${data.nodes.length} nodes, and ${data.edges.length} edges from local file.`);
      return;
    }

    const insertStmt = db.prepare('INSERT INTO documents (content, metadata, embedding) VALUES (?, ?, ?)');
    
    const knowledgeBase = [
      {
        content: '市盈率（Price-to-Earnings Ratio，简称P/E或PE）是指股票价格除以每股收益（EPS）的比率。它常用来评估股价水平是否合理。高市盈率可能意味着股票被高估，或者投资者预期未来盈利会有高增长。',
        topic: 'PE Ratio'
      },
      {
        content: '收入（Revenue）是指企业在日常活动中形成的、会导致所有者权益增加的、与所有者投入资本无关的经济利益的总流入。它是利润表的第一项，也称为“顶线”（Top Line）。净利润（Net Income）是指企业当期利润总额减去所得税后的金额，即企业的税后利润，也称为“底线”（Bottom Line）。两者的主要区别在于，收入是扣除任何费用之前的总额，而净利润是扣除所有成本、费用和税费后的最终盈余。',
        topic: 'Revenue vs Net Income'
      },
      {
        content: '市净率（Price-to-Book Ratio，简称P/B）是股票市价与每股净资产的比值。市净率较低的股票，投资价值较高，相反，则投资价值较低。',
        topic: 'PB Ratio'
      },
      {
        content: 'ROE（净资产收益率，Return on Equity）是净利润与平均股东权益的百分比，是公司盈利能力的重要指标。ROE越高，说明投资带来的收益越高。',
        topic: 'ROE'
      },
      {
        content: '股息率（Dividend Yield）是一年的总派息额与当时市价的比例。它是挑选收益型股票的重要参考标准。',
        topic: 'Dividend Yield'
      }
    ];

    for (const item of knowledgeBase) {
      const embedding = await getEmbedding(item.content);
      
      insertStmt.run(
        item.content,
        JSON.stringify({ topic: item.topic }),
        JSON.stringify(embedding)
      );
    }
    
    console.log('Initial knowledge populated with local embeddings.');
    
    // Pre-populate Knowledge Graph with hardcoded data (Avoids API calls on startup)
    const insertNode = db.prepare('INSERT OR IGNORE INTO kg_nodes (id, label, properties) VALUES (?, ?, ?)');
    const insertEdge = db.prepare('INSERT OR IGNORE INTO kg_edges (source, target, relationship, properties) VALUES (?, ?, ?, ?)');
    
    const initialNodes = [
      { id: '市盈率', label: '指标' },
      { id: '股价', label: '概念' },
      { id: '每股收益', label: '概念' },
      { id: '收入', label: '概念' },
      { id: '净利润', label: '概念' },
      { id: '市净率', label: '指标' },
      { id: '每股净资产', label: '概念' },
      { id: 'ROE', label: '指标' },
      { id: '股东权益', label: '概念' },
      { id: '股息率', label: '指标' },
      { id: '派息额', label: '概念' }
    ];

    const initialEdges = [
      { source: '市盈率', target: '股价', relationship: '用于评估' },
      { source: '市盈率', target: '每股收益', relationship: '计算依赖' },
      { source: '收入', target: '净利润', relationship: '区别于' },
      { source: '市净率', target: '每股净资产', relationship: '计算依赖' },
      { source: 'ROE', target: '净利润', relationship: '计算依赖' },
      { source: 'ROE', target: '股东权益', relationship: '计算依赖' },
      { source: '股息率', target: '派息额', relationship: '计算依赖' },
      { source: '股息率', target: '股价', relationship: '计算依赖' }
    ];

    const updateDoc = db.prepare('UPDATE documents SET kg_extracted = 1');

    db.transaction(() => {
      for (const node of initialNodes) insertNode.run(node.id, node.label, '{}');
      for (const edge of initialEdges) insertEdge.run(edge.source, edge.target, edge.relationship, '{}');
      updateDoc.run();
    })();

    console.log('Knowledge Graph pre-populated with local data.');
  }
}
