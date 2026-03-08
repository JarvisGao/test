import { GoogleGenAI, Type } from '@google/genai';
import { task1App, updateKG } from './task1.js';
import { task2RAG } from './task2.js';
import { task3Agent } from './task3.js';

const getAi = () => new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

export async function routeQuery(messages: any[], onData?: (type: 'system' | 'content', text: string) => void) {
  const ai = getAi();
  
  // Extract the latest query
  const latestMessage = messages[messages.length - 1];
  const query = latestMessage.content;
  
  // Format history for context
  const historyContext = messages.slice(0, -1).map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`).join('\n');
  
  if (onData) onData('system', '🔍 正在分析用户意图...');
  
  const prompt = `
You are a router for a financial assistant system. Classify the user's query into one of three categories:
- "knowledge": The user is asking for financial knowledge, definitions, concepts, explanations, or specific company financial reports/summaries (e.g., "什么是市盈率？", "收入和净利润的区别是什么？", "某公司最近季度财报摘要是什么？").
- "asset": The user is asking for specific stock analysis, stock prices, historical data, or recent news about a specific stock or asset (e.g., "What is the price of AAPL?", "分析一下英伟达最近的股票表现").
- "api": General questions that require using external APIs to answer, or if it doesn't fit the other two.

Query: "${query}"
`;

  const response = await ai.models.generateContent({
    model: 'gemini-3.1-flash-lite-preview',
    contents: prompt,
    config: {
      responseMimeType: 'application/json',
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          route: {
            type: Type.STRING,
            enum: ['asset', 'knowledge', 'api'],
          }
        },
        required: ['route']
      }
    }
  });

  let route = 'api'; // default
  try {
    const data = JSON.parse(response.text || '{}');
    if (data.route) {
      route = data.route;
    }
  } catch (e) {
    console.error('Failed to parse router response', e);
  }

  console.log(`Routing query "${query}" to: ${route}`);
  if (onData) onData('system', `🎯 意图识别完成，分配给: ${route === 'asset' ? '股票分析专家' : route === 'knowledge' ? '金融知识库' : '实时工具代理'}`);

  if (route === 'asset') {
    if (onData) onData('system', '📈 启动股票分析流程...');
    const stream = await task1App.stream({ query, history: historyContext });
    let finalState: any = {};
    
    for await (const chunk of stream) {
      if (chunk.rewriteQuery) {
        finalState = { ...finalState, ...chunk.rewriteQuery };
        if (chunk.rewriteQuery.stockSymbol) {
          if (onData) onData('system', `🔠 提取到股票代码: ${chunk.rewriteQuery.stockSymbol}`);
        }
      }
      if (chunk.retrieveKG) {
        finalState = { ...finalState, ...chunk.retrieveKG };
        if (onData) onData('system', `📚 检索知识图谱上下文完成`);
      }
      if (chunk.fetchPrice) {
        finalState = { ...finalState, ...chunk.fetchPrice };
        if (onData) onData('system', `📊 获取到最新价格数据`);
      }
      if (chunk.fetchNews) {
        finalState = { ...finalState, ...chunk.fetchNews };
        if (onData) onData('system', `📰 获取到最新相关新闻`);
      }
      if (chunk.reason) {
        finalState = { ...finalState, ...chunk.reason };
        if (onData) onData('system', `💡 数据收集完毕，开始生成分析报告...`);
      }
    }

    if (finalState.messages && finalState.messages.length > 0) {
      const msg = finalState.messages[finalState.messages.length - 1];
      if (onData) onData('content', msg);
      return msg;
    } else if (finalState.reasonPrompt) {
      const response = await ai.models.generateContentStream({
        model: 'gemini-3.1-flash-lite-preview',
        contents: finalState.reasonPrompt,
      });
      let fullText = '';
      for await (const chunk of response) {
        if (chunk.text) {
          fullText += chunk.text;
          if (onData) onData('content', chunk.text);
        }
      }
      // update KG in background
      updateKG({ stockSymbol: finalState.stockSymbol, analysis: fullText }).catch(console.error);
      return fullText;
    } else {
      const msg = finalState.summary || 'No analysis could be generated.';
      if (onData) onData('content', msg);
      return msg;
    }
  } else if (route === 'knowledge') {
    return await task2RAG(query, historyContext, onData);
  } else {
    return await task3Agent(query, historyContext, onData);
  }
}
