import { GoogleGenAI, Type, FunctionDeclaration } from '@google/genai';
import YahooFinance from 'yahoo-finance2';
const yahooFinance = new YahooFinance();

const getAi = () => new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Define tools
const getStockPriceTool: FunctionDeclaration = {
  name: 'getStockPrice',
  description: 'Get the current stock price for a given symbol.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      symbol: { type: Type.STRING, description: 'The stock symbol (e.g., AAPL)' }
    },
    required: ['symbol']
  }
};

const getCompanyNewsTool: FunctionDeclaration = {
  name: 'getCompanyNews',
  description: 'Get recent news for a given company symbol.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      symbol: { type: Type.STRING, description: 'The stock symbol (e.g., AAPL)' }
    },
    required: ['symbol']
  }
};

export async function task3Agent(query: string, history: string, onData?: (type: 'system' | 'content', text: string) => void) {
  const ai = getAi();
  if (onData) onData('system', '🛠️ 正在规划工具调用...');
  const prompt = `You are a financial assistant. Use tools to answer the user's question. If the user provides a company name (e.g., "英伟达", "Apple"), you must first figure out its stock symbol (e.g., "NVDA", "AAPL") before calling the tools. Your final answer MUST be in Chinese. 
  
Chat History:
${history}

Question: ${query}`;
  
  const response = await ai.models.generateContent({
    model: 'gemini-3.1-flash-lite-preview',
    contents: prompt,
    config: {
      tools: [{ functionDeclarations: [getStockPriceTool, getCompanyNewsTool] }]
    }
  });

  const functionCalls = response.functionCalls;
  
  if (functionCalls && functionCalls.length > 0) {
    const call = functionCalls[0];
    if (onData) onData('system', `⚙️ 正在调用工具: ${call.name}(${JSON.stringify(call.args)})`);
    let toolResult = '';
    
    try {
      if (call.name === 'getStockPrice') {
        const symbol = (call.args as any).symbol;
        const quote: any = await yahooFinance.quote(symbol);
        toolResult = `Current price of ${symbol} is ${quote.regularMarketPrice} ${quote.currency}.`;
        if (onData) onData('system', `✅ 获取到价格: ${quote.regularMarketPrice} ${quote.currency}`);
      } else if (call.name === 'getCompanyNews') {
        const symbol = (call.args as any).symbol;
        const news: any = await yahooFinance.search(symbol, { newsCount: 3 });
        toolResult = `Recent news for ${symbol}: ${news.news.map((n: any) => n.title).join('; ')}`;
        if (onData) onData('system', `✅ 获取到新闻数据`);
      }
    } catch (e) {
      toolResult = `Error executing tool ${call.name}: ${e}`;
      if (onData) onData('system', `❌ 工具调用失败: ${e}`);
    }

    if (onData) onData('system', '🧠 工具调用完成，正在总结回答...');

    // Second call to summarize tool result
    const secondResponse = await ai.models.generateContentStream({
      model: 'gemini-3.1-flash-lite-preview',
      contents: `Chat History: ${history}\nUser asked: ${query}\nTool result: ${toolResult}\nProvide a final answer in Chinese.`,
    });
    
    let fullText = '';
    for await (const chunk of secondResponse) {
      if (chunk.text) {
        fullText += chunk.text;
        if (onData) onData('content', chunk.text);
      }
    }
    
    return fullText || 'No final answer generated.';
  }

  if (onData) onData('system', '💡 无需调用工具，直接生成回答...');
  if (response.text && onData) onData('content', response.text);
  return response.text || 'No tools were called and no answer generated.';
}
