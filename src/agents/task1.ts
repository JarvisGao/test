import { StateGraph, Annotation, END, START } from '@langchain/langgraph';
import { GoogleGenAI, Type } from '@google/genai';
import YahooFinance from 'yahoo-finance2';
const yahooFinance = new YahooFinance();
import db from '../db.js';

const getAi = () => new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Define the state schema
export const GraphState = Annotation.Root({
  query: Annotation<string>(),
  history: Annotation<string>(),
  stockSymbol: Annotation<string>(),
  priceData: Annotation<any>(),
  newsData: Annotation<any>(),
  analysis: Annotation<string>(),
  reasonPrompt: Annotation<string>(),
  kgContext: Annotation<string>(),
  messages: Annotation<string[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
  summary: Annotation<string>(),
});

// 1. Query Rewriter & Symbol Extractor
async function rewriteQuery(state: typeof GraphState.State) {
  const ai = getAi();
  const response = await ai.models.generateContent({
    model: 'gemini-3.1-flash-lite-preview',
    contents: `Extract the stock symbol from the following query. If the query contains a company name (e.g., "英伟达", "Apple", "微软"), translate it to its corresponding stock symbol (e.g., "NVDA", "AAPL", "MSFT"). If the query is unclear, ask for clarification.
    
Chat History:
${state.history}

Current Query: ${state.query}`,
    config: {
      responseMimeType: 'application/json',
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          symbol: { type: Type.STRING, description: 'The stock symbol (e.g., AAPL, BABA)' },
          clarificationNeeded: { type: Type.BOOLEAN },
          clarificationMessage: { type: Type.STRING }
        },
        required: ['symbol', 'clarificationNeeded']
      }
    }
  });
  
  const result = JSON.parse(response.text || '{}');
  return {
    stockSymbol: result.symbol || '',
    messages: result.clarificationNeeded ? [result.clarificationMessage] : []
  };
}

// 2. KG Retriever
async function retrieveKG(state: typeof GraphState.State) {
  if (!state.stockSymbol) return { kgContext: '' };
  
  const stmt = db.prepare('SELECT * FROM kg_nodes WHERE id = ?');
  const node = stmt.get(state.stockSymbol);
  
  return {
    kgContext: node ? JSON.stringify(node) : 'No context found in KG.'
  };
}

// 3. Price Agent
async function fetchPrice(state: typeof GraphState.State) {
  if (!state.stockSymbol) return { priceData: null };
  
  try {
    const quote: any = await yahooFinance.quote(state.stockSymbol);
    
    // Get historical data with a buffer to ensure we find a valid trading day
    const endDate = new Date();
    
    // For 7d change: Look back 14 days to ensure we cover weekends/holidays
    const startDate7dBuffer = new Date();
    startDate7dBuffer.setDate(endDate.getDate() - 14);
    
    // For 30d change: Look back 45 days
    const startDate30dBuffer = new Date();
    startDate30dBuffer.setDate(endDate.getDate() - 45);
    
    // Fetch historical data
    const hist7dData: any[] = await yahooFinance.historical(state.stockSymbol, { period1: startDate7dBuffer, period2: endDate });
    const hist30dData: any[] = await yahooFinance.historical(state.stockSymbol, { period1: startDate30dBuffer, period2: endDate });
    
    // Helper to find the closing price closest to N days ago (but not after)
    const getPriceDaysAgo = (data: any[], daysAgo: number) => {
        if (!data || data.length === 0) return null;
        
        const targetDate = new Date();
        targetDate.setDate(targetDate.getDate() - daysAgo);
        targetDate.setHours(23, 59, 59, 999); 
        
        let bestEntry = null;
        for (const entry of data) {
            const entryDate = new Date(entry.date);
            if (entryDate <= targetDate) {
                bestEntry = entry;
            } else {
                break; 
            }
        }
        
        return bestEntry;
    };

    const entry7dAgo = getPriceDaysAgo(hist7dData, 7);
    const entry30dAgo = getPriceDaysAgo(hist30dData, 30);
    
    // Use adjClose if available, otherwise close
    const price7dAgo = entry7dAgo ? (entry7dAgo.adjClose || entry7dAgo.close) : (hist7dData[0]?.adjClose || hist7dData[0]?.close);
    const price30dAgo = entry30dAgo ? (entry30dAgo.adjClose || entry30dAgo.close) : (hist30dData[0]?.adjClose || hist30dData[0]?.close);
    
    const currentPrice = quote.regularMarketPrice;

    const priceData = {
      currentPrice: currentPrice,
      currency: quote.currency,
      change7d: (price7dAgo && currentPrice) ? ((currentPrice - price7dAgo) / price7dAgo) * 100 : 0,
      change30d: (price30dAgo && currentPrice) ? ((currentPrice - price30dAgo) / price30dAgo) * 100 : 0,
      // Debug info to help verify
      debug: {
        price7dAgo,
        price30dAgo,
        date7dAgo: entry7dAgo ? new Date(entry7dAgo.date).toISOString().split('T')[0] : 'N/A',
        date30dAgo: entry30dAgo ? new Date(entry30dAgo.date).toISOString().split('T')[0] : 'N/A',
        targetDate30dAgo: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
      }
    };
    
    return { priceData };
  } catch (error) {
    console.error('Error fetching price:', error);
    return { priceData: { error: 'Failed to fetch price data.' } };
  }
}

// 4. News Agent
async function fetchNews(state: typeof GraphState.State) {
  if (!state.stockSymbol) return { newsData: null };
  
  try {
    const news: any = await yahooFinance.search(state.stockSymbol, { newsCount: 5 });
    return { newsData: news.news };
  } catch (error) {
    console.error('Error fetching news:', error);
    return { newsData: { error: 'Failed to fetch news data.' } };
  }
}

// 5. Reasoning Agent
async function reason(state: typeof GraphState.State) {
  if (!state.stockSymbol || !state.priceData) return { reasonPrompt: 'Cannot analyze without data.' };
  
  const prompt = `
You are a financial analyst. Analyze the following data for ${state.stockSymbol}.
Your response MUST be in Chinese.

Chat History:
${state.history}

Price Data: ${JSON.stringify(state.priceData)}
News Data: ${JSON.stringify(state.newsData)}
KG Context: ${state.kgContext}

Provide a structured, professional, data-driven analysis in Chinese.
1. State the current price and changes (7d, 30d).
2. **CRITICAL**: Explicitly state the calculation basis for transparency. 
   - Format: "7日涨跌幅: [Current Price] vs [Price 7 Days Ago] (on [Date]) = [Change]%"
   - Format: "30日涨跌幅: [Current Price] vs [Price 30 Days Ago] (on [Date]) = [Change]%"
   - Use the data from the 'debug' field in Price Data.
3. Summarize the trend (Up/Down/Sideways).
4. Analyze possible reasons based on the news and context.
Distinguish between objective data and analytical description.
Use Chain of Thought reasoning.
`;

  return { reasonPrompt: prompt };
}

// 6. KG Writer
export async function updateKG(state: { stockSymbol?: string, analysis?: string }) {
  if (!state.stockSymbol || !state.analysis) return {};
  
  try {
    const stmt = db.prepare('INSERT OR REPLACE INTO kg_nodes (id, label, properties) VALUES (?, ?, ?)');
    stmt.run(state.stockSymbol, 'Company', JSON.stringify({
      lastAnalysis: state.analysis,
      lastUpdated: new Date().toISOString()
    }));
  } catch (error) {
    console.error('Error updating KG:', error);
  }
  return {};
}

// 7. Summarizer
async function summarize(state: typeof GraphState.State) {
  return { summary: state.reasonPrompt };
}

// Build the graph
const workflow = new StateGraph(GraphState)
  .addNode('rewriteQuery', rewriteQuery)
  .addNode('retrieveKG', retrieveKG)
  .addNode('fetchPrice', fetchPrice)
  .addNode('fetchNews', fetchNews)
  .addNode('reason', reason)
  .addNode('summarize', summarize)
  
  .addEdge(START, 'rewriteQuery')
  .addEdge('rewriteQuery', 'retrieveKG')
  .addEdge('retrieveKG', 'fetchPrice')
  .addEdge('fetchPrice', 'fetchNews')
  .addEdge('fetchNews', 'reason')
  .addEdge('reason', 'summarize')
  .addEdge('summarize', END);

export const task1App = workflow.compile();
