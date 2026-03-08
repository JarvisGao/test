import { useState, useEffect } from 'react';
import { Send, Bot, BookOpen, MessageSquare } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import ReadmePage from './components/ReadmePage';

type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  logs?: string[];
};

type View = 'chat' | 'readme';

export default function App() {
  const [view, setView] = useState<View>('chat');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  // Load history from database on mount
  useEffect(() => {
    fetch('/api/history')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) {
          setMessages(data);
        }
      })
      .catch(err => console.error('Failed to load history:', err));
  }, []);

  const saveMessageToDb = async (msg: Message) => {
    try {
      await fetch('/api/history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(msg),
      });
    } catch (error) {
      console.error('Failed to save message to DB:', error);
    }
  };

  const clearHistory = async () => {
    try {
      await fetch('/api/history', { method: 'DELETE' });
      setMessages([]);
    } catch (error) {
      console.error('Failed to clear history:', error);
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
    };

    // Save user message immediately
    saveMessageToDb(userMsg);

    const assistantMsgId = (Date.now() + 1).toString();
    const initialAssistantMsg: Message = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
    };

    const newMessages = [...messages, userMsg];
    setMessages([...newMessages, initialAssistantMsg]);
    setInput('');
    setLoading(true);

    try {
      // Send only necessary fields to the server
      const messagesToSend = newMessages.map(({ role, content }) => ({ role, content }));
      
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: messagesToSend }),
      });
      
      if (!res.body) throw new Error('No response body');
      
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let buffer = '';
      let currentAssistantContent = '';
      let currentLogs: string[] = [];

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep the last partial line in the buffer
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.slice(6);
              if (dataStr === '[DONE]') {
                done = true;
                break;
              }
              try {
                const data = JSON.parse(dataStr);
                if (data.type === 'system') {
                  currentLogs.push(data.text);
                  setMessages((prev) => 
                    prev.map((msg) => 
                      msg.id === assistantMsgId 
                        ? { ...msg, logs: [...(msg.logs || []), data.text] } 
                        : msg
                    )
                  );
                } else if (data.type === 'content') {
                  currentAssistantContent += data.text;
                  setMessages((prev) => 
                    prev.map((msg) => 
                      msg.id === assistantMsgId 
                        ? { ...msg, content: msg.content + data.text } 
                        : msg
                    )
                  );
                } else if (data.type === 'error') {
                  currentAssistantContent += '\n\n**Error:** ' + data.text;
                  setMessages((prev) => 
                    prev.map((msg) => 
                      msg.id === assistantMsgId 
                        ? { ...msg, content: msg.content + '\n\n**Error:** ' + data.text } 
                        : msg
                    )
                  );
                }
              } catch (e) {
                console.error('Error parsing SSE data:', e, dataStr);
              }
            }
          }
        }
      }
      
      // Save assistant message to DB after completion
      saveMessageToDb({
        id: assistantMsgId,
        role: 'assistant',
        content: currentAssistantContent,
        logs: currentLogs
      });

    } catch (error) {
      console.error('Error sending message:', error);
      setMessages((prev) => 
        prev.map((msg) => 
          msg.id === assistantMsgId 
            ? { ...msg, content: 'Failed to connect to the server.' } 
            : msg
        )
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-zinc-50 text-zinc-900 font-sans flex-col">
      {/* Header */}
      <header className="bg-zinc-900 text-zinc-100 p-4 shadow-md flex items-center justify-between z-10">
        <div className="flex items-center gap-2">
          <Bot size={24} className="text-emerald-400" />
          <h1 className="text-xl font-bold font-serif tracking-tight">FinQ&A 智能金融助手</h1>
        </div>
        <div className="flex items-center gap-4">
          <button 
            onClick={() => setView('chat')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md transition-colors ${view === 'chat' ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-white'}`}
          >
            <MessageSquare size={18} />
            <span className="text-sm font-medium">对话</span>
          </button>
          <button 
            onClick={() => setView('readme')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md transition-colors ${view === 'readme' ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-white'}`}
          >
            <BookOpen size={18} />
            <span className="text-sm font-medium">系统架构</span>
          </button>
          {view === 'chat' && (
            <button 
              onClick={clearHistory}
              className="text-sm text-zinc-400 hover:text-red-400 transition-colors ml-4 border-l border-zinc-700 pl-4"
            >
              清空对话
            </button>
          )}
        </div>
      </header>

      {/* Main Content Area */}
      {view === 'readme' ? (
        <div className="flex-1 overflow-y-auto bg-zinc-50">
          <ReadmePage />
        </div>
      ) : (
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6 max-w-4xl mx-auto w-full">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-zinc-400">
                <Bot size={48} className="mb-4 opacity-50" />
                <p className="text-lg font-medium text-zinc-600">欢迎使用智能金融助手</p>
                <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm w-full max-w-lg">
                  <div className="bg-white p-4 rounded-xl border border-zinc-200 shadow-sm">
                    <p className="font-semibold text-zinc-700 mb-2">📈 股票分析</p>
                    <p>"英伟达的最新股价是多少？"</p>
                    <p>"分析一下AAPL最近的股票表现"</p>
                  </div>
                  <div className="bg-white p-4 rounded-xl border border-zinc-200 shadow-sm">
                    <p className="font-semibold text-zinc-700 mb-2">📚 金融知识</p>
                    <p>"什么是市盈率？"</p>
                    <p>"收入和净利润的区别是什么？"</p>
                    <p>"某公司最近季度财报摘要是什么？"</p>
                  </div>
                </div>
              </div>
            ) : (
              messages.map((msg) => (
                <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] p-4 rounded-2xl ${msg.role === 'user' ? 'bg-zinc-900 text-white rounded-br-sm' : 'bg-white shadow-sm border border-zinc-200 rounded-bl-sm'}`}>
                    {msg.role === 'assistant' ? (
                      <div className="flex flex-col gap-3">
                        {msg.logs && msg.logs.length > 0 && (
                          <details className="text-xs text-zinc-500 bg-zinc-50 border border-zinc-200 p-2 rounded-md cursor-pointer group">
                            <summary className="font-medium hover:text-zinc-700 select-none">
                              查看执行路径 ({msg.logs.length} 步)
                            </summary>
                            <div className="mt-2 flex flex-col gap-1.5 pl-3 border-l-2 border-zinc-300 py-1">
                              {msg.logs.map((log, i) => (
                                <div key={i} className="font-mono">{log}</div>
                              ))}
                            </div>
                          </details>
                        )}
                        <div className="prose prose-sm prose-zinc max-w-none">
                          <ReactMarkdown 
                            remarkPlugins={[remarkMath]} 
                            rehypePlugins={[rehypeKatex]}
                          >
                            {msg.content}
                          </ReactMarkdown>
                        </div>
                      </div>
                    ) : (
                      <p>{msg.content}</p>
                    )}
                  </div>
                </div>
              ))
            )}
            {messages.length > 0 && loading && messages[messages.length - 1].content === '' && (
              <div className="flex justify-start">
                <div className="bg-white shadow-sm border border-zinc-200 p-4 rounded-2xl rounded-bl-sm flex items-center gap-2">
                  <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  <div className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }} />
                </div>
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="p-4 bg-white border-t border-zinc-200">
            <div className="max-w-4xl mx-auto relative flex items-center">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                placeholder="询问股票数据、金融知识或市场新闻..."
                className="w-full bg-zinc-100 border-transparent focus:bg-white focus:border-zinc-300 focus:ring-2 focus:ring-zinc-900 rounded-xl py-4 pl-4 pr-12 transition-all outline-none"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || loading}
                className="absolute right-2 p-2 bg-zinc-900 text-white rounded-lg hover:bg-zinc-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
