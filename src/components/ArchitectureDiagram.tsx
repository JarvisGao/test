import React from 'react';
import { User, BrainCircuit, Database, Globe, Search, FileText, ArrowRight, Share2, Layers, Cpu } from 'lucide-react';

const Node = ({ icon: Icon, title, subtitle, color = "blue", className = "" }: any) => (
  <div className={`flex flex-col items-center justify-center p-4 bg-white rounded-xl border-2 shadow-sm z-10 w-48 transition-transform hover:scale-105 ${className}`}
       style={{ borderColor: `var(--color-${color}-500, #3b82f6)` }}>
    <div className={`p-3 rounded-full mb-3 bg-${color}-100 text-${color}-600`}>
      <Icon size={24} />
    </div>
    <h3 className="font-bold text-zinc-800 text-sm text-center">{title}</h3>
    {subtitle && <p className="text-xs text-zinc-500 text-center mt-1">{subtitle}</p>}
  </div>
);

const Label = ({ text }: { text: string }) => (
  <div className="bg-white px-2 py-1 text-[10px] font-mono text-zinc-500 border border-zinc-200 rounded shadow-sm z-20">
    {text}
  </div>
);

export default function ArchitectureDiagram() {
  return (
    <div className="w-full overflow-x-auto p-8 bg-zinc-50 rounded-xl border border-zinc-200 my-8">
      <div className="min-w-[800px] flex flex-col gap-12 relative">
        
        {/* Layer 1: User Input */}
        <div className="flex justify-center relative">
          <Node icon={User} title="User Input" subtitle="Natural Language Query" color="zinc" />
          <div className="absolute top-full h-12 w-0.5 bg-zinc-300 left-1/2 -translate-x-1/2"></div>
          <div className="absolute top-[calc(100%+24px)] left-1/2 -translate-x-1/2">
            <ArrowRight className="text-zinc-400 rotate-90" size={20} />
          </div>
        </div>

        {/* Layer 2: Router Agent */}
        <div className="flex justify-center relative">
          <Node icon={BrainCircuit} title="Router Agent" subtitle="Intent Classification (Gemini 3.1 Flash Lite)" color="purple" />
        </div>

        {/* Connecting Lines Router -> Tasks */}
        <div className="relative h-16 w-full">
          {/* Horizontal Line */}
          <div className="absolute top-0 left-[20%] right-[20%] h-8 border-x-2 border-t-2 border-zinc-300 rounded-t-2xl"></div>
          {/* Vertical Center Line (hidden/transparent to space out) */}
          <div className="absolute top-0 left-1/2 h-8 w-0.5 bg-zinc-300 -translate-x-1/2"></div>
          
          {/* Arrows */}
          <div className="absolute bottom-0 left-[20%] -translate-x-1/2"><ArrowRight className="text-zinc-400 rotate-90" size={20} /></div>
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2"><ArrowRight className="text-zinc-400 rotate-90" size={20} /></div>
          <div className="absolute bottom-0 right-[20%] translate-x-1/2"><ArrowRight className="text-zinc-400 rotate-90" size={20} /></div>

          {/* Labels */}
          <div className="absolute top-2 left-[30%] -translate-x-1/2"><Label text="asset" /></div>
          <div className="absolute top-2 left-1/2 -translate-x-1/2 bg-white z-10"><Label text="knowledge" /></div>
          <div className="absolute top-2 right-[30%] translate-x-1/2"><Label text="tool/api" /></div>
        </div>

        {/* Layer 3: Task Agents */}
        <div className="grid grid-cols-3 gap-8 relative">
          
          {/* Task 1: Asset Agent */}
          <div className="flex flex-col gap-4 p-4 bg-blue-50/50 rounded-2xl border border-blue-100 relative group">
            <div className="absolute -top-3 left-4 bg-blue-100 text-blue-700 px-2 py-0.5 text-xs font-bold rounded">Task 1: Asset Agent</div>
            <Node icon={Share2} title="LangGraph Workflow" subtitle="State Machine" color="blue" className="w-full" />
            
            <div className="flex flex-col gap-2 pl-4 border-l-2 border-blue-200 ml-6 py-2">
              <div className="flex items-center gap-2 text-xs text-zinc-600"><div className="w-2 h-2 rounded-full bg-blue-400"></div>Query Rewriter</div>
              <div className="flex items-center gap-2 text-xs text-zinc-600"><div className="w-2 h-2 rounded-full bg-blue-400"></div>KG Retriever (History)</div>
              <div className="flex items-center gap-2 text-xs text-zinc-600"><div className="w-2 h-2 rounded-full bg-blue-400"></div>Yahoo Finance API</div>
              <div className="flex items-center gap-2 text-xs text-zinc-600"><div className="w-2 h-2 rounded-full bg-blue-400"></div>Reasoning (CoT)</div>
            </div>
          </div>

          {/* Task 2: Knowledge Agent (RAG) */}
          <div className="flex flex-col gap-4 p-4 bg-emerald-50/50 rounded-2xl border border-emerald-100 relative">
            <div className="absolute -top-3 left-4 bg-emerald-100 text-emerald-700 px-2 py-0.5 text-xs font-bold rounded">Task 2: RAG Agent</div>
            <Node icon={Database} title="Hybrid Search" subtitle="KG + Vector + BM25" color="emerald" className="w-full" />
            
            <div className="grid grid-cols-2 gap-2 mt-2">
              <div className="p-2 bg-white rounded border border-emerald-100 text-center">
                <div className="text-[10px] font-bold text-emerald-600">SQLite FTS5</div>
                <div className="text-[9px] text-zinc-400">Keyword Search</div>
              </div>
              <div className="p-2 bg-white rounded border border-emerald-100 text-center">
                <div className="text-[10px] font-bold text-emerald-600">HNSW Index</div>
                <div className="text-[9px] text-zinc-400">Vector Search</div>
              </div>
            </div>

            <div className="flex justify-center my-1"><ArrowRight className="text-emerald-300 rotate-90" size={16} /></div>

            <Node icon={Globe} title="Google Fallback" subtitle="If local search fails" color="orange" className="w-full scale-90" />
          </div>

          {/* Task 3: Tool Agent */}
          <div className="flex flex-col gap-4 p-4 bg-amber-50/50 rounded-2xl border border-amber-100 relative">
            <div className="absolute -top-3 left-4 bg-amber-100 text-amber-700 px-2 py-0.5 text-xs font-bold rounded">Task 3: Tool Agent</div>
            <Node icon={Cpu} title="Function Calling" subtitle="Tool Execution" color="amber" className="w-full" />
            
            <div className="flex flex-col gap-2 pl-4 border-l-2 border-amber-200 ml-6 py-2">
              <div className="flex items-center gap-2 text-xs text-zinc-600"><div className="w-2 h-2 rounded-full bg-amber-400"></div>Tool Planning</div>
              <div className="flex items-center gap-2 text-xs text-zinc-600"><div className="w-2 h-2 rounded-full bg-amber-400"></div>Execute Tool</div>
              <div className="flex items-center gap-2 text-xs text-zinc-600"><div className="w-2 h-2 rounded-full bg-amber-400"></div>Parse Output</div>
            </div>
          </div>

        </div>

        {/* Layer 4: Aggregation & Output */}
        <div className="relative h-12 w-full">
           {/* Arrows converging */}
           <div className="absolute top-0 left-[20%] right-[20%] h-6 border-x-2 border-b-2 border-zinc-300 rounded-b-2xl"></div>
           <div className="absolute bottom-0 left-1/2 -translate-x-1/2"><ArrowRight className="text-zinc-400 rotate-90" size={20} /></div>
        </div>

        <div className="flex justify-center">
          <Node icon={FileText} title="Final Response" subtitle="Structured Markdown Output" color="indigo" />
        </div>

        {/* Database Layer (Bottom) */}
        <div className="mt-8 pt-8 border-t-2 border-dashed border-zinc-200 flex justify-center gap-8">
          <div className="flex items-center gap-3 px-6 py-3 bg-zinc-100 rounded-lg border border-zinc-300 text-zinc-600">
            <Database size={20} />
            <span className="font-mono text-sm font-bold">SQLite (kg_nodes, vectors, history)</span>
          </div>
        </div>

      </div>
    </div>
  );
}
