import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import ArchitectureDiagram from './ArchitectureDiagram';

export default function ReadmePage() {
  const [content, setContent] = useState('');

  useEffect(() => {
    // In a real production build, we might need to fetch this from an API endpoint
    // or import it as a raw string if configured in Vite.
    // For simplicity here, we'll fetch it from the public directory if we move it there,
    // or just hardcode the content for now since we can't easily move files to public in this env without a build step.
    // Actually, let's create an API endpoint to serve the README content.
    fetch('/api/readme')
      .then(res => res.text())
      .then(text => setContent(text))
      .catch(err => console.error('Failed to load README:', err));
  }, []);

  return (
    <div className="max-w-5xl mx-auto p-6 bg-white shadow-sm rounded-xl border border-zinc-200 my-8">
      
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-zinc-800 mb-6 px-2 border-l-4 border-emerald-500">系统架构概览</h2>
        <ArchitectureDiagram />
      </div>

      <div className="prose prose-zinc max-w-none">
        <ReactMarkdown 
          remarkPlugins={[remarkMath]} 
          rehypePlugins={[rehypeKatex]}
        >
          {content}
        </ReactMarkdown>
      </div>
    </div>
  );
}
