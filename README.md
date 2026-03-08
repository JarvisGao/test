# FinQ&A 智能金融问答系统

FinQ&A 是一个基于检索增强生成（RAG）架构的智能金融问答助手。系统结合了本地知识图谱（Knowledge Graph）、向量检索（HNSW）、全文检索（BM25）以及 Google 搜索回退机制，为用户提供专业、准确的金融知识与资产分析服务。

---

## 1. 系统架构图

系统采用模块化的 Agent 架构，核心流程如下：

```mermaid
graph TD
    User([用户输入]) --> Router[Router Agent]
    
    Router -- 闲聊/通用 --> ChatAgent[Chat Agent]
    Router -- 股票数据/价格 --> StockAgent[Stock Agent]
    Router -- 金融知识/概念 --> RAGAgent[RAG Agent]
    
    subgraph RAG Agent (Task 2)
        RAGAgent --> KGSearch[1. 知识图谱检索]
        KGSearch -- 无结果 --> HybridSearch[2. 混合检索]
        HybridSearch -- BM25 --> FTS5[(SQLite FTS5)]
        HybridSearch -- Vector --> HNSW[(HNSW 内存索引)]
        HybridSearch -- RRF融合 --> TopK[Top K 结果]
        
        HybridSearch -- 无结果 --> WebSearch[3. Google 搜索回退]
        WebSearch --> GoogleAPI[Google Search API]
        
        KGSearch -- 有结果 --> PromptBuilder[Prompt 构建]
        TopK --> PromptBuilder
        GoogleAPI --> PromptBuilder
        
        PromptBuilder --> LLM[Gemini 3.1 Flash Lite]
    end
    
    LLM --> Output([结构化专业回答])
    ChatAgent --> Output
    StockAgent --> Output
```

---

## 2. 技术选型说明

本系统在技术选型上兼顾了**轻量化**与**高性能**：

*   **大语言模型 (LLM)**: `Gemini 3.1 Flash Lite`
    *   *理由*: 极高的性价比和响应速度，适合高频调用和实时问答场景，同时保持了足够的推理能力。
*   **向量数据库与索引**: `SQLite` + `hnsw` (纯 JS 实现)
    *   *理由*: 避免引入庞大的外部向量数据库（如 Milvus/Pinecone），保持系统的轻量和易部署性。HNSW 算法在内存中提供极快的近似最近邻（ANN）搜索。
*   **全文检索**: `SQLite FTS5`
    *   *理由*: SQLite 原生支持的高性能全文检索虚拟表，用于实现 BM25 算法，弥补纯向量检索在“专有名词”匹配上的不足。
*   **Embedding 模型**: `Xenova/all-MiniLM-L6-v2` (Transformers.js)
    *   *理由*: 本地运行的轻量级模型，无需调用外部 API 即可生成高质量的文本向量，保护数据隐私并降低延迟。
*   **知识图谱存储**: `SQLite` (关系表 `kg_nodes`, `kg_edges`)
    *   *理由*: 利用关系型数据库模拟简单的图结构，足以应对中小规模的金融实体关系查询。
*   **前端框架**: `React` + `Vite` + `Tailwind CSS`
    *   *理由*: 现代化的前端技术栈，提供流畅的打字机效果和 Markdown 渲染体验。

---

## 3. Prompt 设计思路

在 RAG Agent 的 Prompt 设计中，我们重点突出了**专业性**和**严谨性**：

1.  **角色设定 (Persona)**：将 AI 设定为“顶级的华尔街金融分析师和财富管理专家”，从语气上奠定专业基础。
2.  **结构化输出约束**：强制要求使用 Markdown 格式（加粗、列表、分段），使长篇的金融分析报告更易于阅读。
3.  **数据引用要求**：明确要求 AI 必须引用上下文中的具体数据，防止大模型产生“幻觉”（Hallucination）。
4.  **诚实性声明**：当本地知识库和 Google 搜索均无结果时，允许 AI 使用通用知识回答，但**强制要求**在开头声明：“*注：以下信息基于通用金融知识，非本地知识库检索结果。*” 这在金融领域尤为重要，确保用户了解信息的可靠度。
5.  **来源透明度**：在 Prompt 中注入了 `source` 变量（如“知识图谱”、“本地向量数据库”或“Google Web Search”），并在回答末尾展示，增强结果的可信度。

---

## 4. 数据来源说明

系统的数据来源分为三个层次：

1.  **初始内置数据 (Seed Data)**：系统启动时，会自动向 SQLite 数据库中注入一些基础的金融概念（如市盈率、ROE、股息率等）作为冷启动数据。
2.  **自动化爬虫 (Web Scraper)**：
    *   提供了 `scrape_finance.ts` 脚本，使用 `axios` 和 `cheerio` 从维基百科等公开网站抓取金融条目。
    *   抓取后，自动调用本地 Embedding 模型进行向量化，并存入数据库。
    *   同时触发 `text2structure.ts` 中的 Agent，利用大模型从抓取的文本中提取实体和关系，动态扩充知识图谱。
3.  **实时网络搜索 (Google Fallback)**：当本地数据库（包括爬取的数据）无法覆盖用户的长尾问题或最新资讯时，系统会实时调用 `googlethis` 抓取 Google 搜索的前 5 条摘要。

---

## 5. TODO list

1.  我没有做finetuning, 如果有很多数据的话 finetuning后的本地模型效果会更好, 还需要对安全, 语气类的问题做一个对齐。(可以用知识图谱的数据做一个structure2question任务, 来做finetuning)
2.  知识图谱目前只是简单的爬虫获取了一点点数据，之后可以扩充并采用商业级的数据库来实现
3.  目前的工具没有拓展太多, 之后可以定时爬虫丰富数据库内容, 这个时候可以加入skill来间接式披露工具调用
4.  整体的langgraph没完全用react的模型, 如果是遇到复杂问题, 需要对问题进行拆解, 在执行react的模式的agent会更准确
