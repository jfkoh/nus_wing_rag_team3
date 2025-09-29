# High-Accuracy RAG System Implementation Plan
## NUS WING RAG Project Team 3 - Small LM + RAG

**Project Overview:** Building a highly accurate Retrieval-Augmented Generation system using 10 research papers (6 Astrophysics + 4 Neuroscience/Cognition) with focus on precision and local deployment capabilities.

---

## 1. DETAILED IMPLEMENTATION PLAN

### 1.1 Data Analysis & Preprocessing
**Current Dataset:**
- 10 research papers (PDF format, 5-47 pages each)
- Two distinct domains: Astrophysics (6 papers) and Neuroscience/Cognition (4 papers)
- Total estimated content: ~150-200 pages of scientific text
- ArXiv papers with structured academic format

**Preprocessing Strategy:**
- PDF text extraction with layout preservation (pymupdf/pdfplumber)
- Section-aware chunking (Abstract, Introduction, Methods, Results, Discussion, References)
- Mathematical formula preservation using LaTeX extraction
- Figure and table caption extraction
- Citation link preservation
- Domain-specific preprocessing for scientific terminology

### 1.2 Small Language Model Options

#### **Option A: Local Deployment (Recommended for Privacy)**
1. **Llama 3.2 3B Instruct** (Meta)
   - Size: 3B parameters (~6GB VRAM)
   - Strengths: Strong instruction following, good reasoning
   - Quantization: 4-bit GGUF for 2-3GB memory usage

2. **Phi-3.5-mini-instruct** (Microsoft)
   - Size: 3.8B parameters (~7GB VRAM)
   - Strengths: Excellent reasoning, good with scientific content
   - Optimized for instruction following

3. **Gemma 2 2B IT** (Google)
   - Size: 2B parameters (~4GB VRAM)
   - Strengths: Fast inference, good factual accuracy
   - Efficient for resource-constrained environments

#### **Option B: API-based (Higher Accuracy)**
1. **GPT-4o-mini** (OpenAI)
   - Cost-effective, high accuracy
   - Good scientific reasoning capabilities

2. **Claude 3.5 Haiku** (Anthropic)
   - Excellent for scientific content
   - Strong reasoning and factual accuracy

### 1.3 Vector Database & Embedding Options

#### **Vector Database Options:**
1. **ChromaDB** (Recommended for prototyping)
   - Lightweight, easy setup
   - Good for < 1M documents
   - Built-in persistence

2. **Weaviate**
   - Production-ready
   - Hybrid search capabilities
   - Good performance scaling

3. **Qdrant**
   - High performance
   - Advanced filtering
   - Good for scientific applications

#### **Embedding Model Options:**
1. **all-MiniLM-L6-v2** (Sentence Transformers)
   - Size: 22MB, fast inference
   - Good general-purpose embeddings

2. **e5-large-v2** (Microsoft)
   - Size: 1.34GB, higher accuracy
   - Excellent for scientific content

3. **BGE-large-en-v1.5** (BAAI)
   - State-of-the-art performance
   - Good for academic content

### 1.4 Architecture Components

#### **Core Pipeline:**
```
PDF → Text Extraction → Chunking → Embedding → Vector DB
                                                    ↓
Query → Query Embedding → Similarity Search → Context Retrieval
                                                    ↓
Context + Query → LLM → Response Generation → Post-processing
```

#### **Advanced Features:**
1. **Hybrid Search:** Combine semantic + keyword search
2. **Reranking:** Use cross-encoder for better relevance
3. **Query Expansion:** Expand queries with domain-specific terms
4. **Multi-hop Reasoning:** Chain multiple retrievals for complex queries
5. **Source Attribution:** Track and cite specific paper sections

### 1.5 Chunking Strategy
1. **Hierarchical Chunking:**
   - Document level (paper metadata)
   - Section level (Introduction, Methods, etc.)
   - Paragraph level (semantic units)
   - Sentence level (fine-grained retrieval)

2. **Overlap Strategy:**
   - 20% overlap between chunks
   - Preserve context boundaries
   - Maintain citation integrity

### 1.6 Evaluation Framework

#### **Accuracy Metrics:**
1. **Retrieval Metrics:**
   - Precision@K (K=1,3,5,10)
   - Recall@K
   - Mean Reciprocal Rank (MRR)
   - Normalized Discounted Cumulative Gain (NDCG)

2. **Generation Metrics:**
   - BLEU/ROUGE scores
   - BERTScore for semantic similarity
   - Factual accuracy assessment
   - Citation accuracy

3. **End-to-End Metrics:**
   - Human evaluation (relevance, accuracy, completeness)
   - Domain expert assessment
   - Response time and latency

---

## 2. BRIEF IMPLEMENTATION PLAN

### Core Components:
• **Data:** 10 research papers (Astrophysics + Neuroscience)
• **Model:** Llama 3.2 3B or Phi-3.5-mini (local) or GPT-4o-mini (API)
• **Embeddings:** e5-large-v2 or BGE-large-en-v1.5
• **Vector DB:** ChromaDB (prototype) → Weaviate (production)
• **Framework:** LangChain or LlamaIndex for orchestration

### Implementation Steps:
1. **Setup Environment**
   • Install dependencies (transformers, sentence-transformers, chromadb)
   • Set up local LLM inference (ollama/vllm)
   • Configure vector database

2. **Data Processing**
   • Extract text from PDFs preserving structure
   • Implement hierarchical chunking (section → paragraph)
   • Generate embeddings and store in vector DB

3. **RAG Pipeline**
   • Build query processing (expansion + embedding)
   • Implement hybrid retrieval (semantic + keyword)
   • Add reranking layer for precision
   • Integrate LLM for response generation

4. **Optimization**
   • Fine-tune chunk sizes and overlap
   • Optimize retrieval parameters
   • Implement caching for common queries
   • Add source attribution and citations

5. **Evaluation**
   • Create test question sets for both domains
   • Implement automated metrics pipeline
   • Conduct human evaluation rounds
   • Performance benchmarking

### Testing Strategy:
• **Unit Tests:** Individual component testing
• **Integration Tests:** End-to-end pipeline testing
• **Domain Tests:** Astrophysics vs Neuroscience accuracy
• **Stress Tests:** Large query volumes and edge cases
• **Human Evaluation:** Expert review of responses

### Success Metrics:
• Retrieval Precision@5 > 85%
• Response accuracy > 90% (human eval)
• Average response time < 3 seconds
• Citation accuracy > 95%
• Cross-domain query handling capability

---

## 3. RECOMMENDED IMPLEMENTATION SEQUENCE

### Phase 1: Foundation (Week 1)
- Set up development environment
- Implement PDF processing pipeline
- Basic ChromaDB integration
- Simple retrieval testing

### Phase 2: Core RAG (Week 2)
- Integrate chosen LLM (Llama 3.2 3B)
- Implement basic RAG pipeline
- Add query processing and response generation
- Initial accuracy testing

### Phase 3: Optimization (Week 3)
- Advanced chunking strategies
- Hybrid search implementation
- Reranking integration
- Performance optimization

### Phase 4: Evaluation & Refinement (Week 4)
- Comprehensive evaluation framework
- Human evaluation setup
- Final optimizations
- Documentation and deployment

---

## 4. RISK MITIGATION

### Technical Risks:
• **Model Hallucination:** Implement strong source attribution and fact-checking
• **Retrieval Quality:** Use multiple retrieval strategies and reranking
• **Computational Resources:** Optimize model quantization and caching
• **Domain Specificity:** Create domain-aware preprocessing and evaluation

### Mitigation Strategies:
• Extensive testing with domain experts
• Conservative response generation with uncertainty indicators
• Robust citation and source tracking
• Fallback mechanisms for edge cases

---

**Next Steps:** Begin with Phase 1 implementation focusing on robust data processing and basic retrieval functionality, then iterate based on evaluation results.
