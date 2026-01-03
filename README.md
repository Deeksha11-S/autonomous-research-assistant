# 🤖 Autonomous AI Research Assistant

A **fully agentic multi-agent system** that autonomously discovers emerging scientific domains, formulates novel research questions, gathers and processes data from disparate sources, designs experiments, critiques its own results through multiple iterations, and generates structured research papers with interactive visualizations—all with **zero human intervention after startup**.

![Architecture Diagram](https://img.shields.io/badge/Architecture-Multi--Agent-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

## 🎯 Project Overview

This system implements a **complete autonomous research pipeline** where specialized AI agents collaborate to conduct scientific research from scratch. Unlike traditional RAG systems or single-agent approaches, this system demonstrates **true agency** with planning, tool use, self-criticism, memory management, and emergent multi-agent collaboration.

### **Key Capabilities:**
- **Autonomous Domain Discovery**: Identifies emerging scientific fields post-2024
- **Novel Question Generation**: Creates non-trivial research questions requiring synthesis
- **Multi-Source Data Acquisition**: Gathers data from PDFs, APIs, GitHub, arXiv, and web sources
- **Self-Critique Mechanism**: Ruthlessly critiques methodology and forces iterations
- **Confidence Quantification**: Calculates and reports uncertainty scores
- **Research Paper Generation**: Produces complete Markdown papers with visualizations
- **Live Dashboard**: Real-time monitoring of agent activities and progress

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                      │
│  Real-time Monitoring + Interactive Visualizations + UI     │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                       │
│  Workflow Management + Memory + Conflict Resolution         │
└─────────────────────────────────────────────────────────────┘
                               │
    ┌──────────┬──────────┬──────────┬──────────┬──────────┐
    │          │          │          │          │          │
┌───▼──┐  ┌───▼──┐  ┌───▼──┐  ┌───▼──┐  ┌───▼──┐  ┌───▼──┐
│Domain │  │Question│  │ Data  │  │Expert│  │Critic│  │Uncert│
│Scout  │  │Generator│  │Alchemist│  │Designer│  │Agent │  │Agent  │
└───────┘  └────────┘  └────────┘  └───────┘  └──────┘  └───────┘
    │          │          │          │          │          │
┌─────────────────────────────────────────────────────────────┐
│                    Tool Ecosystem                           │
│  Web Scrapers │ PDF Processors │ Search APIs │ Viz Tools   │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                    Memory Systems                           │
│  Vector DB (Chroma) + Summary Memory + Knowledge Graph     │
└─────────────────────────────────────────────────────────────┘
```

### **Agent Specializations:**

| Agent | Role | Key Responsibilities |
|-------|------|----------------------|
| **Domain Scout** | Field Discovery | Identifies emerging scientific domains post-2024 using real-time search |
| **Question Generator** | Research Design | Creates 3-5 novel, non-trivial research questions requiring synthesis |
| **Data Alchemist** | Data Acquisition | Gathers data from ≥3 disparate sources (PDFs, APIs, CSVs, images) |
| **Experiment Designer** | Methodology Design | Proposes hypotheses and designs experiments based on data insights |
| **Critic Agent** | Quality Assurance | Attacks methodology, statistics, assumptions; forces iterations if needed |
| **Uncertainty Quantifier** | Confidence Scoring | Calculates confidence scores; abstains if <60% confident |
| **Orchestrator** | Workflow Management | Manages memory, resolves conflicts, enforces iteration limits |

## 🚀 Quick Start Guide

### **Prerequisites**
- **Python 3.9+**
- **Groq API Key** (free tier available at [console.groq.com](https://console.groq.com))
- **Tavily API Key** (free tier available at [tavily.com](https://tavily.com)) - Optional but recommended
- **4GB+ RAM**
- **Git** installed

### **Step 1: Clone and Setup**

```bash
# Clone the repository
git clone <repository-url>
cd ai_research_assistant

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### **Step 2: Install Dependencies**

```bash
# Install Python packages
pip install -r requirements.txt

# Install Playwright browsers for web scraping
playwright install chromium
```

### **Step 3: Configure API Keys**

```bash
# Copy the environment template
cp .env.example .env

# Edit .env file with your API keys
# Required: GROQ_API_KEY
# Optional but recommended: TAVILY_API_KEY for web search
```

**Example `.env` file:**
```env
# Required: Get from https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# Optional but recommended: Get from https://tavily.com
TAVILY_API_KEY=your_tavily_api_key_here

# Alternative search API (optional)
SERPAPI_API_KEY=your_serpapi_api_key_here
```

### **Step 4: Run the Application**

#### **Option A: Using the Quick Start Script (Recommended)**
```bash
python run.py
```
This script will:
1. Check dependencies
2. Validate API keys
3. Setup necessary directories
4. Launch Streamlit and open your browser automatically

#### **Option B: Manual Launch**
```bash
# Start Streamlit application
streamlit run app.py

# Application will be available at:
# http://localhost:8501
```

#### **Option C: Docker Deployment**
```bash
# Build Docker image
docker build -t ai-research-assistant .

# Run with Docker Compose
docker-compose up -d

# Access at: http://localhost:8501
```

## 🎮 Using the Application

### **Dashboard Interface**

1. **Start Research**: Click the "🚀 Start Research" button to begin autonomous research
2. **Real-time Monitoring**: Watch agent activities and messages in real-time
3. **Progress Visualization**: Track research progress through interactive charts
4. **Results Panel**: View confidence scores, selected domains, and download research papers
5. **Configuration Sidebar**: Adjust parameters like max iterations and confidence thresholds

### **Research Workflow**

When you click "Start Research", the system autonomously executes:

```
1. Domain Discovery (10-30 seconds)
   → Searches for emerging scientific fields post-2024
   → Evaluates novelty and feasibility

2. Question Generation (5-15 seconds)
   → Creates 3-5 novel research questions
   → Rates questions for novelty/feasibility

3. Data Acquisition (30-60 seconds)
   → Gathers data from ≥3 disparate sources
   → Processes messy data (OCR, table extraction, schema alignment)

4. Experiment Design (10-20 seconds)
   → Formulates testable hypotheses
   → Designs experiments with control measures

5. Self-Critique (5-15 seconds)
   → Critiques methodology and statistics
   → Forces iteration if p-value >0.05 or effect size trivial

6. Confidence Quantification (5-10 seconds)
   → Calculates confidence scores
   → Determines if should abstain (<60% confidence)

7. Research Paper Generation (10-20 seconds)
   → Compiles findings into structured Markdown paper
   → Generates interactive visualizations

Total Time: 1.5-3 minutes per iteration (max 5 iterations)
```

### **Expected Output**

After the autonomous process completes, you'll receive:

1. **Complete Research Paper** (Markdown format) including:
   - Abstract and Introduction
   - Methodology and Data Sources
   - Results and Analysis
   - Limitations and Future Work (written by Critic Agent)
   - References and Citations

2. **Interactive Visualizations**:
   - Agent Confidence Scores (bar chart)
   - Research Process Timeline
   - Data Sources Distribution (pie chart)
   - Uncertainty Analysis Heatmap

3. **Confidence Report**:
   - Component-wise confidence scores
   - Overall confidence percentage
   - Uncertainty sources and recommendations

## 🔧 Configuration Options

### **Environment Variables**

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GROQ_API_KEY` | Groq Cloud API key for LLM access | - | **Yes** |
| `TAVILY_API_KEY` | Tavily API key for web search | - | No |
| `SERPAPI_API_KEY` | SerpAPI key (alternative search) | - | No |
| `LLM_MODEL` | LLM model to use | `llama-3.1-70b-versatile` | No |
| `MAX_ITERATIONS` | Maximum research iterations | `5` | No |
| `MIN_CONFIDENCE` | Minimum confidence threshold | `0.6` | No |

### **Streamlit Sidebar Settings**

Adjust these in the web interface:

- **Max Iterations**: 1-5 (controls refinement cycles)
- **Min Confidence**: 50-90% (threshold for accepting results)
- **Domain Filters**: Include/exclude specific scientific domains
- **Data Source Preferences**: Prioritize certain data types

## 🧪 Testing and Development

### **Run Individual Agents**

```python
from agents.domain_scout import DomainScoutAgent

agent = DomainScoutAgent()
result = await agent.execute({})
print(f"Discovered domains: {result['domains']}")
```

### **Test Complete Workflow**

```python
from workflows.research_workflow import ResearchWorkflow

workflow = ResearchWorkflow()
results = await workflow.run(max_iterations=3)
print(f"Research completed: {results['status']}")
```

### **Run Unit Tests**

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_agents.py

# Run with coverage report
python -m pytest --cov=agents tests/
```

## 📁 Project Structure

```
ai_research_assistant/
├── agents/                    # Specialized AI agents
│   ├── base_agent.py         # Base agent class
│   ├── domain_scout.py       # Domain discovery agent
│   ├── question_generator.py # Research question generator
│   ├── data_alchemist.py     # Data gathering agent
│   ├── experiment_designer.py # Experiment design agent
│   ├── critic.py            # Self-critique agent
│   ├── uncertainty_agent.py  # Confidence quantification
│   └── orchestrator.py      # Workflow coordinator
├── tools/                    # Tool implementations
│   ├── web_scraper.py       # Web scraping utilities
│   ├── data_processor.py    # Data cleaning and processing
│   ├── search_tools.py      # Search API integrations
│   ├── visualization_tools.py # Plotly visualization helpers
│   └── pdf_tools.py         # PDF processing utilities
├── memory/                   # Memory systems
│   ├── vector_memory.py     # ChromaDB vector storage
│   └── summary_memory.py    # Summary-based memory
├── workflows/               # Workflow orchestration
│   └── research_workflow.py # LangGraph workflow definition
├── utils/                   # Utilities and helpers
│   ├── config.py           # Configuration management
│   ├── logger.py           # Logging setup
│   └── helpers.py          # Helper functions
├── data/                   # Data storage
│   ├── temp/              # Temporary files
│   └── chroma_db/         # Vector database storage
├── app.py                  # Streamlit main application
├── run.py                  # Quick start script
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
├── .env.example           # Environment template
└── README.md             # This file
```

## 🛠️ Technical Stack

### **Core Framework**
- **LangGraph**: Agent orchestration and workflow management
- **Groq LLM**: Llama-3.1-70B for agent reasoning (free tier available)
- **Streamlit**: Web interface and real-time dashboard

### **Data Processing**
- **ChromaDB**: Vector database for memory and similarity search
- **BeautifulSoup + Playwright**: Web scraping and data extraction
- **PyMuPDF + pdfplumber**: PDF processing and text extraction
- **Tavily API**: Real-time web search (free tier available)

### **Visualization**
- **Plotly**: Interactive charts and dashboards
- **Matplotlib**: Static visualizations

### **Deployment**
- **Docker**: Containerization for consistent environments
- **Docker Compose**: Multi-service orchestration
- **Vercel/Netlify**: Frontend deployment (optional)
- **Railway/Render**: Backend deployment (optional)

## 🔍 How It Works: Technical Deep Dive

### **1. Agent Communication Protocol**
Agents communicate through a structured message passing system with:
- **Role-based system prompts** defining agent responsibilities
- **Tool calling** for external actions (web search, data processing)
- **Confidence scoring** on all outputs
- **Abstention mechanism** when confidence <60%

### **2. Memory Systems**
- **Vector Memory**: Stores research findings in ChromaDB with semantic search
- **Summary Memory**: Maintains context across conversations
- **Knowledge Graph**: Tracks relationships between concepts, data sources, and findings

### **3. Self-Critique Loop**
The Critic Agent evaluates research using:
- **Statistical validity checks** (p-values, effect sizes, sample sizes)
- **Methodological soundness assessment** (controls, randomization, blinding)
- **Data quality evaluation** (source diversity, recency, completeness)
- **Logical consistency verification** (argument coherence, evidence support)

### **4. Uncertainty Quantification**
Confidence scores are calculated based on:
- **Source credibility** (peer-reviewed vs. blog posts)
- **Data recency** (2024 publications vs. older sources)
- **Methodological rigor** (controlled experiments vs. observational studies)
- **Result consistency** (multiple sources supporting same conclusion)
- **Statistical power** (sample sizes, effect sizes)

## 🐛 Troubleshooting

### **Common Issues and Solutions**

| Issue | Solution |
|-------|----------|
| **"API key not found" error** | Ensure `.env` file exists in project root with correct API keys |
| **Playwright browser errors** | Run `playwright install chromium` and ensure Chrome is installed |
| **Memory/performance issues** | Reduce `MAX_ITERATIONS` in config or use smaller LLM model |
| **Slow web scraping** | Check internet connection; consider using mock data for testing |
| **Dependency conflicts** | Use fresh virtual environment: `rm -rf venv && python -m venv venv` |
| **Streamlit port already in use** | Change port: `streamlit run app.py --server.port=8502` |

### **Debug Mode**

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set in `.env`:
```env
LOG_LEVEL=DEBUG
```

## 📈 Performance Optimization

### **For Faster Execution**
1. Use `llama-3.1-8b` instead of `llama-3.1-70b` in config
2. Reduce `MAX_ITERATIONS` to 2-3
3. Limit data sources to 2-3 per agent
4. Disable PDF processing if not needed

### **For Better Results**
1. Provide Tavily API key for better web search
2. Increase `MAX_ITERATIONS` to 4-5 for more refinement
3. Adjust `MIN_CONFIDENCE` to 70% for stricter quality control
4. Enable all domain filters for broader discovery

## 🔮 Future Enhancements

### **Planned Features**
- [ ] **Human-in-the-loop mode**: Allow human feedback during research
- [ ] **Multi-modal agents**: Process images, audio, and video data
- [ ] **Cross-domain synthesis**: Combine insights from multiple fields
- [ ] **Real-time collaboration**: Multiple instances working on related problems
- [ ] **Advanced visualization**: 3D plots, network graphs, interactive dashboards
- [ ] **API endpoints**: REST API for programmatic access
- [ ] **Plugin system**: Community-contributed agents and tools

### **Research Directions**
- **Emergent collaboration**: Studying how agents develop communication patterns
- **Meta-learning**: Agents improving their own research methodologies
- **Distributed research**: Coordinating multiple autonomous systems
- **Ethical oversight**: Built-in ethical reasoning and bias detection

## 🤝 Contributing

We welcome contributions! Here's how to help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### **Areas Needing Contributions**
- Additional data source integrations
- New agent specializations
- Improved visualization components
- Documentation and tutorials
- Performance optimizations
- Test coverage improvements

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Check code quality
flake8 .
black --check .
mypy .

# Run pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## 📚 Learning Resources

### **Understanding the Architecture**
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Agent Systems Design Patterns](https://www.patterns.app/blog/ai-agents-design-patterns)
- [Multi-Agent Systems Research](https://arxiv.org/search/?query=multi-agent+systems)

### **Related Projects**
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT): Early autonomous agent system
- [CrewAI](https://github.com/joaomdmoura/crewAI): Framework for orchestrating role-playing agents
- [LangChain](https://github.com/langchain-ai/langchain): Framework for LLM applications

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Groq** for providing free access to high-performance LLMs
- **LangChain/LangGraph team** for the incredible orchestration framework
- **Streamlit** for making interactive apps incredibly simple
- **All open-source contributors** whose work made this project possible

## 📞 Support and Community

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and community support
- **Email**: [Your contact email]
- **Twitter/X**: [Your Twitter handle]

## 🎓 Educational Use

This project is excellent for learning about:
- Multi-agent systems design
- Autonomous AI systems
- Scientific research methodology
- LLM application development
- Real-time data visualization
- Web scraping and data processing

---

**Disclaimer**: This is a research prototype. The generated research papers should be validated by human experts before use in actual scientific work. The system is designed to demonstrate autonomous research capabilities, not to replace human researchers.

---

<div align="center">
  
**Built with ❤️ by [Your Name/Organization]**

*If this project helps your research, please consider giving it a ⭐ on GitHub!*

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/ai-research-assistant&type=Date)](https://star-history.com/#yourusername/ai-research-assistant&Date)

</div>