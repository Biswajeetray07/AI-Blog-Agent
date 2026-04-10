<p align="center">
  <img src="assets/architecture.png" alt="AI Blog Agent Architecture" width="800"/>
</p>

<h1 align="center">📝 AI Blog Writing Agent</h1>

<p align="center">
  <strong>An autonomous, multi-agent AI pipeline that transforms a topic into a fully-written, researched, and illustrated technical blog post — in a single click.</strong>
</p>

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#-system-architecture">Architecture</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-aws-ec2-deployment">Deployment</a> •
  <a href="#-usage-guide">Usage</a> •
  <a href="#-tech-stack">Tech Stack</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/LangGraph-Multi_Agent-2D72D2?style=for-the-badge" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/Mistral_AI-LLM-FF7000?style=for-the-badge" alt="Mistral"/>
  <img src="https://img.shields.io/badge/Stable_Diffusion-Image_Gen-A100FF?style=for-the-badge" alt="Stable Diffusion"/>
  <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/AWS_EC2-Deployed-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white" alt="AWS EC2"/>
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
</p>

---

## 🎯 Why This Project

Content creation is one of the most time-consuming tasks in developer advocacy and technical marketing. This agent automates the **entire blog-writing pipeline** — from **topic analysis** and **real-time web research** to **structured planning**, **parallel section writing**, and **AI image generation** — reducing hours of manual work to minutes while maintaining professional quality.

**Key differentiator:** Unlike simple "generate a blog" prompts, this system uses a **multi-agent graph architecture** where specialized nodes handle routing, research, planning, writing, and illustration — each with domain-specific prompts, scope guards, and robust error handling.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Intelligent Topic Routing** | Automatically classifies topics into `closed_book`, `hybrid`, or `open_book` mode to determine if live web research is needed |
| 🔍 **Real-Time Web Research** | Integrates Tavily Search API for up-to-date evidence gathering with date-aware recency filtering (7 / 45 / 3650 days) |
| 📋 **Structured Blog Planning** | Generates detailed 5–9 section outlines with goals, bullet points, word targets, and metadata flags per task |
| ✍️ **Parallel Section Writing** | Fan-out architecture writes all sections **concurrently** via LangGraph's `Send` mechanism for maximum speed |
| 🖼️ **AI Image Generation** | Automatically decides where technical diagrams add value and generates them via **NVIDIA Stable Diffusion 3 Medium** |
| 🤖 **Mistral Large 3 (675B)** | Powered by Mistral's largest instruction-tuned model via NVIDIA NIM for high-quality, structured text generation |
| 📊 **Interactive Dashboard** | Streamlit UI with 5 tabs — Plan, Evidence, Markdown Preview, Images, and Logs — for full pipeline visibility |
| 💾 **Blog Library** | Auto-saves generated blogs to disk with reload-from-sidebar functionality for past generations |
| 📦 **Multi-Format Export** | Download as raw Markdown, bundled ZIP (Markdown + images), or images-only ZIP |
| 🔧 **Robust JSON Parsing** | 4-tier JSON extraction cascade with truncation repair for reliable handling of LLM output quirks |
| 🌐 **AWS EC2 Deployed** | Production-ready deployment on AWS EC2 with secure environment configuration and public accessibility |

---

## 🏗️ System Architecture

The agent is built as a **stateful directed acyclic graph (DAG)** using LangGraph, with conditional edges, parallel fan-out, and a nested subgraph for the image generation pipeline:

```
                                    ┌─────────────────────────┐
                                    │     NVIDIA NIM API       │
                                    │  Mistral Large 3 (675B)  │
                                    │       Instruct           │
                                    └───────────┬─────────────┘
                                                │
                                                ▼
┌──────┐    ┌────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌─────────┐    ┌───────┐
│Topic │───▶│ Router │───▶│ Research │───▶│ Orchestrator │───▶│ Workers  │───▶│ Reducer │───▶│ Final │
│Input │    │        │    │ (Tavily) │    │   (Planner)  │    │(Fan-out) │    │         │    │ Blog  │
└──────┘    └────────┘    └──────────┘    └──────────────┘    └──────────┘    └─────────┘    └───────┘
                │              ▲                                                   │
                │   (optional) │                                                   │
                └──────────────┘                                          ┌────────┴────────┐
              closed_book skips                                           │  Reducer Sub-   │
              research entirely                                           │  graph (3 nodes)│
                                                                          │                 │
                                                                          │ 1. Merge Content│
                                                                          │ 2. Decide Images│
                                                                          │ 3. Generate &   │
                                                                          │    Place Images  │
                                                                          └─────────────────┘
```

### Graph Nodes

| Node | Type | Purpose |
|------|------|---------|
| **Router** | Conditional | Analyzes the topic and decides the research mode — `closed_book` (evergreen topics), `hybrid` (needs fresh examples), or `open_book` (news/weekly roundups requiring full live data) |
| **Research** | Linear | Executes 3–10 scoped Tavily web searches, synthesizes results into structured `EvidenceItem` objects with URLs, dates, and snippets. Applies date-based recency filtering |
| **Orchestrator** | Linear | Produces a structured `Plan` with 5–9 `Task` objects — each with title, goal, bullets, word target, and metadata flags (`requires_research`, `requires_citations`, `requires_code`) |
| **Workers** | Fan-out (Parallel) | Each worker receives one `Task` and writes a complete Markdown section. All workers execute **concurrently** via LangGraph's `Send` mechanism |
| **Reducer** | Subgraph (3 nodes) | **Merge** → orders/combines sections; **Decide Images** → LLM determines where diagrams add value; **Generate & Place** → calls NVIDIA SD3 API, saves images, inserts Markdown references |

---

## 🔬 Pipeline Deep Dive

### Stage 1: Intelligent Routing

The Router node receives a topic and the current date, then classifies it into one of three modes:

- **`closed_book`** — Evergreen topics (e.g., *"Self Attention in Transformers"*). No web search needed. Recency window: ~10 years.
- **`hybrid`** — Mostly evergreen but benefits from fresh examples/tools (e.g., *"Best Python ORMs in 2026"*). Recency window: 45 days.
- **`open_book`** — Volatile/news topics (e.g., *"This Week in AI"*). Full web research required. Recency window: 7 days. Automatically forces `blog_kind = "news_roundup"`.

### Stage 2: Web Research (Conditional)

When research is needed, the system:
1. Generates 3–10 high-signal, scoped search queries via LLM
2. Executes each via Tavily's `advanced` search depth (up to 6 results per query)
3. Feeds raw results through an LLM to produce structured, deduplicated `EvidenceItem` objects
4. Applies ISO date-based recency filtering to remove stale evidence

### Stage 3: Orchestration (Planning)

The Orchestrator crafts a detailed blog outline:
- **5–9 structured tasks** with goals, 3–6 bullet points, and word targets (120–550 words each)
- **Metadata flags** per task: `requires_research`, `requires_citations`, `requires_code`
- **Blog-level metadata**: title, audience, tone, blog kind (`explainer`, `tutorial`, `news_roundup`, `comparison`, `system_design`), constraints
- **Grounding rules**: `open_book` plans only reference evidence-backed claims; `hybrid` mixes evergreen content with cited material

### Stage 4: Parallel Section Writing (Fan-out)

Each task is dispatched to an independent Worker node via LangGraph's `Send` API:
- Workers receive the full plan context, evidence pack, and their specific task
- Each worker outputs a complete Markdown section starting with `## Section Title`
- **Scope guards** prevent drift (e.g., news roundups won't produce tutorials)
- **Citation enforcement**: `open_book` sections must cite provided URLs; unsupported claims are flagged as *"Not found in provided sources"*
- All workers run **concurrently**, dramatically reducing end-to-end generation time

### Stage 5: Reduction & Image Generation

The Reducer is itself a **3-node compiled subgraph**:

1. **Merge Content** — Sorts sections by task ID, joins them, and prepends the blog title as `# Title`
2. **Decide Images** — An LLM reviews the merged blog and proposes up to 3 technical diagrams with prompts, alt text, captions, and size/quality specifications. Placeholders (`[[IMAGE_1]]`, `[[IMAGE_2]]`, `[[IMAGE_3]]`) are injected before section headers
3. **Generate & Place Images** — Calls the **NVIDIA Stable Diffusion 3 Medium** API to generate each image, saves PNGs to `output/images/`, and replaces placeholders with Markdown image references. **Failures degrade gracefully** with informative error blocks instead of breaking the document

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | [Mistral Large 3 (675B Instruct)](https://build.nvidia.com/mistralai/mistral-large-3-675b-instruct-2512) via NVIDIA NIM | All text generation — routing decisions, research synthesis, planning, section writing, image planning |
| **Image Generation** | [Stable Diffusion 3 Medium](https://build.nvidia.com/stabilityai/stable-diffusion-3-medium) via NVIDIA NIM | Technical diagram and illustration generation |
| **Agent Framework** | [LangGraph](https://github.com/langchain-ai/langgraph) ≥ 0.2.0 | Stateful DAG orchestration with conditional edges, fan-out/fan-in, and compiled subgraphs |
| **Web Research** | [Tavily](https://tavily.com/) Search API | Real-time web search with advanced search depth and date filtering |
| **LLM Interface** | [LangChain](https://github.com/langchain-ai/langchain) (`langchain-openai`) | OpenAI-compatible chat interface to NVIDIA NIM endpoints |
| **Data Validation** | [Pydantic](https://docs.pydantic.dev/) v2 | Schema enforcement for all structured LLM outputs (`Plan`, `Task`, `RouterDecision`, `ImageSpec`, etc.) |
| **Frontend** | [Streamlit](https://streamlit.io/) ≥ 1.30 | Interactive dashboard with tabs, data tables, real-time progress, image rendering, and download buttons |
| **Deployment** | [AWS EC2](https://aws.amazon.com/ec2/) | Production hosting with Linux server configuration, virtual environments, and secure API key management |
| **Language** | Python 3.10+ | Core runtime |

---

## 📁 Project Structure

```
AI-Blog-Agent/
│
├── bwa_backend.py          # Core LangGraph pipeline (~820 lines)
│   ├── Pydantic Schemas    #   Task, Plan, RouterDecision, EvidenceItem, ImageSpec, GlobalImagePlan
│   ├── LLM initialization  #   NVIDIA NIM API via ChatOpenAI (OpenAI-compatible)
│   ├── JSON Repair Engine   #   4-tier extraction cascade + truncation repair
│   ├── Router Node          #   Topic classification → closed_book / hybrid / open_book
│   ├── Research Node        #   Tavily search → LLM synthesis → deduplication → recency filter
│   ├── Orchestrator Node    #   Structured plan generation (5–9 tasks with metadata)
│   ├── Worker Node          #   Section writer with scope guards + citation enforcement
│   ├── Reducer Subgraph     #   Merge → Decide Images → Generate & Place Images
│   └── Graph Compilation    #   StateGraph → conditional edges → compiled app
│
├── bwa_frontend.py         # Streamlit dashboard (~486 lines)
│   ├── Stream Helpers       #   try_stream() with 3-tier fallback (updates → values → invoke)
│   ├── Markdown Renderer    #   Custom renderer with local image path resolution
│   ├── Blog Management      #   Past blogs listing, loading, title extraction from parsed MD
│   └── UI Layout            #   Sidebar + 5 tabs (Plan, Evidence, Preview, Images, Logs)
│
├── requirements.txt        # Python dependencies (LangGraph, LangChain, Streamlit, Tavily, etc.)
├── .env.example            # API key template (copy to .env)
├── .env                    # Your API keys (gitignored)
├── .gitignore              # Standard Python + project-specific ignores
├── assets/                 # Static assets
│   └── architecture.png    #   System architecture diagram
└── output/                 # Generated content (auto-created at runtime)
    ├── *.md                #   Generated Markdown blog files
    └── images/             #   Generated illustration PNGs from Stable Diffusion
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **NVIDIA NIM API Key** — Free tier available at [build.nvidia.com](https://build.nvidia.com/)
- **NVIDIA Image API Key** — For Stable Diffusion 3 image generation via [build.nvidia.com](https://build.nvidia.com/)
- **Tavily API Key** *(optional but recommended)* — For live web research at [tavily.com](https://tavily.com/)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Biswajeetray07/AI-Blog-Agent.git
cd AI-Blog-Agent

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS / Linux:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your API keys (see table below)

# 5. Launch the application
streamlit run bwa_frontend.py
```

### Environment Variables

Create a `.env` file in the project root (or copy from `.env.example`):

| Variable | Required | Purpose |
|----------|----------|---------|
| `NVIDIA_API_KEY` | ✅ Yes | NVIDIA NIM API key for Mistral Large 3 (675B) text generation |
| `NVIDIA_IMG_API_KEY` | ✅ Yes | NVIDIA NIM API key for Stable Diffusion 3 Medium image generation |
| `TAVILY_API_KEY` | ⚡ Recommended | Enables live web research for `hybrid` and `open_book` topics |

```env
# .env
NVIDIA_API_KEY=nvapi-your-text-generation-key-here
NVIDIA_IMG_API_KEY=nvapi-your-image-generation-key-here
TAVILY_API_KEY=tvly-your-tavily-key-here
```

---

## ☁️ AWS EC2 Deployment

The application is deployed on **AWS EC2** for production hosting. Here's how the deployment is configured:

### Infrastructure Setup

```bash
# 1. Launch an EC2 instance (Amazon Linux 2 / Ubuntu)
#    - Instance type: t2.medium or higher recommended
#    - Security Group: Open ports 8501 (Streamlit) and 22 (SSH)

# 2. SSH into the instance
ssh -i "your-key.pem" ec2-user@<your-ec2-public-ip>

# 3. Install Python and dependencies
sudo yum update -y          # Amazon Linux
sudo yum install python3 python3-pip git -y

# 4. Clone the repository
git clone https://github.com/Biswajeetray07/AI-Blog-Agent.git
cd AI-Blog-Agent

# 5. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 6. Configure environment variables
nano .env
# Add your NVIDIA_API_KEY, NVIDIA_IMG_API_KEY, and TAVILY_API_KEY

# 7. Run the application (background process)
nohup streamlit run bwa_frontend.py --server.port 8501 --server.address 0.0.0.0 &
```

### Security Configuration

| Configuration | Details |
|--------------|---------|
| **Security Group** | Inbound rules for ports `22` (SSH) and `8501` (Streamlit) |
| **API Key Management** | All API keys stored in `.env` file (gitignored), never hardcoded |
| **SSH Access** | Key-pair based authentication via `.pem` file |
| **On-Demand Hosting** | EC2 instance can be started/stopped as needed to minimize costs |

### Accessing the Application

Once deployed, the application is accessible at:
```
http://<your-ec2-public-ip>:8501
```

---

## 📖 Usage Guide

### Generating a Blog

1. **Open the Streamlit app** at `http://localhost:8501` (local) or your EC2 public IP
2. **Enter a topic** in the sidebar text area. Examples:
   - Evergreen: *"Self Attention in Transformers explained with code"*
   - Hybrid: *"Best Python Web Frameworks in 2026"*
   - News: *"This Week in AI — April 2026"*
3. **Set the as-of date** (defaults to today — affects recency filtering for research)
4. **Click 🚀 Generate Blog** and watch the pipeline progress in real-time

### Dashboard Tabs

| Tab | Contents |
|-----|----------|
| 🧩 **Plan** | Blog title, audience, tone, blog kind, and a sortable data table of all tasks with metadata flags |
| 🔎 **Evidence** | Table of research sources with titles, publication dates, publishers, and clickable URLs |
| 📝 **Markdown Preview** | Full rendered blog with embedded AI-generated images + download buttons |
| 🖼️ **Images** | Image plan JSON specifications + visual gallery of all generated illustrations |
| 🧾 **Logs** | Raw event log from graph execution showing node-by-node progress for debugging |

### Export Options

- **⬇️ Download Markdown** — Raw `.md` file of the generated blog
- **📦 Download Bundle** — ZIP archive containing the Markdown file + all generated images
- **⬇️ Download Images** — ZIP of just the generated illustrations

### Loading Past Blogs

Previously generated blogs are auto-saved to `output/` and appear in the sidebar under **Past blogs**. Select any entry and click **📂 Load selected blog** to reload it into the dashboard.

---

## 💡 Design Decisions & Engineering Highlights

### 1. Multi-Agent Graph over Monolithic Prompt
Rather than a single massive prompt, each pipeline stage has a **specialized system prompt** with scope guards. This prevents prompt confusion, enables parallel execution, and makes each node independently testable and improvable.

### 2. Robust JSON Extraction Engine
LLMs frequently produce malformed JSON — wrapped in code fences, preceded by explanations, or truncated by token limits. The `_extract_json()` function implements a **4-tier extraction cascade**:
1. Direct `json.loads()` parsing
2. Regex extraction from `` ```json ``` `` code fences
3. Outermost brace detection and extraction
4. **Truncation repair** — bracket/brace counting with auto-closing for max-token cutoffs

This is wrapped in `_llm_structured_call()` which retries up to 3 times on parse failure, with `<think>` tag stripping for reasoning models.

### 3. Conditional Research with Recency Filtering
The Router doesn't just decide *if* to research — it sets a **recency window** that propagates through the entire graph state. Open-book blogs only see evidence from the last 7 days, preventing stale news from polluting roundups.

### 4. Fan-out / Fan-in via LangGraph Send
Worker nodes execute **concurrently** using LangGraph's `Send` primitive. Sections are collected via an `Annotated[List, operator.add]` reducer, ensuring all outputs are merged regardless of completion order — a true map-reduce pattern.

### 5. Graceful Image Failure Degradation
If NVIDIA's image API fails for any spec, the system inserts a styled **fallback block** with the original prompt, alt text, and error details — so the blog remains fully usable even without images. No crashes, no blank spaces.

### 6. Streaming with Multi-Level Fallbacks
The frontend's `try_stream()` function attempts three strategies in order:
1. LangGraph `stream(mode="updates")` — node-by-node progress updates
2. LangGraph `stream(mode="values")` — full state snapshots
3. Simple `invoke()` — blocking single-shot execution

This ensures compatibility across LangGraph versions and handles edge cases in graph configurations.

### 7. Pydantic Schema-Driven Architecture
Every LLM interaction is governed by a **Pydantic v2 model** (`Plan`, `Task`, `RouterDecision`, `EvidenceItem`, `ImageSpec`, `GlobalImagePlan`). The schema JSON is injected directly into prompts, and responses are validated against these models — catching structural errors before they propagate through the pipeline.

---

## 🗺️ Future Roadmap

- [ ] **Docker containerization** — Package the application for portable deployment
- [ ] **Model selection UI** — Let users choose between LLM providers (NVIDIA, OpenAI, Anthropic)
- [ ] **SEO scoring** — Analyze generated content for keyword density, readability, and structure
- [ ] **Multi-language support** — Generate blogs in languages other than English
- [ ] **Template system** — Pre-defined blog templates (tutorial, comparison, case study)
- [ ] **REST API mode** — FastAPI endpoint for programmatic blog generation
- [ ] **Human-in-the-loop** — Allow editing the plan before generation proceeds
- [ ] **CI/CD pipeline** — Automated testing and deployment workflows

---

## 📄 License

This project is for educational and demonstration purposes.

---

<p align="center">
  Built with ❤️ using <strong>LangGraph</strong> · <strong>Mistral AI</strong> · <strong>Stable Diffusion</strong> · <strong>Tavily</strong> · <strong>Streamlit</strong> · <strong>AWS EC2</strong>
</p>
