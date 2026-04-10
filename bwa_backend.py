from __future__ import annotations

import json
import logging
import operator
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Output directory for generated blogs and images
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (ported from your image flow) ----
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


# -----------------------------
# 2) LLM
# -----------------------------
# --- NVIDIA API LLM initialization ---
llm = ChatOpenAI(
    model="mistralai/mistral-large-3-675b-instruct-2512",
    api_key=os.environ.get("NVIDIA_API_KEY", ""),
    base_url="https://integrate.api.nvidia.com/v1",
    max_tokens=4096,
    temperature=0.20,
    top_p=0.70,
)


# =============================================================
# Robust JSON Parsing Helpers
# =============================================================
def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models may emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _repair_truncated_json(text: str) -> Optional[dict]:
    """
    Attempt to repair JSON that was truncated mid-output (e.g. due to max_tokens).
    Strategy: find the last complete object/item boundary and close all open brackets.
    """
    # Find the last complete value boundary (after a comma or closing brace/bracket)
    # Try progressively shorter substrings ending at sensible boundaries
    for end_char in ['},', '},\n', '}\n', '}']:
        last_pos = text.rfind(end_char)
        if last_pos != -1:
            candidate = text[:last_pos + 1]  # include the '}'
            # Count open vs close braces and brackets
            open_braces = candidate.count('{') - candidate.count('}')
            open_brackets = candidate.count('[') - candidate.count(']')
            # Close them
            candidate += ']' * open_brackets + '}' * open_braces
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


def _extract_json(text: str) -> dict:
    """
    Extract JSON from LLM output. Handles:
    - Raw JSON
    - JSON in markdown code fences
    - Mixed text with embedded JSON
    - Truncated JSON (best-effort repair)
    """
    text = _strip_think_tags(text)

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` blocks
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            # Code fence found but content isn't valid JSON — may be truncated
            repaired = _repair_truncated_json(m.group(1).strip())
            if repaired is not None:
                return repaired

    # Try finding outermost braces
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    # Last resort: try repairing truncated JSON from the first brace onward
    if brace_start != -1:
        repaired = _repair_truncated_json(text[brace_start:])
        if repaired is not None:
            return repaired

    raise ValueError(f"Could not extract valid JSON from LLM output:\n{text[:500]}")


def _llm_structured_call(messages: list, model_class: type[BaseModel], max_retries: int = 3):
    """
    Invoke the LLM, extract JSON from response, and parse into a Pydantic model.
    This replaces all with_structured_output calls for HuggingFace compatibility.
    Retries up to max_retries times on parse failure.
    """
    # Add JSON schema hint to the last message
    schema_hint = f"\n\nYou MUST respond with ONLY a valid JSON object matching this schema:\n{json.dumps(model_class.model_json_schema(), indent=2)}"

    # Append schema hint to last human message
    enhanced_messages = []
    for msg in messages:
        enhanced_messages.append(msg)

    if enhanced_messages and isinstance(enhanced_messages[-1], HumanMessage):
        enhanced_messages[-1] = HumanMessage(
            content=enhanced_messages[-1].content + schema_hint
        )
    else:
        enhanced_messages.append(HumanMessage(content=schema_hint))

    last_error = None
    for attempt in range(max_retries):
        try:
            response = llm.invoke(enhanced_messages)
            raw_text = response.content.strip()
            # Strip <think> tags from reasoning models before JSON extraction
            raw_text = _strip_think_tags(raw_text)
            parsed = _extract_json(raw_text)
            return model_class.model_validate(parsed)
        except (ValueError, json.JSONDecodeError, Exception) as e:
            last_error = e
            logging.warning(
                f"LLM structured call attempt {attempt + 1}/{max_retries} failed "
                f"for {model_class.__name__}: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(1)  # Brief pause before retry

    raise ValueError(
        f"Failed to get valid {model_class.__name__} from LLM "
        f"after {max_retries} attempts. Last error: {last_error}"
    )


# =============================================================
# Helper to safely get plan attributes (works with dict or Pydantic)
# =============================================================
def _get_plan_attr(plan, attr, default=None):
    """Get attribute from a Plan, whether it's a Pydantic object or a dict."""
    if hasattr(plan, attr):
        return getattr(plan, attr)
    if isinstance(plan, dict):
        return plan.get(attr, default)
    return default


def _plan_to_dict(plan) -> dict:
    """Convert plan to dict whether it's Pydantic or already a dict."""
    if hasattr(plan, "model_dump"):
        return plan.model_dump()
    if isinstance(plan, dict):
        return plan
    return json.loads(json.dumps(plan, default=str))


def _evidence_to_dicts(evidence) -> list:
    """Convert evidence list to list of dicts."""
    out = []
    for e in (evidence or []):
        if hasattr(e, "model_dump"):
            out.append(e.model_dump())
        elif isinstance(e, dict):
            out.append(e)
        else:
            out.append({"title": str(e), "url": ""})
    return out


# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.

Return ONLY a valid JSON object. Do not include any other text.
"""

def router_node(state: State) -> dict:
    decision = _llm_structured_call(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ],
        RouterDecision,
    )

    mode = decision.mode if hasattr(decision, "mode") else decision.get("mode", "closed_book")
    needs_research = decision.needs_research if hasattr(decision, "needs_research") else decision.get("needs_research", False)
    queries = decision.queries if hasattr(decision, "queries") else decision.get("queries", [])

    if mode == "open_book":
        recency_days = 7
    elif mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": needs_research,
        "mode": mode,
        "queries": queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    try:
        from tavily import TavilyClient
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            print("DEBUG: TAVILY_API_KEY not found in environment.")
            return []
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results
        )
        out: List[dict] = []
        for r in response.get("results", []):
            # Truncate snippet to save tokens since the model has to rewrite it in JSON
            snippet = r.get("content") or r.get("snippet") or ""
            if len(snippet) > 150:
                snippet = snippet[:147] + "..."
            
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": snippet,
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception as e:
        print(f"DEBUG tavily search error: {e}")
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.

Return ONLY a valid JSON object. Do not include any other text.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))

    if not raw:
        return {"evidence": []}

    pack = _llm_structured_call(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw}"
                )
            ),
        ],
        EvidencePack,
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don't invent events).

Output must match Plan schema. Return ONLY a valid JSON object. Do not include any other text.
"""

def orchestrator_node(state: State) -> dict:
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    evidence_dicts = _evidence_to_dicts(evidence)

    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = _llm_structured_call(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{evidence_dicts[:16]}"
                )
            ),
        ],
        Plan,
    )

    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    plan = state["plan"]
    assert plan is not None

    plan_dict = _plan_to_dict(plan)
    evidence_dicts = _evidence_to_dicts(state.get("evidence", []))

    tasks = _get_plan_attr(plan, "tasks", [])
    # If plan is a dict, tasks will be a list of dicts; if Pydantic, a list of Task objects
    task_list = []
    for t in tasks:
        if hasattr(t, "model_dump"):
            task_list.append(t.model_dump())
        elif isinstance(t, dict):
            task_list.append(t)

    # Guard: if the plan has no tasks, create a fallback task
    if not task_list:
        logging.warning("Plan has 0 tasks — creating a fallback task.")
        task_list = [
            {
                "id": 1,
                "title": "Overview",
                "goal": f"Provide a comprehensive overview of {state['topic']}.",
                "bullets": [
                    "Introduce the topic and its significance.",
                    "Cover the key concepts and ideas.",
                    "Summarize the current state and future outlook.",
                ],
                "target_words": 400,
                "tags": [],
                "requires_research": False,
                "requires_citations": False,
                "requires_code": False,
            }
        ]

    return [
        Send(
            "worker",
            {
                "task": td,
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": plan_dict,
                "evidence": evidence_dicts,
            },
        )
        for td in task_list
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    # Strip any <think> tags from reasoning models
    section_md = _strip_think_tags(section_md)

    # Remove leading markdown code fences if the model wrapped the entire response
    if section_md.startswith("```markdown"):
        section_md = re.sub(r"^```markdown\s*\n?", "", section_md)
        section_md = re.sub(r"\n?```\s*$", "", section_md)
    elif section_md.startswith("```") and not section_md.startswith("```\n#"):
        section_md = re.sub(r"^```\w*\s*\n?", "", section_md)
        section_md = re.sub(r"\n?```\s*$", "", section_md)

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")

    sections = state.get("sections") or []
    if not sections:
        logging.warning("merge_content called with empty sections list.")
        body = "*No content was generated. Please try again with a different topic.*"
    else:
        ordered_sections = [md for _, md in sorted(sections, key=lambda x: x[0])]
        body = "\n\n".join(ordered_sections).strip()

    blog_title = _get_plan_attr(plan, "blog_title", "Blog")
    merged_md = f"# {blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Use placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- Avoid decorative images; prefer technical diagrams with short labels.

Return ONLY a valid JSON object matching the schema. DO NOT output the full markdown blog.
"""

def decide_images(state: State) -> dict:
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None
    blog_kind = _get_plan_attr(plan, "blog_kind", "explainer")

    image_plan = _llm_structured_call(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Propose image prompts based on this blog content:\n\n"
                    # Pass a truncated version to save input tokens since we only need the gist
                    f"{merged_md[:2500]}"
                )
            ),
        ],
        GlobalImagePlan,
    )

    specs = [img.model_dump() for img in image_plan.images]
    md_with_placeholders = merged_md

    if specs:
        lines = md_with_placeholders.split("\n")
        header_indices = [i for i, line in enumerate(lines) if line.startswith("## ")]
        
        # Insert placeholders before headers
        for i, spec in enumerate(specs):
            if header_indices:
                # Distribute evenly
                idx_in_headers = i % len(header_indices)
                insert_line = header_indices[idx_in_headers]
                lines.insert(insert_line, f"\n{spec['placeholder']}\n")
                # shift subsequent indices
                header_indices = [x + 1 for x in header_indices]
            else:
                lines.append(f"\n{spec['placeholder']}\n")
                
        md_with_placeholders = "\n".join(lines)

    return {
        "md_with_placeholders": md_with_placeholders,
        "image_specs": specs,
    }


# =============================================================
# Image Generation with NVIDIA Stable Diffusion 3 Medium
# =============================================================
NVIDIA_IMG_API_KEY = os.environ.get("NVIDIA_IMG_API_KEY", "")
NVIDIA_IMG_URL = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium"


def _generate_image_bytes(prompt: str) -> bytes:
    """
    Generate image bytes using NVIDIA Stable Diffusion 3 Medium API.
    Returns raw image bytes (PNG).
    """
    import requests
    import base64

    headers = {
        "Authorization": f"Bearer {NVIDIA_IMG_API_KEY}",
        "Accept": "application/json",
    }

    payload = {
        "prompt": prompt,
        "cfg_scale": 5,
        "aspect_ratio": "16:9",
        "seed": 0,
        "steps": 50,
        "negative_prompt": "blurry, low quality, distorted, text, watermark",
    }

    response = requests.post(NVIDIA_IMG_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    body = response.json()

    # NVIDIA API returns base64-encoded image in the response
    image_b64 = body.get("image") or body.get("artifacts", [{}])[0].get("base64", "")
    if not image_b64:
        raise RuntimeError(f"No image data in NVIDIA API response: {list(body.keys())}")

    return base64.b64decode(image_b64)


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []
    blog_title = _get_plan_attr(plan, "blog_title", "blog")

    # If no images requested, just write merged markdown
    if not image_specs:
        filename = f"{_safe_slug(blog_title)}.md"
        (OUTPUT_DIR / filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = OUTPUT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        # generate only if needed
        if not out_path.exists():
            try:
                img_bytes = _generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](output/images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{_safe_slug(blog_title)}.md"
    (OUTPUT_DIR / filename).write_text(md, encoding="utf-8")
    return {"final": md}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
