# multi_agent_research_and_coding_companion

An advanced **multi-agent AI research companion** built in Python using the Google Gemini API.  
This project routes user prompts through a hierarchical council of specialized agents for research, coding, security review, optimization, and final synthesis.

---

## Overview

Enterprise AI Authority is a modular LLM orchestration framework designed to simulate a structured decision-making system for AI-assisted work. Instead of relying on a single model response, it breaks a task into multiple expert roles, runs them through separate sub-councils, and then combines the results into one final answer. [file:128]

The system is inspired by enterprise-style AI orchestration and is built to handle both research-heavy prompts and code-heavy prompts efficiently. It supports routing, retry logic, concurrency, model pacing, and final arbitration through a Supreme Council stage. [file:128][web:139][web:145]

---

## Key Features

### Intelligent Prompt Routing
The app first analyzes a user prompt and classifies it into one of three modes:
- `CORE_RESEARCH`
- `CORE_CODE`
- `HYBRID_WORKLOAD` [file:128]

### Research Sub-Council
For research-heavy prompts, the system uses a multi-step board:
- Research Lead,
- Devil’s Advocate,
- Fact-Checker,
- Chair of Research Board.

This creates a layered academic-style reasoning pipeline that challenges assumptions and synthesizes a final brief. [file:128]

### Technical Sub-Council
For coding-heavy prompts, the system uses a technical board:
- Principal Systems Architect,
- Competitive Programmer,
- Cyber Security Engineer,
- Technical Director.

This pipeline focuses on architecture, optimization, safety, and production readiness. [file:128]

### Hybrid Parallel Execution
If the prompt needs both research and coding, the system runs both sub-councils in parallel using a thread pool, then merges the outputs into one final verdict. [file:128]

### Rate Limiting
The framework includes a per-model cooldown system so requests are spaced safely and consistently. This helps reduce API pressure and makes the system more stable in multi-agent workflows. [file:128]

### Retry and Error Handling
The agent layer classifies API errors into:
- fatal,
- transient,
- rate-limited.

Based on that classification, it applies adaptive retry behavior with backoff. [file:128]

### Final Supreme Council
After the sub-councils finish, a final arbitration stage synthesizes the research and technical briefs into one authoritative response. [file:128]

---

## Tech Stack

- **Python 3**
- **Google Gemini API**
- **python-dotenv**
- **argparse**
- **logging**
- **threading**
- **concurrent.futures**

---

## Project Architecture

The system is organized as a hierarchical pipeline:

1. **Router**
   - Classifies the prompt type.

2. **Research Council**
   - Produces a research brief for theory-heavy prompts.

3. **Technical Council**
   - Produces an engineering brief for code-heavy prompts.

4. **Supreme Council**
   - Merges and finalizes the response.

This structure makes the application more than a wrapper around an LLM. It acts more like a coordinated AI decision engine. [file:128][web:139][web:144]

---

## How It Works

### 1. Input Prompt
The user provides a prompt from the CLI or interactively.

### 2. Routing
The router model decides whether the request is:
- research,
- code,
- or hybrid. [file:128]

### 3. Specialist Agents
Depending on the route, the framework launches:
- a research pipeline,
- a code pipeline,
- or both in parallel. [file:128]

### 4. Synthesis
The Supreme Council combines the results and prints the final response to the terminal. [file:128]

---

## Special Engineering Ideas

This project includes several strong engineering concepts:
- per-model locks,
- cooldown-based request spacing,
- retry logic with severity classification,
- empty-response validation,
- thread-safe model access,
- modular role-based prompting,
- concurrent hybrid execution. [file:128]

These are the kinds of details that make the project look like a serious AI systems engineering effort rather than a simple prompt wrapper. [web:139][web:145]

---

## Why This Project Is Interesting

This project demonstrates how multi-agent orchestration can improve structure, reliability, and specialization in AI systems. Modern agent frameworks often rely on sequential, concurrent, or handoff-based orchestration patterns, and your design fits that style well. [web:144][web:145]

It also reflects a real trend in LLM system design: using specialized agents to increase reasoning quality and reduce the weaknesses of single-model outputs. [web:139][web:143]

---

## Installation

### Requirements
- Python 3.10+
- Google Gemini API key
- `.env` file with your API key

### Environment setup
Create a `.env` file:
```env
GEMINI_API_KEY=your_api_key_here
AGENT_MODEL=gemini-2.5-flash
ROUTER_MODEL=gemini-2.5-flash-lite
```

### Install dependencies
```bash
pip install google-genai python-dotenv
```

### Run the project
```bash
python your_script_name.py "Your prompt here"
```

Or run interactively:
```bash
python your_script_name.py
```

---

## Usage Example

```bash
python your_script_name.py "Explain why multi-agent AI systems are useful for research workflows."
```

For hybrid prompts, the system will run both the research and technical sub-councils and then synthesize the final answer. [file:128]

---

## Configuration

You can customize the following values in the script:
- `AGENT_MODEL`
- `ROUTER_MODEL`
- model cooldown times,
- temperature settings for each agent role,
- retry count,
- default timeouts and pacing logic. [file:128]

---

## Strengths of the Project

- Clean modular architecture.
- Multiple specialized agent roles.
- Robust retry handling.
- Thread-safe rate limiting.
- Supports both research and coding workflows.
- Strong CLI-based workflow.
- Good base for future extension into a full agent platform. [file:128]

---

## Future Improvements

Possible upgrades include:
- web UI dashboard,
- conversation memory,
- persistent session history,
- plugin-based agent roles,
- structured JSON outputs,
- tool use for code execution and retrieval,
- citation-aware final synthesis,
- role-specific memory systems,
- support for more models and providers. [file:128][web:144][web:145]

---

## Learning Outcome

This project helped demonstrate skills in:
- AI orchestration,
- prompt engineering,
- concurrency,
- API handling,
- error recovery,
- systems design,
- agent-based reasoning pipelines,
- production-style software structure. [file:128]

---

## Disclaimer

This project is intended for educational and experimental use.  
It is not a replacement for human judgment, especially in high-stakes research or deployment settings. [web:143][web:145]

---

## Author

**Jabid Muntasir**  
Interested in Machine Learning, Competitive Programming, and Software Development

---

## License
**MIT License**

---
