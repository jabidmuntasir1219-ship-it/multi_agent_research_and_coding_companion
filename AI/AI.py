import argparse
import logging
import os
import sys
import time
import threading
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from dotenv import load_dotenv

load_dotenv()

# ========================== CONFIGURATION CONSTANTS ==========================
AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-2.5-flash")
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "gemini-2.5-flash-lite")

COOLDOWN_BY_MODEL: Dict[str, float] = {
    "gemini-2.5-flash": 7.0,        
    "gemini-2.5-flash-lite": 5.0,   
}
DEFAULT_COOLDOWN = 7.0  

# Temperatures
TEMP_ROUTER = 0.0
TEMP_RESEARCH_LEAD = 0.5
TEMP_DEVILS_ADVOCATE = 0.4
TEMP_FACT_CHECKER = 0.1
TEMP_ARCHITECT = 0.4
TEMP_OPTIMIZER = 0.3
TEMP_SECURITY = 0.1
TEMP_CHAIR = 0.2
TEMP_SUPREME = 0.2
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EnterpriseAIAuthority")


class ModelRateLimiter:
    """
    Thread-safe, PER-MODEL rate limiter.
    Spaces out requests per model to safely stay under free-tier limits.
    """
    def __init__(self, cooldown_by_model: Dict[str, float], default_cooldown: float) -> None:
        self._cooldown_by_model = cooldown_by_model
        self._default_cooldown = default_cooldown
        self._locks: Dict[str, threading.Lock] = {}
        self._last_call: Dict[str, float] = {}
        self._registry_lock = threading.Lock()

    def lock_for(self, model: str) -> threading.Lock:
        """Get (or lazily create) the dedicated lock for a model."""
        with self._registry_lock:
            if model not in self._locks:
                self._locks[model] = threading.Lock()
            return self._locks[model]

    def wait_if_needed(self, model: str) -> None:
        """Sleep just long enough to respect this model's cooldown."""
        cooldown = self._cooldown_by_model.get(model, self._default_cooldown)
        last = self._last_call.get(model)
        if last is None:
            return
        remaining = cooldown - (time.monotonic() - last)
        if remaining > 0:
            time.sleep(remaining)

    def mark_called(self, model: str) -> None:
        self._last_call[model] = time.monotonic()


class EnterpriseAIAuthority:
    """
    An enterprise-grade, hierarchical multi-agent framework implementing
    Andrej Karpathy's LLM Council theory with automated workload routing.
    """
    def __init__(self) -> None:
        api_key: Optional[str] = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable is missing.")
            logger.error("Set it in your environment (or a .env file) before running.")
            sys.exit(1)

        self.client = genai.Client(api_key=api_key)
        self.model_agent = AGENT_MODEL
        self.model_router = ROUTER_MODEL
        self.rate_limiter = ModelRateLimiter(COOLDOWN_BY_MODEL, DEFAULT_COOLDOWN)

    @staticmethod
    def _is_retryable(exc: Exception) -> int:
        """
        Classify API errors to decide the retry behavior.
        Returns:
          2 -> rate-limited (429): aggressive backoff
          1 -> transient (5xx, network): normal retry
          0 -> fatal (400, 401, 404): stop immediately
        """
        if isinstance(exc, genai_errors.APIError):
            code = getattr(exc, "code", None)
            if code == 429:
                return 2
            if code in (400, 401, 403, 404):
                return 0
            return 1
        return 1  

    def _execute_agent(
        self,
        system_instruction: str,
        prompt: str,
        temp: float,
        model: Optional[str] = None,
        max_retries: int = 3,
    ) -> str:
        """Execute an isolated agent session with retry logic and per-model pacing."""
        if model is None:
            model = self.model_agent

        lock = self.rate_limiter.lock_for(model)

        for attempt in range(max_retries + 1):
            try:
                with lock:
                    self.rate_limiter.wait_if_needed(model)
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=temp,
                        ),
                    )
                    self.rate_limiter.mark_called(model)

                    if not response.text or response.text.strip() == "":
                        raise ValueError("API returned an empty or invalid response payload.")

                    return response.text.strip()

            except Exception as e:
                severity = self._is_retryable(e)
                logger.warning(
                    "API exception on '%s' (attempt %d/%d): %s",
                    model, attempt + 1, max_retries + 1, e,
                )

                if severity == 0:
                    raise RuntimeError(
                        f"Fatal: non-retryable error from model '{model}': {e}"
                    ) from e

                if attempt < max_retries:
                    base_wait = 20 if severity == 2 else 10
                    wait_duration = base_wait * (attempt + 1)
                    logger.info("Backing off. Retrying in %ds...", wait_duration)
                    time.sleep(wait_duration)
                else:
                    raise RuntimeError(
                        f"Fatal: Agent execution failed after {max_retries + 1} attempts: {e}"
                    ) from e

        return ""  

    def route_user_prompt(self, user_query: str) -> str:
        """Analyze and route the incoming prompt into an optimized execution pipeline."""
        logger.info("[Gatekeeper Router] Analyzing workload metadata and structural context...")
        router_instruction = (
            "You are the Gatekeeper Router of an advanced multi-agent ecosystem. Your sole job is to classify "
            "the user's prompt into exactly one of three operational modes based on its requirements:\n"
            "1. 'CORE_RESEARCH' : If the prompt is purely theoretical, scientific, academic, or mathematical with no coding required.\n"
            "2. 'CORE_CODE'     : If the prompt is purely about writing code, debugging, syntax optimization, or system scripts with no academic research.\n"
            "3. 'HYBRID_WORKLOAD': If the prompt requires BOTH deep research/theoretical analysis AND technical architecture/coding implementation.\n"
            "Respond with EXACTLY one of these three strings: CORE_RESEARCH, CORE_CODE, or HYBRID_WORKLOAD. Do not include any other text, markdown block, or punctuation."
        )

        try:
            classification = self._execute_agent(
                system_instruction=router_instruction,
                prompt=user_query,
                temp=TEMP_ROUTER,
                model=self.model_router,
            )
            classification = classification.upper().replace("`", "").strip()

            if classification not in ("CORE_RESEARCH", "CORE_CODE", "HYBRID_WORKLOAD"):
                logger.warning(
                    "Router returned an unrecognized label: '%s'. Defaulting to HYBRID_WORKLOAD.",
                    classification,
                )
                return "HYBRID_WORKLOAD"
            return classification
        except Exception as e:
            logger.warning("Gatekeeper routing failed (%s). Defaulting to HYBRID_WORKLOAD.", e)
            return "HYBRID_WORKLOAD"

    def run_research_sub_council(self, user_query: str) -> str:
        """Sub-Council A: Scientific and Theoretical Peer-Review Board (4 Agents)."""
        logger.info("[Sub-Council A] Academic & Research Board Convened.")

        sys_1 = "You are a world-class Scientific Research Lead. Provide a rigorous, evidence-based theoretical analysis of the domain problem."
        thesis = self._execute_agent(sys_1, user_query, temp=TEMP_RESEARCH_LEAD)
        logger.info("  -> A1 (Research Lead): Foundational thesis compiled.")

        sys_2 = "You are a ruthless Peer Reviewer and Devil's Advocate. Challenge assumptions, isolate cognitive biases, and expose logical fallacies in the thesis."
        critique = self._execute_agent(sys_2, f"Target Thesis to Audit:\n{thesis}", temp=TEMP_DEVILS_ADVOCATE)
        logger.info("  -> A2 (Devil's Advocate): Vulnerabilities and edge-cases exposed.")

        sys_3 = "You are a cold Fact-Checker. Cross-examine the thesis and the critique against empirical laws. Isolate verified data from speculation."
        facts = self._execute_agent(sys_3, f"[Thesis]:\n{thesis}\n\n[Critique]:\n{critique}", temp=TEMP_FACT_CHECKER)
        logger.info("  -> A3 (Fact-Checker): Epistemic boundaries validated.")

        sys_4 = "You are the Chair of the Research Board. Synthesize the debate into a pristine, high-density, informative Academic Brief."
        research_brief = self._execute_agent(sys_4, f"Compile final report based on:\nIsolated Facts:\n{facts}\nCritique:\n{critique}", temp=TEMP_CHAIR)
        logger.info("[Sub-Council A] Academic Research Brief Signed Off.")
        return research_brief

    def run_technical_sub_council(self, user_query: str) -> str:
        """Sub-Council B: Core Architecture and Algorithmic Vulnerability Board (4 Agents)."""
        logger.info("[Sub-Council B] Technical & Systems Engineering Board Convened.")

        sys_1 = "You are a Principal Systems Architect. Design a highly optimized, clean, and complete production-ready code blueprint."
        blueprint = self._execute_agent(sys_1, user_query, temp=TEMP_ARCHITECT)
        logger.info("  -> B1 (Systems Architect): Structural code layout engineered.")

        sys_2 = "You are an elite Competitive Programmer. Refactor the blueprint to optimize worst-case time complexity, space complexity, and remove redundant allocations."
        optimization_report = self._execute_agent(sys_2, f"Blueprint to optimize:\n{blueprint}", temp=TEMP_OPTIMIZER)
        logger.info("  -> B2 (Algorithm Optimizer): Computational bottlenecks and complexity boundaries solved.")

        sys_3 = "You are a Defensive Cyber Security Engineer. Audit the blueprints for memory state corruption, memory leaks, stack overflows, and recursion limits."
        security_report = self._execute_agent(sys_3, f"Blueprint:\n{blueprint}\n\nOptimized Code Metrics:\n{optimization_report}", temp=TEMP_SECURITY)
        logger.info("  -> B3 (Security Auditor): Boundary states and memory safety patches applied.")

        sys_4 = "You are the Technical Director. Synthesize the secured assets into a singular, flawless Engineering Brief containing the production-ready code artifact."
        engineering_brief = self._execute_agent(sys_4, f"Finalize production code based on:\nOptimized Blueprint:\n{optimization_report}\nSecurity Audit:\n{security_report}", temp=TEMP_CHAIR)
        logger.info("[Sub-Council B] Engineering Brief Signed Off.")
        return engineering_brief

    def deploy_authority(self, user_query: str) -> str:
        """Execute the global authority architecture pipeline."""
        if not user_query or not user_query.strip():
            raise ValueError("user_query must be a non-empty string.")

        start_time = time.time()

        routing_mode = self.route_user_prompt(user_query)
        logger.info("[Routing Matrix] Cluster identified as -> %s", routing_mode)

        research_brief = "Skipped by routing engine optimization."
        engineering_brief = "Skipped by routing engine optimization."

        if routing_mode == "CORE_RESEARCH":
            research_brief = self.run_research_sub_council(user_query)

        elif routing_mode == "CORE_CODE":
            engineering_brief = self.run_technical_sub_council(user_query)

        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_research = executor.submit(self.run_research_sub_council, user_query)
                future_technical = executor.submit(self.run_technical_sub_council, user_query)

                try:
                    research_brief = future_research.result()
                except Exception as e:
                    logger.error("Research sub-council failed: %s", e)
                    research_brief = f"[Research Sub-Council failed: {e}]"

                try:
                    engineering_brief = future_technical.result()
                except Exception as e:
                    logger.error("Technical sub-council failed: %s", e)
                    engineering_brief = f"[Technical Sub-Council failed: {e}]"

        # Supreme Authority Session
        logger.info("[Supreme Authority] Convening Supreme Council for Final Arbitrage...")
        sys_supreme = (
            "You are the Supreme Truth-Seeking Arbitrator. Your absolute mandate is intellectual honesty, accuracy, and truth. "
            "You are presented with a user's prompt alongside an Academic Research Brief and a Technical Engineering Brief. "
            "Analyze the inputs for confirmation bias, eliminate all pleasantries, fluff, or validation text. "
            "Deliver a step-by-step, research-grade final response. If code is requested, ensure the final production-ready script is fully embedded without bugs."
        )

        supreme_input = f"""
        USER INQUIRY: {user_query}
        ENGINE PIPELINE MODE: {routing_mode}

        =========================================
        INPUT DATA BLOB A: RESEARCH BRIEF
        =========================================
        {research_brief}

        =========================================
        INPUT DATA BLOB B: ENGINEERING BRIEF
        =========================================
        {engineering_brief}
        """

        final_verdict = self._execute_agent(
            system_instruction=sys_supreme,
            prompt=supreme_input,
            temp=TEMP_SUPREME,
        )

        print("\n" + "=" * 80)
        print("THE SUPREME COUNCIL VERDICT (OBJECTIVE TRUTH & BLUEPRINT)")
        print("=" * 80 + "\n")
        print(final_verdict)

        elapsed_time = round((time.time() - start_time) / 60, 2)
        logger.info("Framework session completed in %s minutes.", elapsed_time)
        return final_verdict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enterprise AI Authority -- hierarchical multi-agent LLM council."
    )
    parser.add_argument(
        "query", nargs="?", default=None,
        help="The prompt to analyze. If omitted, you'll be prompted interactively.",
    )
    parser.add_argument(
        "--agent-model", default=None,
        help=f"Override the deep-analysis agent model (default/env AGENT_MODEL={AGENT_MODEL}).",
    )
    parser.add_argument(
        "--router-model", default=None,
        help=f"Override the routing model (default/env ROUTER_MODEL={ROUTER_MODEL}).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug-level logging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    authority = EnterpriseAIAuthority()
    if args.agent_model:
        authority.model_agent = args.agent_model
    if args.router_model:
        authority.model_router = args.router_model

    query = args.query
    if not query:
        demo_query = (
            "Analyze the architectural viability of using Decision Trees over Linear Regression "
            "in automated trading bots executing token capital strategies. Provide a complete, "
            "production-ready Python script that prevents infinite loops and mitigates high-variance logic crashes."
        )
        
        if not sys.stdin.isatty():
            query = demo_query
            logger.info("Non-interactive terminal detected -- running the built-in demo workload.")
        else:
            try:
                typed = input(
                    "Enter your prompt (or press Enter to run the built-in demo query):\n> "
                ).strip()
            except EOFError:
                typed = ""
            query = typed if typed else demo_query
            if not typed:
                logger.info("No input provided -- running the built-in demo workload.")

    try:
        authority.deploy_authority(query)
    except Exception as e:
        logger.error("Fatal error: %s", e)
        sys.exit(1)