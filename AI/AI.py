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
from prompt_toolkit import prompt

load_dotenv()

# ========================== CONFIGURATION CONSTANTS ==========================
AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-2.5-flash")
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "gemini-2.5-flash-lite")

COOLDOWN_BY_MODEL: Dict[str, float] = {
    "gemini-2.5-flash": 12.0,
    "gemini-2.5-flash-lite": 8.0,
}
DEFAULT_COOLDOWN = 12.0

TEMP_ROUTER = 0.0
TEMP_RESEARCH_LEAD = 0.5
TEMP_DEVILS_ADVOCATE = 0.4
TEMP_FACT_CHECKER = 0.1
TEMP_ARCHITECT = 0.4
TEMP_OPTIMIZER = 0.3
TEMP_SECURITY = 0.1
TEMP_CHAIR = 0.2
TEMP_TECHNICAL_DIRECTOR = 0.2
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
    Thread‑safe, per‑model rate limiter using a Condition variable.
    Guarantees strict spacing intervals between model invocations.
    """
    def __init__(self, cooldown_by_model: Dict[str, float], default_cooldown: float) -> None:
        self._cooldown_by_model = cooldown_by_model
        self._default_cooldown = default_cooldown
        self._conditions: Dict[str, threading.Condition] = {}
        self._last_call: Dict[str, float] = {}
        self._registry_lock = threading.Lock()

    def _get_condition(self, model: str) -> threading.Condition:
        with self._registry_lock:
            if model not in self._conditions:
                self._conditions[model] = threading.Condition(threading.Lock())
            return self._conditions[model]

    def wait_and_reserve(self, model: str) -> None:
        """
        Blocks until the cooldown window has closed, then reserves the next slot.
        Wakes up waiting threads safely using notification chains.
        """
        cooldown = self._cooldown_by_model.get(model, self._default_cooldown)
        cond = self._get_condition(model)

        with cond:
            while True:
                now = time.monotonic()
                last = self._last_call.get(model, 0.0)
                remaining = cooldown - (now - last)
                
                if remaining <= 0:
                    self._last_call[model] = now
                    # Wake up any other waiting threads so they can recalculate their slots
                    cond.notify_all()
                    break
                
                cond.wait(timeout=remaining)


class EnterpriseAIAuthority:
    """
    Enterprise‑grade, hierarchical multi‑agent framework implementing
    Andrej Karpathy's LLM Council theory with automated workload routing.
    """
    def __init__(self) -> None:
        api_key: Optional[str] = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable is missing.")
            sys.exit(1)

        self.client = genai.Client(api_key=api_key)
        self.model_agent = AGENT_MODEL
        self.model_router = ROUTER_MODEL
        self.rate_limiter = ModelRateLimiter(COOLDOWN_BY_MODEL, DEFAULT_COOLDOWN)

    @staticmethod
    def _is_retryable(exc: Exception) -> int:
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
        prompt_text: str,
        temp: float,
        model: Optional[str] = None,
        max_retries: int = 3,
    ) -> str:
        if model is None:
            model = self.model_agent

        for attempt in range(max_retries + 1):
            try:
                # Pacing check happens immediately before execution attempt
                self.rate_limiter.wait_and_reserve(model)
                
                logger.info(f"Dispatching payload to model -> '{model}' (Attempt {attempt + 1})")
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=temp,
                    ),
                )

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
                    raise RuntimeError(f"Fatal: non‑retryable error from model '{model}': {e}") from e

                if attempt < max_retries:
                    base_wait = 12 if severity == 2 else 6
                    wait_duration = base_wait * (attempt + 1)
                    logger.info("Backing off. Retrying in %ds...", wait_duration)
                    time.sleep(wait_duration)
                else:
                    raise RuntimeError(f"Fatal: Agent execution failed after {max_retries + 1} attempts: {e}") from e
        return ""

    def route_user_prompt(self, user_query: str) -> str:
        logger.info("[Gatekeeper Router] Analyzing workload metadata and structural context...")
        router_instruction = (
            "You are the Gatekeeper Router of an advanced multi‑agent ecosystem. Your sole job is to classify "
            "the user's prompt into exactly one of three operational modes based on its requirements:\n"
            "1. 'CORE_RESEARCH' : If the prompt is purely theoretical, scientific, academic, or mathematical with no coding required.\n"
            "2. 'CORE_CODE'     : If the prompt is purely about writing code, debugging, syntax optimization, or system scripts with no academic research.\n"
            "3. 'HYBRID_WORKLOAD': If the prompt requires BOTH deep research/theoretical analysis AND technical architecture/coding implementation.\n"
            "Respond with EXACTLY one of these three strings: CORE_RESEARCH, CORE_CODE, or HYBRID_WORKLOAD. Do not include any other text, markdown block, or punctuation."
        )

        try:
            classification = self._execute_agent(
                system_instruction=router_instruction,
                prompt_text=user_query,
                temp=TEMP_ROUTER,
                model=self.model_router,
            )
            classification = classification.upper().replace("`", "").strip()

            if classification not in ("CORE_RESEARCH", "CORE_CODE", "HYBRID_WORKLOAD"):
                logger.warning("Router returned an unrecognized label: '%s'. Defaulting to HYBRID_WORKLOAD.", classification)
                return "HYBRID_WORKLOAD"
            return classification
        except Exception as e:
            logger.warning("Gatekeeper routing failed (%s). Defaulting to HYBRID_WORKLOAD.", e)
            return "HYBRID_WORKLOAD"

    def run_research_sub_council(self, user_query: str) -> str:
        logger.info("[Sub‑Council A] Academic & Research Board Convened.")
        sys_1 = "You are a world‑class Scientific Research Lead. Provide a rigorous, evidence‑based theoretical analysis of the domain problem."
        thesis = self._execute_agent(sys_1, user_query, temp=TEMP_RESEARCH_LEAD)
        logger.info("   -> A1 (Research Lead): Foundational thesis compiled.")

        sys_2 = "You are a ruthless Peer Reviewer and Devil's Advocate. Challenge assumptions, isolate cognitive biases, and expose logical fallacies in the thesis."
        critique = self._execute_agent(sys_2, f"Target Thesis to Audit:\n{thesis}", temp=TEMP_DEVILS_ADVOCATE)
        logger.info("   -> A2 (Devil's Advocate): Vulnerabilities and edge‑cases exposed.")

        sys_3 = "You are a cold Fact‑Checker. Cross‑examine the thesis and the critique against empirical laws. Isolate verified data from speculation."
        facts = self._execute_agent(sys_3, f"[Thesis]:\n{thesis}\n\n[Critique]:\n{critique}", temp=TEMP_FACT_CHECKER)
        logger.info("   -> A3 (Fact‑Checker): Epistemic boundaries validated.")

        sys_4 = "You are the Chair of the Research Board. Synthesize the debate into a pristine, high‑density, informative Academic Brief."
        research_brief = self._execute_agent(sys_4, f"Compile final report based on:\nIsolated Facts:\n{facts}\nCritique:\n{critique}", temp=TEMP_CHAIR)
        logger.info("[Sub‑Council A] Academic Research Brief Signed Off.")
        return research_brief

    def run_technical_sub_council(self, user_query: str) -> str:
        logger.info("[Sub‑Council B] Technical & Systems Engineering Board Convened.")
        sys_1 = "You are a Principal Systems Architect. Design a highly optimized, clean, and complete production‑ready code blueprint."
        blueprint = self._execute_agent(sys_1, user_query, temp=TEMP_ARCHITECT)
        logger.info("   -> B1 (Systems Architect): Structural code layout engineered.")

        sys_2 = "You are an elite Competitive Programmer. Refactor the blueprint to optimize worst‑case time complexity, space complexity, and remove redundant allocations."
        optimization_report = self._execute_agent(sys_2, f"Blueprint to optimize:\n{blueprint}", temp=TEMP_OPTIMIZER)
        logger.info("   -> B2 (Algorithm Optimizer): Computational bottlenecks and complexity boundaries solved.")

        sys_3 = "You are a Defensive Cyber Security Engineer. Audit the blueprints for memory state corruption, memory leaks, stack overflows, and recursion limits."
        security_report = self._execute_agent(sys_3, f"Blueprint:\n{blueprint}\n\nOptimized Code Metrics:\n{optimization_report}", temp=TEMP_SECURITY)
        logger.info("   -> B3 (Security Auditor): Boundary states and memory safety patches applied.")

        sys_4 = "You are the Technical Director. Synthesize the secured assets into a singular, flawless Engineering Brief containing the production‑ready code artifact."
        engineering_brief = self._execute_agent(sys_4, f"Finalize production code based on:\nOptimized Blueprint:\n{optimization_report}\nSecurity Audit:\n{security_report}", temp=TEMP_TECHNICAL_DIRECTOR)
        logger.info("[Sub‑Council B] Engineering Brief Signed Off.")
        return engineering_brief

    def deploy_authority(self, user_query: str) -> str:
        if not user_query or not user_query.strip():
            raise ValueError("user_query must be a non‑empty string.")

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

                # Use explicit built-in standard TimeoutError handling robustly
                try:
                    research_brief = future_research.result(timeout=600)
                except TimeoutError:
                    logger.error("Research sub‑council timed out after 600 seconds.")
                    research_brief = "[Research Sub‑Council timed out]"
                except Exception as e:
                    logger.error("Research sub‑council failed: %s", e)
                    research_brief = f"[Research Sub‑Council failed: {e}]"

                try:
                    engineering_brief = future_technical.result(timeout=600)
                except TimeoutError:
                    logger.error("Technical sub‑council timed out after 600 seconds.")
                    engineering_brief = "[Technical Sub‑Council timed out]"
                except Exception as e:
                    logger.error("Technical sub‑council failed: %s", e)
                    engineering_brief = f"[Technical Sub‑Council failed: {e}]"

        logger.info("[Supreme Authority] Convening Supreme Council for Final Arbitrage...")
        sys_supreme = (
            "You are the Supreme Truth‑Seeking Arbitrator. Your absolute mandate is intellectual honesty, accuracy, and truth. "
            "You are presented with a user's prompt alongside an Academic Research Brief and a Technical Engineering Brief. "
            "Analyze the inputs for confirmation bias, eliminate all pleasantries, fluff, or validation text. "
            "Deliver a step‑by‑step, research‑grade final response. If code is requested, ensure the final production‑ready script is fully embedded without bugs."
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
            prompt_text=supreme_input,
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
    parser = argparse.ArgumentParser(description="Enterprise AI Authority -- hierarchical multi‑agent LLM council.")
    parser.add_argument("query", nargs="?", default=None, help="The prompt to analyze.")
    parser.add_argument("--agent-model", default=None, help="Override agent model.")
    parser.add_argument("--router-model", default=None, help="Override router model.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug‑level logging.")
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
            "production‑ready Python script that prevents infinite loops and mitigates high‑variance logic crashes."
        )
        if not sys.stdin.isatty():
            query = demo_query
            logger.info("Non‑interactive terminal detected -- running the built‑in demo workload.")
        else:
            try:
                typed = prompt('> ', multiline=True)
            except EOFError:
                typed = ""
            query = typed if typed else demo_query
            if not typed:
                logger.info("No input provided -- running the built‑in demo workload.")

    try:
        authority.deploy_authority(query)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error("Fatal error: %s", e)
        sys.exit(1)
