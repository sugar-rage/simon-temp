"""Ollama Client - post-TTS offline contextual intelligence layer.

Handles non-blocking, safe execution of local LLM queries via Ollama API.
Never blocks the vision pipeline or speech engine.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, Any, Dict

logger = logging.getLogger("simon.core.visual_context.ollama")


class OllamaClient:
    """Client for interacting with local Ollama model for post-TTS intelligence."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "auto",
        timeout_s: float = 10.0,
        enabled: bool = True,
    ):
        self._host = host.rstrip("/")
        self._configured_model = model
        self._model = self._resolve_model(model)
        self._timeout_s = timeout_s
        self._enabled = enabled
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="OllamaWorker")

    def _resolve_model(self, requested_model: str) -> str:
        """Resolve model name, querying /api/tags if auto or fallback is needed."""
        if requested_model and requested_model != "auto" and requested_model != "llama3.2":
            return requested_model
        try:
            url = f"{self._host}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                if requested_model in models:
                    return requested_model
                if models:
                    logger.info("[OLLAMA] Auto-selected local model: %s", models[0])
                    return models[0]
        except Exception:
            pass
        return "mistral" if requested_model == "auto" else requested_model

    def analyze_post_tts_async(
        self,
        payload: Dict[str, Any],
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Submit a post-TTS visual context analysis task asynchronously."""
        self._executor.submit(self._run_analysis, payload, on_complete)

    def answer_query_async(
        self,
        query: str,
        context_data: str,
        on_complete: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Submit a user query grounded in recent visual context asynchronously."""
        self._executor.submit(self._run_query, query, context_data, on_complete)

    def answer_query_sync(self, query: str, context_data: str) -> str:
        """Synchronous query for user Q&A when called from a query handler."""
        if not self._enabled:
            return self._fallback_answer_query(query, context_data)
        try:
            return self._call_ollama_query(query, context_data)
        except Exception as e:
            logger.warning("[OLLAMA] Ollama query failed (%s), using fallback heuristic", e)
            return self._fallback_answer_query(query, context_data)

    def _run_analysis(
        self,
        payload: Dict[str, Any],
        on_complete: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        """Worker thread execution for visual context analysis."""
        logger.info("[OLLAMA] post-TTS analysis started")
        result: Dict[str, Any]
        try:
            if not self._enabled:
                result = self._fallback_extract(payload)
                result["_source"] = "fallback"
            else:
                result = self._call_ollama_analysis(payload)
                result["_source"] = "ollama"
            logger.info("[OLLAMA] analysis completed (source=%s)", result.get("_source"))
        except Exception as e:
            logger.warning("[OLLAMA] Ollama API error or unavailable (%s), using heuristic fallback", e)
            result = self._fallback_extract(payload)
            result["_source"] = "fallback"
            logger.info("[OLLAMA] analysis completed (fallback)")

        if on_complete:
            try:
                on_complete(result)
            except Exception as exc:
                logger.error("[OLLAMA] Error in on_complete callback: %s", exc)

    def _run_query(
        self,
        query: str,
        context_data: str,
        on_complete: Optional[Callable[[str], None]],
    ) -> None:
        try:
            answer = self.answer_query_sync(query, context_data)
            if on_complete:
                on_complete(answer)
        except Exception as exc:
            logger.error("[OLLAMA] Error in query worker: %s", exc)

    def _call_ollama_analysis(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call Ollama /api/generate with JSON schema prompt for visual intelligence."""
        prompt = (
            "You are an assistive vision AI for SIMON. Analyze this visual scene data and return ONLY a valid JSON object.\n"
            "Input Visual Data:\n"
            f"{json.dumps(payload, indent=2)}\n\n"
            "Return JSON with keys:\n"
            "{\n"
            '  "event": "<short event name, e.g. road_closure, bus_stop, restaurant, stop_sign, warning, shop>",\n'
            '  "direction": "<direction if known, e.g. ahead, left, right or null>",\n'
            '  "distance": "<distance if known, e.g. 500m, 2km or null>",\n'
            '  "recommended_action": "<concise advice or null>",\n'
            '  "place_name": "<place or shop name if recognized, or null>",\n'
            '  "category": "<safety, navigation, restaurant, store, transport, or general>",\n'
            '  "summary": "<1 sentence natural summary of the scene knowledge>"\n'
            "}"
        )
        url = f"{self._host}/api/generate"
        req_data = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(req_data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            raw_response = res_json.get("response", "{}")
            parsed = json.loads(raw_response)
            return parsed

    def _call_ollama_query(self, query: str, context_data: str) -> str:
        """Call Ollama for answering a user question based on recent visual memory."""
        prompt = (
            "You are SIMON, a helpful assistive voice AI. Answer the user's question concisely in 1 or 2 sentences based ONLY on the recent visual memory provided below.\n"
            "If the information is not in the memory, say honestly that you haven't seen it recently.\n\n"
            f"Recent Visual Memory:\n{context_data}\n\n"
            f"User Question: {query}\n\n"
            "Answer:"
        )
        url = f"{self._host}/api/generate"
        req_data = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(req_data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            return res_json.get("response", "").strip()

    def _fallback_extract(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic heuristic extractor when Ollama is offline or unavailable."""
        ocr_text = payload.get("ocr_text", "")
        classification = payload.get("classification", "UNKNOWN")
        spoken = payload.get("spoken_response", "")
        direction = payload.get("direction", "ahead")
        distance = payload.get("distance")

        # Ambiguous interpretation
        if payload.get("is_ambiguous"):
            ocr_clean = ocr_text.replace("...", " ").strip()
            return {
                "event": "ambiguous_sign",
                "direction": direction,
                "distance": distance,
                "recommended_action": "proceed with caution",
                "place_name": None,
                "category": "safety" if "SAFETY" in classification else "general",
                "summary": f"Ambiguous sign detected containing fragment '{ocr_clean}'.",
            }

        # Place / Restaurant extraction
        if "RESTAURANT" in ocr_text.upper() or "HOTEL" in ocr_text.upper() or "CAFE" in ocr_text.upper():
            lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
            place_name = lines[0] if lines else ocr_text
            return {
                "event": "restaurant",
                "direction": direction,
                "distance": distance,
                "recommended_action": None,
                "place_name": place_name,
                "category": "restaurant",
                "summary": f"Restaurant '{place_name}' detected nearby.",
            }

        # Road closure
        if "ROAD CLOSED" in ocr_text.upper() or "ROAD BLOCKED" in ocr_text.upper():
            return {
                "event": "road_closure",
                "direction": direction,
                "distance": distance or "ahead",
                "recommended_action": "take diversion" if "DIVERSION" in ocr_text.upper() else "stop and turn around",
                "place_name": None,
                "category": "safety",
                "summary": "Road closure ahead. Take alternate route.",
            }

        # Distance / Landmark
        if "KM" in ocr_text.upper() or "METERS" in ocr_text.upper():
            return {
                "event": "destination_sign",
                "direction": direction,
                "distance": distance,
                "recommended_action": None,
                "place_name": ocr_text,
                "category": "navigation",
                "summary": f"Distance marker for '{ocr_text}'.",
            }

        # Generic fallback
        return {
            "event": classification.lower(),
            "direction": direction,
            "distance": distance,
            "recommended_action": None,
            "place_name": ocr_text if len(ocr_text) < 40 else None,
            "category": "safety" if "SAFETY" in classification else "general",
            "summary": spoken if spoken else f"Detected {ocr_text} {direction or 'ahead'}.",
        }

    def _fallback_answer_query(self, query: str, context_data: str) -> str:
        """Deterministic rule-based query matching when Ollama is offline."""
        q_lower = query.lower()
        if not context_data or "No recent visual memory" in context_data:
            return "I haven't seen any signs or places about that recently."

        if "restaurant" in q_lower or "food" in q_lower or "eat" in q_lower:
            for line in context_data.splitlines():
                if any(w in line.lower() for w in ("restaurant", "hotel", "cafe", "meals", "bakery")):
                    return f"I noticed {line.strip()}."
            return "I haven't seen any restaurants nearby recently."

        if "bus" in q_lower or "stop" in q_lower:
            for line in context_data.splitlines():
                if "bus" in line.lower():
                    return f"I noticed {line.strip()}."
            return "I haven't seen any bus stops nearby."

        if "sign" in q_lower or "board" in q_lower or "see" in q_lower or "around" in q_lower:
            lines = [l.strip() for l in context_data.splitlines() if l.strip()]
            if lines:
                return f"Recently I observed: {', '.join(lines[:2])}."

        return "I haven't seen any signs or information about that recently."
