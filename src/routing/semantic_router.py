"""
Semantic Router C++ Wrapper using ctypes.

Provides a fast, zero-latency in-process semantic clustering router powered by
AVX2 SIMD operations. Used to partition conversation histories into topic blocks
and dynamically route queries to the relevant conversation chapter.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# C-ABI Struct Definitions
# ---------------------------------------------------------------------------

class _SCR_MessageView(ctypes.Structure):
    _fields_ = [
        ("turn", ctypes.c_int),
        ("role", ctypes.c_char_p),
        ("content", ctypes.c_char_p),
    ]


class _SCR_RouteResult(ctypes.Structure):
    _fields_ = [
        ("has_match", ctypes.c_int),
        ("matched_block_id", ctypes.c_size_t),
        ("best_similarity", ctypes.c_float),
        ("new_topic_created", ctypes.c_int),
        ("num_matched_messages", ctypes.c_size_t),
        ("matched_messages", ctypes.POINTER(_SCR_MessageView)),
        ("has_safety_net", ctypes.c_int),
        ("safety_net", _SCR_MessageView),
        ("internal_payload", ctypes.c_void_p),
    ]


@dataclass
class RoutedMessage:
    turn: int
    role: str
    content: str


@dataclass
class RoutePayload:
    new_query: str
    matched_block_id: Optional[int]
    best_similarity: float
    new_topic_created: bool
    matched_messages: list[RoutedMessage] = field(default_factory=list)
    safety_net: Optional[RoutedMessage] = None

    def to_llm_messages(self) -> list[dict[str, str]]:
        """
        Produce a clean list of {'role': ..., 'content': ...} dictionaries
        for LLM context assembly:
          1. Matched topic block messages
          2. N-1 Safety Net (if from a different topic)
          3. The new user query
        """
        msgs = [{"role": m.role, "content": m.content} for m in self.matched_messages]
        if self.safety_net is not None:
            msgs.append({"role": self.safety_net.role, "content": self.safety_net.content})
        msgs.append({"role": "user", "content": self.new_query})
        return msgs


# ---------------------------------------------------------------------------
# Loader for semantic_router.dll
# ---------------------------------------------------------------------------

def _load_dll() -> ctypes.CDLL:
    candidates = [
        Path(__file__).parent / "semantic_router.dll",
        Path(__file__).parent.parent.parent.parent / "semantic_router_cpp" / "semantic_router.dll",
    ]
    for p in candidates:
        if p.is_file():
            try:
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(str(p.parent.resolve()))
                    except Exception:
                        pass
                dll = ctypes.CDLL(str(p.resolve()))
                logger.debug("Loaded semantic_router DLL from: %s", p)
                return dll
            except Exception as e:
                logger.error("Failed to load DLL from %s: %s", p, e)

    raise FileNotFoundError(
        f"semantic_router.dll not found in candidate paths: {[str(c) for c in candidates]}"
    )


_DLL: Optional[ctypes.CDLL] = None


def get_dll() -> ctypes.CDLL:
    global _DLL
    if _DLL is None:
        _DLL = _load_dll()
        # Bind function prototypes
        _DLL.scr_create.argtypes = [ctypes.c_size_t, ctypes.c_float]
        _DLL.scr_create.restype = ctypes.c_void_p

        _DLL.scr_destroy.argtypes = [ctypes.c_void_p]
        _DLL.scr_destroy.restype = None

        _DLL.scr_add_turn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
        ]
        _DLL.scr_add_turn.restype = ctypes.c_size_t

        _DLL.scr_route.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(_SCR_RouteResult),
        ]
        _DLL.scr_route.restype = ctypes.c_int

        _DLL.scr_free_route_result.argtypes = [ctypes.POINTER(_SCR_RouteResult)]
        _DLL.scr_free_route_result.restype = None

        _DLL.scr_export_json.argtypes = [ctypes.c_void_p]
        _DLL.scr_export_json.restype = ctypes.c_void_p

        _DLL.scr_free_string.argtypes = [ctypes.c_void_p]
        _DLL.scr_free_string.restype = None

        _DLL.scr_num_blocks.argtypes = [ctypes.c_void_p]
        _DLL.scr_num_blocks.restype = ctypes.c_size_t

        _DLL.scr_num_turns.argtypes = [ctypes.c_void_p]
        _DLL.scr_num_turns.restype = ctypes.c_size_t

        _DLL.scr_set_threshold.argtypes = [ctypes.c_void_p, ctypes.c_float]
        _DLL.scr_set_threshold.restype = None

        _DLL.scr_get_threshold.argtypes = [ctypes.c_void_p]
        _DLL.scr_get_threshold.restype = ctypes.c_float

    return _DLL


class SemanticRouter:
    """Python interface to the C++ AVX2 Semantic Clustering Router."""

    def __init__(self, dim: int = 384, threshold: float = 0.55):
        self._dll = get_dll()
        self._dim = dim
        self._handle = self._dll.scr_create(dim, threshold)
        if not self._handle:
            raise RuntimeError("Failed to allocate C++ SemanticClusteringRouter instance")

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "_handle") and self._handle:
            self._dll.scr_destroy(self._handle)
            self._handle = None

    def _to_float_array(self, embedding: Any):
        if embedding is None:
            return None
        if not isinstance(embedding, (list, tuple)):
            logger.debug("Provided embedding is not a list/tuple (%s), ignoring.", type(embedding))
            return None
        if len(embedding) != self._dim:
            logger.warning("Embedding dimension mismatch: expected %d, got %d. Falling back to internal engine.", self._dim, len(embedding))
            return None
        return (ctypes.c_float * self._dim)(*embedding)

    def add_turn(
        self,
        role: str,
        content: str,
        embedding: Optional[list[float]] = None
    ) -> int:
        """
        Record a conversation turn. If embedding is None, uses internal fallback engine.
        Returns the block_id to which the turn was assigned.
        """
        if not self._handle:
            raise RuntimeError("Router handle is closed")

        emb_ptr = self._to_float_array(embedding)
        return int(self._dll.scr_add_turn(
            self._handle,
            role.encode("utf-8"),
            content.encode("utf-8"),
            emb_ptr,
        ))

    def route(
        self,
        query: str,
        embedding: Optional[list[float]] = None
    ) -> RoutePayload:
        """
        Route a user query against current topic blocks.
        Returns a RoutePayload containing the matched block, messages, and safety net.
        """
        if not self._handle:
            raise RuntimeError("Router handle is closed")

        emb_ptr = self._to_float_array(embedding)
        res = _SCR_RouteResult()

        ok = self._dll.scr_route(
            self._handle,
            query.encode("utf-8"),
            emb_ptr,
            ctypes.byref(res),
        )
        if not ok:
            raise RuntimeError("scr_route failed")

        try:
            matched_block_id = int(res.matched_block_id) if res.has_match else None
            matched_msgs: list[RoutedMessage] = []

            for i in range(res.num_matched_messages):
                m = res.matched_messages[i]
                role = m.role.decode("utf-8", errors="replace") if m.role else ""
                content = m.content.decode("utf-8", errors="replace") if m.content else ""
                matched_msgs.append(RoutedMessage(turn=m.turn, role=role, content=content))

            safety_net: Optional[RoutedMessage] = None
            if res.has_safety_net:
                sn = res.safety_net
                sn_role = sn.role.decode("utf-8", errors="replace") if sn.role else ""
                sn_content = sn.content.decode("utf-8", errors="replace") if sn.content else ""
                safety_net = RoutedMessage(turn=sn.turn, role=sn_role, content=sn_content)

            return RoutePayload(
                new_query=query,
                matched_block_id=matched_block_id,
                best_similarity=float(res.best_similarity),
                new_topic_created=bool(res.new_topic_created),
                matched_messages=matched_msgs,
                safety_net=safety_net,
            )
        finally:
            self._dll.scr_free_route_result(ctypes.byref(res))

    def export_json(self) -> str:
        if not self._handle:
            raise RuntimeError("Router handle is closed")
        ptr = self._dll.scr_export_json(self._handle)
        if not ptr:
            return "{}"
        try:
            val = ctypes.cast(ptr, ctypes.c_char_p).value
            return val.decode("utf-8") if val else "{}"
        finally:
            self._dll.scr_free_string(ptr)

    @property
    def num_blocks(self) -> int:
        return int(self._dll.scr_num_blocks(self._handle)) if self._handle else 0

    @property
    def num_turns(self) -> int:
        return int(self._dll.scr_num_turns(self._handle)) if self._handle else 0

    @property
    def threshold(self) -> float:
        return float(self._dll.scr_get_threshold(self._handle)) if self._handle else 0.0

    @threshold.setter
    def threshold(self, val: float):
        if self._handle:
            self._dll.scr_set_threshold(self._handle, float(val))
