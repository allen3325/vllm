# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
State checkpoint manager for interruptible inference.

This module provides functionality to save and restore the complete state
of the vLLM engine for sleep/awake operations, enabling:
1. Safe preemption of running requests during sleep
2. KV cache state preservation
3. Seamless resumption of inference on wake_up
"""

import pickle
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class SchedulerCheckpoint:
    """Checkpoint of scheduler state for sleep/awake."""

    # All tracked requests (request_id -> Request)
    requests: dict[str, Request]

    # Waiting queue state
    waiting_queue_data: list[tuple[str, int, float]]  # (request_id, priority, arrival_time)

    # Running requests
    running_request_ids: list[str]

    # KV cache block allocations per request
    # request_id -> list[block_ids]
    kv_block_allocations: dict[str, list[int]]

    # Prefix cache state (if enabled)
    prefix_cache_state: dict[str, Any] | None


@dataclass
class OutputProcessorCheckpoint:
    """Checkpoint of output processor state."""

    # RequestState objects that need to be preserved
    request_states: dict[str, dict[str, Any]]  # request_id -> serialized state


@dataclass
class EngineCheckpoint:
    """Complete engine checkpoint for sleep/awake."""

    scheduler_checkpoint: SchedulerCheckpoint
    output_processor_checkpoint: OutputProcessorCheckpoint | None

    # Additional metadata
    timestamp: float
    vllm_version: str


class CheckpointManager:
    """Manages state checkpointing for interruptible inference."""

    def __init__(self):
        self._checkpoint: EngineCheckpoint | None = None
        self._kv_cache_metadata: dict[str, Any] | None = None

    def save_checkpoint(
        self,
        scheduler_checkpoint: SchedulerCheckpoint,
        output_processor_checkpoint: OutputProcessorCheckpoint | None = None,
    ) -> None:
        """Save a checkpoint of the engine state."""
        import time
        import vllm

        self._checkpoint = EngineCheckpoint(
            scheduler_checkpoint=scheduler_checkpoint,
            output_processor_checkpoint=output_processor_checkpoint,
            timestamp=time.time(),
            vllm_version=vllm.__version__,
        )
        logger.info(
            "Saved engine checkpoint with %d requests",
            len(scheduler_checkpoint.requests),
        )

    def restore_checkpoint(self) -> EngineCheckpoint | None:
        """Restore the most recent checkpoint."""
        if self._checkpoint is None:
            logger.warning("No checkpoint available to restore")
            return None

        logger.info(
            "Restoring engine checkpoint with %d requests",
            len(self._checkpoint.scheduler_checkpoint.requests),
        )
        return self._checkpoint

    def clear_checkpoint(self) -> None:
        """Clear the saved checkpoint."""
        self._checkpoint = None
        self._kv_cache_metadata = None
        logger.info("Cleared engine checkpoint")

    def has_checkpoint(self) -> bool:
        """Check if a checkpoint is available."""
        return self._checkpoint is not None

    def save_kv_cache_metadata(self, metadata: dict[str, Any]) -> None:
        """Save KV cache metadata (block allocations, hashes, etc.)."""
        self._kv_cache_metadata = metadata
        logger.debug("Saved KV cache metadata")

    def restore_kv_cache_metadata(self) -> dict[str, Any] | None:
        """Restore KV cache metadata."""
        return self._kv_cache_metadata


def serialize_request(request: Request) -> dict[str, Any]:
    """Serialize a Request object to a dictionary."""
    # Serialize all essential fields
    data = {
        "request_id": request.request_id,
        "client_index": request.client_index,
        "priority": request.priority,
        "arrival_time": request.arrival_time,
        "status": request.status,
        "stop_reason": request.stop_reason,
        "eos_token_id": request.eos_token_id,
        "max_tokens": request.max_tokens,
        "num_prompt_tokens": request.num_prompt_tokens,
        "num_computed_tokens": request.num_computed_tokens,
        "num_cached_tokens": request.num_cached_tokens,
        "num_preemptions": request.num_preemptions,
        "num_nans_in_logits": request.num_nans_in_logits,
        "num_output_placeholders": request.num_output_placeholders,

        # Token data
        "prompt_token_ids": request.prompt_token_ids,
        "_output_token_ids": request._output_token_ids.copy() if hasattr(request, "_output_token_ids") else [],
        "_all_token_ids": request._all_token_ids.copy() if hasattr(request, "_all_token_ids") else [],
        "spec_token_ids": request.spec_token_ids.copy(),

        # Block hashes for prefix caching
        "block_hashes": [h.to_bytes() if hasattr(h, "to_bytes") else bytes(h) for h in request.block_hashes],

        # Multi-modal
        "num_encoder_inputs": request.num_encoder_inputs,
        "has_encoder_inputs": request.has_encoder_inputs,

        # Params (serialize with pickle for complex objects)
        "sampling_params": pickle.dumps(request.sampling_params) if request.sampling_params else None,
        "pooling_params": pickle.dumps(request.pooling_params) if request.pooling_params else None,
        "lora_request": pickle.dumps(request.lora_request) if request.lora_request else None,
        "structured_output_request": pickle.dumps(request.structured_output_request) if request.structured_output_request else None,

        # Embeddings (save to CPU if tensor)
        "prompt_embeds": request.prompt_embeds.cpu() if request.prompt_embeds is not None else None,

        # Multi-modal features
        "mm_features": pickle.dumps(request.mm_features) if request.mm_features else None,

        # KV transfer params
        "kv_transfer_params": request.kv_transfer_params,

        # Cache salt
        "cache_salt": request.cache_salt,

        # Events
        "events": pickle.dumps(request.events),

        # Trace headers
        "trace_headers": dict(request.trace_headers) if request.trace_headers else None,
    }

    return data


def deserialize_request(data: dict[str, Any]) -> Request:
    """Deserialize a Request object from a dictionary."""
    from vllm.v1.core.kv_cache_utils import BlockHash

    # Recreate the Request object
    # Note: We need to reconstruct it manually since the __init__ has specific logic

    # First create with basic params
    sampling_params = pickle.loads(data["sampling_params"]) if data["sampling_params"] else None
    pooling_params = pickle.loads(data["pooling_params"]) if data["pooling_params"] else None
    lora_request = pickle.loads(data["lora_request"]) if data["lora_request"] else None
    mm_features = pickle.loads(data["mm_features"]) if data["mm_features"] else None

    request = Request(
        request_id=data["request_id"],
        prompt_token_ids=data["prompt_token_ids"],
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        eos_token_id=data["eos_token_id"],
        client_index=data["client_index"],
        arrival_time=data["arrival_time"],
        prompt_embeds=data["prompt_embeds"],
        mm_features=mm_features,
        lora_request=lora_request,
        cache_salt=data["cache_salt"],
        priority=data["priority"],
        trace_headers=data["trace_headers"],
        block_hasher=None,  # Will be set by scheduler if needed
    )

    # Restore state fields
    request.status = data["status"]
    request.stop_reason = data["stop_reason"]
    request.num_computed_tokens = data["num_computed_tokens"]
    request.num_cached_tokens = data["num_cached_tokens"]
    request.num_preemptions = data["num_preemptions"]
    request.num_nans_in_logits = data["num_nans_in_logits"]
    request.num_output_placeholders = data["num_output_placeholders"]

    # Restore token lists
    request._output_token_ids = data["_output_token_ids"]
    request._all_token_ids = data["_all_token_ids"]
    request.spec_token_ids = data["spec_token_ids"]

    # Recreate ConstantList wrappers (required for read-only access)
    from vllm.v1.utils import ConstantList
    request.output_token_ids = ConstantList(request._output_token_ids)
    request.all_token_ids = ConstantList(request._all_token_ids)

    # Restore block hashes
    request.block_hashes = [BlockHash(h) for h in data["block_hashes"]]

    # Restore structured output request
    if data["structured_output_request"]:
        request.structured_output_request = pickle.loads(data["structured_output_request"])

    # Restore KV transfer params
    request.kv_transfer_params = data["kv_transfer_params"]

    # Restore events
    request.events = pickle.loads(data["events"])

    return request
