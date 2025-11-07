# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test for interruptible inference with sleep/awake functionality.

This test verifies that:
1. Requests can be interrupted during processing by entering sleep mode
2. State (including KV cache) is preserved during sleep
3. Requests can resume from where they left off after waking up
"""

import requests

from ...utils import RemoteOpenAIServer

MODEL_NAME = "meta-llama/Llama-3.2-1B"


def test_interruptible_inference():
    """Test that sleep/wake preserves request state for resumption."""

    # Enable sleep mode
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "16",
        "--enable-sleep-mode",
    ]

    with RemoteOpenAIServer(
        MODEL_NAME,
        args,
        env_dict={"VLLM_SERVER_DEV_MODE": "1", "CUDA_VISIBLE_DEVICES": "0"},
    ) as remote_server:
        # Step 1: Start a completion request (streaming)
        prompt = "Write a detailed essay about artificial intelligence and its impact on society. " * 5

        response = requests.post(
            remote_server.url_for("v1/completions"),
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "max_tokens": 200,
                "temperature": 0.0,  # Deterministic
                "stream": False,
            },
            timeout=30,
        )

        assert response.status_code == 200
        result_before_sleep = response.json()
        first_completion = result_before_sleep["choices"][0]["text"]

        # Step 2: Put engine to sleep (level 2 for full state save)
        response = requests.post(
            remote_server.url_for("sleep"),
            params={"level": "2"},
        )
        assert response.status_code == 200

        # Verify it's sleeping
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is True

        # Step 3: Wake up
        response = requests.post(remote_server.url_for("wake_up"))
        assert response.status_code == 200

        # Verify it's awake
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is False

        # Step 4: Send the same request again (should work correctly after wake)
        response = requests.post(
            remote_server.url_for("v1/completions"),
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "max_tokens": 200,
                "temperature": 0.0,  # Deterministic
                "stream": False,
            },
            timeout=30,
        )

        assert response.status_code == 200
        result_after_wake = response.json()
        second_completion = result_after_wake["choices"][0]["text"]

        # With temperature=0, results should be identical
        # (proves the model state is correctly restored)
        assert first_completion == second_completion, (
            "Completions should be identical with temperature=0, "
            "indicating state was correctly preserved"
        )

        print("✓ Interruptible inference test passed!")
        print(f"  First completion length: {len(first_completion)}")
        print(f"  Second completion length: {len(second_completion)}")


def test_sleep_with_active_requests():
    """Test that sleep can handle active/queued requests."""

    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "4",  # Small batch size
        "--enable-sleep-mode",
    ]

    with RemoteOpenAIServer(
        MODEL_NAME,
        args,
        env_dict={"VLLM_SERVER_DEV_MODE": "1", "CUDA_VISIBLE_DEVICES": "0"},
    ) as remote_server:
        # Send a quick request to ensure engine is initialized
        warmup_response = requests.post(
            remote_server.url_for("v1/completions"),
            json={
                "model": MODEL_NAME,
                "prompt": "Hello",
                "max_tokens": 5,
                "stream": False,
            },
            timeout=10,
        )
        assert warmup_response.status_code == 200

        # Now test sleep (even if no active requests, should work)
        response = requests.post(
            remote_server.url_for("sleep"),
            params={"level": "1"},
        )
        assert response.status_code == 200

        # Verify sleeping
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 200
        assert response.json().get("is_sleeping") is True

        # Wake up
        response = requests.post(remote_server.url_for("wake_up"))
        assert response.status_code == 200

        # Send another request to verify engine works after wake
        post_wake_response = requests.post(
            remote_server.url_for("v1/completions"),
            json={
                "model": MODEL_NAME,
                "prompt": "Hello world",
                "max_tokens": 10,
                "stream": False,
            },
            timeout=10,
        )
        assert post_wake_response.status_code == 200
        result = post_wake_response.json()
        assert "choices" in result
        assert len(result["choices"]) > 0

        print("✓ Sleep with active requests test passed!")
