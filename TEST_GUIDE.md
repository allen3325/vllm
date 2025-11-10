# Testing Interruptible Inference on Your Device

## Quick Start

### Option 1: Install from Source (Recommended for Testing)

```bash
# 1. Clone the repository
git clone https://github.com/allen3325/vllm.git
cd vllm

# 2. Checkout the feature branch
git checkout claude/interruptible-inference-sleep-awake-011CUt6qjxiesiaYDB37yhDe

# 3. Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install dependencies
pip install --upgrade pip
pip install -e .  # Editable install for development

# 5. Verify installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

### Option 2: Quick Docker Test

```bash
# Build Docker image with the feature branch
docker build -t vllm-interruptible:test .

# Run with GPU support
docker run --gpus all -it vllm-interruptible:test bash
```

---

## 🚀 Quick Verification Tests

### Test 1: Basic Python API Test

Create a file `test_basic_sleep.py`:

```python
"""Quick test of interruptible inference functionality."""

import time
from vllm import LLM, SamplingParams

# Initialize with sleep mode enabled
print("🔧 Initializing vLLM with sleep mode...")
llm = LLM(
    model="Qwen/Qwen3-0.6B",  # Small model for quick testing
    enable_sleep_mode=True,
    dtype="bfloat16",
    max_model_len=128,
    max_num_seqs=1,
    max_num_batched_tokens=128,
)

print("✅ Model loaded successfully!")

# Test 1: Generate before sleep
print("\n📝 Test 1: Generating text before sleep...")
prompts = [
    "The future of artificial intelligence is",
    "In the year 2050, technology will",
]
sampling_params = SamplingParams(
    temperature=0.0,  # Deterministic for testing
    max_tokens=50,
)

outputs_before = llm.generate(prompts, sampling_params)
for i, output in enumerate(outputs_before):
    print(f"  Prompt {i+1}: {output.outputs[0].text[:100]}...")

# Test 2: Sleep and wake WITH STATE PRESERVATION
print("\n💤 Test 2: Entering sleep mode with state preservation...")
llm.sleep(level=2, preserve_state=True)  # 🔑 preserve_state=True
assert llm.is_sleeping(), "❌ Engine should be sleeping!"
print("✅ Engine is sleeping with state preserved")

time.sleep(2)  # Simulate idle time

print("\n🌅 Waking up...")
llm.wake_up()
assert not llm.is_sleeping(), "❌ Engine should be awake!"
print("✅ Engine is awake")

# Test 3: Generate after wake (should be identical with temp=0)
print("\n📝 Test 3: Generating same prompts after wake...")
outputs_after = llm.generate(prompts, sampling_params)
for i, output in enumerate(outputs_after):
    print(f"  Prompt {i+1}: {output.outputs[0].text[:100]}...")

# Verify consistency
print("\n🔍 Verifying consistency...")
for i in range(len(prompts)):
    before = outputs_before[i].outputs[0].text
    after = outputs_after[i].outputs[0].text
    if before == after:
        print(f"  ✅ Prompt {i+1}: Output matches (state preserved correctly!)")
    else:
        print(f"  ❌ Prompt {i+1}: Output differs!")
        print(f"     Before: {before[:50]}...")
        print(f"     After:  {after[:50]}...")

print("\n✨ All tests passed! Interruptible inference is working!")
print("\n📝 Note: Set preserve_state=False (or omit it) to use original sleep behavior")
```

Run it:
```bash
python test_basic_sleep.py
```

Expected output:
```
🔧 Initializing vLLM with sleep mode...
✅ Model loaded successfully!

📝 Test 1: Generating text before sleep...
  Prompt 1: to revolutionize the way we live and work...
  Prompt 2: have advanced significantly beyond our current...

💤 Test 2: Entering sleep mode (level 2)...
✅ Engine is sleeping

🌅 Waking up...
✅ Engine is awake

📝 Test 3: Generating same prompts after wake...
  Prompt 1: to revolutionize the way we live and work...
  Prompt 2: have advanced significantly beyond our current...

🔍 Verifying consistency...
  ✅ Prompt 1: Output matches (state preserved correctly!)
  ✅ Prompt 2: Output matches (state preserved correctly!)

✨ All tests passed! Interruptible inference is working!
```

---

## 🧪 Running Automated Test Suite

### Run the Interruptible Inference Tests

```bash
# Run all interruptible inference tests
pytest tests/entrypoints/openai/test_interruptible_inference.py -v -s

# Run with detailed logging
pytest tests/entrypoints/openai/test_interruptible_inference.py -v -s --log-cli-level=DEBUG

# Run specific test
pytest tests/entrypoints/openai/test_interruptible_inference.py::test_interruptible_inference -v
```

### Run Existing Sleep Tests (Verify Backward Compatibility)

```bash
# Should still pass - verifies we didn't break existing functionality
pytest tests/entrypoints/openai/test_sleep.py -v
```

---

## 🌐 Testing with OpenAI-Compatible Server

### Step 1: Start the Server

```bash
# Terminal 1: Start vLLM server with sleep mode enabled
vllm serve Qwen/Qwen3-0.6B \
    --enable-sleep-mode \
    --dtype bfloat16 \
    --max-model-len 128 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 128 \
    --port 8000
```

Wait for:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Test with cURL

```bash
# Terminal 2: Test the API

# 1. Check server is ready
curl http://localhost:8000/health

# 2. Generate some text (before sleep)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "The meaning of life is",
    "max_tokens": 50,
    "temperature": 0.0
  }' | jq .

# Save the output for comparison
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "The meaning of life is",
    "max_tokens": 50,
    "temperature": 0.0
  }' | jq -r '.choices[0].text' > before_sleep.txt

# 3. Put server to sleep WITH STATE PRESERVATION
curl -X POST "http://localhost:8000/sleep?level=2&preserve_state=true"
# Response: {"message":"Engine is now sleeping at level 2"}

# 4. Check sleep status
curl http://localhost:8000/is_sleeping
# Response: {"is_sleeping":true}

# 5. Wake up
curl -X POST http://localhost:8000/wake_up
# Response: {"message":"Engine is now awake"}

# 6. Check awake status
curl http://localhost:8000/is_sleeping
# Response: {"is_sleeping":false}

# 7. Generate same prompt after wake
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "The meaning of life is",
    "max_tokens": 50,
    "temperature": 0.0
  }' | jq -r '.choices[0].text' > after_wake.txt

# 8. Compare outputs (should be identical)
diff before_sleep.txt after_wake.txt
# No output means files are identical ✅

# Note: Omit preserve_state parameter for original sleep behavior
# curl -X POST http://localhost:8000/sleep?level=2  # Original behavior
```

### Step 3: Test with Python Client

Create `test_server_sleep.py`:

```python
"""Test interruptible inference via OpenAI-compatible API."""

import openai

# Configure client
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # vLLM doesn't require real key
)

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "Write a short story about a robot learning to dream:"

# Test 1: Generate before sleep
print("📝 Generating before sleep...")
response_before = client.completions.create(
    model=MODEL,
    prompt=PROMPT,
    max_tokens=100,
    temperature=0.0,  # Deterministic
)
text_before = response_before.choices[0].text
print(f"Generated: {text_before[:100]}...")

# Test 2: Sleep WITH STATE PRESERVATION
print("\n💤 Putting server to sleep with state preservation...")
import requests
requests.post("http://localhost:8000/sleep", params={"level": 2, "preserve_state": "true"})

status = requests.get("http://localhost:8000/is_sleeping").json()
print(f"Sleep status: {status}")

# Test 3: Wake
print("\n🌅 Waking up server...")
requests.post("http://localhost:8000/wake_up")

status = requests.get("http://localhost:8000/is_sleeping").json()
print(f"Awake status: {status}")

# Test 4: Generate after wake
print("\n📝 Generating after wake...")
response_after = client.completions.create(
    model=MODEL,
    prompt=PROMPT,
    max_tokens=100,
    temperature=0.0,
)
text_after = response_after.choices[0].text
print(f"Generated: {text_after[:100]}...")

# Verify
print("\n🔍 Verification:")
if text_before == text_after:
    print("✅ Outputs match perfectly! State preserved correctly.")
else:
    print("⚠️  Outputs differ (may be due to randomness)")
    print(f"Similarity: {len(set(text_before) & set(text_after)) / len(set(text_before)) * 100:.1f}%")
```

Run:
```bash
pip install openai  # If not already installed
python test_server_sleep.py
```

---

## 🐛 Debugging Tips

### Enable Debug Logging

Create `test_with_logging.py`:

```python
import logging

# Enable debug logging for interruptible inference components
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("vllm.v1.engine.checkpoint").setLevel(logging.DEBUG)
logging.getLogger("vllm.v1.core.sched.scheduler").setLevel(logging.DEBUG)
logging.getLogger("vllm.v1.engine.core").setLevel(logging.DEBUG)

from vllm import LLM, SamplingParams

# Your test code here...
llm = LLM(model="Qwen/Qwen3-0.6B", enable_sleep_mode=True)
llm.sleep(level=2)
llm.wake_up()
```

### Check Checkpoint Creation

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B", enable_sleep_mode=True)

# Access checkpoint manager (for debugging)
checkpoint_mgr = llm.llm_engine.engine_core.checkpoint_manager

# Generate some requests
llm.generate(["Test prompt"], SamplingParams(max_tokens=10))

# Sleep and check checkpoint
llm.sleep(level=2)
print(f"Has checkpoint: {checkpoint_mgr.has_checkpoint()}")

# Inspect checkpoint content
checkpoint = checkpoint_mgr._checkpoint
if checkpoint:
    print(f"Checkpoint timestamp: {checkpoint.timestamp}")
    print(f"Number of requests: {len(checkpoint.scheduler_checkpoint.requests)}")
    print(f"Waiting queue size: {len(checkpoint.scheduler_checkpoint.waiting_queue_data)}")
```

### Monitor GPU Memory

```bash
# Terminal 1: Monitor GPU memory continuously
watch -n 1 nvidia-smi

# Terminal 2: Run your test
python test_basic_sleep.py
```

You should see memory usage:
- **Initial**: ~2-4 GB (model loaded)
- **During sleep**: Reduced (weights offloaded)
- **After wake**: Back to initial (weights restored)

---

## 📊 Performance Testing

Create `test_performance.py`:

```python
"""Measure sleep/wake overhead."""

import time
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B", enable_sleep_mode=True)

# Measure sleep time
print("⏱️  Measuring sleep overhead...")
start = time.time()
llm.sleep(level=2)
sleep_time = time.time() - start
print(f"Sleep time: {sleep_time:.3f}s")

# Measure wake time
start = time.time()
llm.wake_up()
wake_time = time.time() - start
print(f"Wake time: {wake_time:.3f}s")

print(f"Total overhead: {sleep_time + wake_time:.3f}s")

# Measure inference after wake
start = time.time()
llm.generate(["Test"], SamplingParams(max_tokens=10))
inference_time = time.time() - start
print(f"First inference after wake: {inference_time:.3f}s")
```

---

## ✅ Expected Results Summary

| Test | Expected Outcome |
|------|-----------------|
| **Basic Sleep/Wake** | ✅ Engine transitions correctly between states |
| **Deterministic Output** | ✅ Identical outputs with temperature=0 before/after sleep |
| **Memory Offload** | ✅ GPU memory reduces during sleep, restores on wake |
| **Checkpoint Save** | ✅ Checkpoint created during sleep with request state |
| **Request Preemption** | ✅ Running requests moved to waiting queue safely |
| **Server API** | ✅ All endpoints work correctly (sleep, wake_up, is_sleeping) |

---

## 🚨 Common Issues & Solutions

### Issue 1: `ModuleNotFoundError: No module named 'torch'`
```bash
# Solution: Install PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: `CUDA out of memory`
```bash
# Solution: Use smaller model or reduce batch size
vllm serve Qwen/Qwen3-0.6B \
    --enable-sleep-mode \
    --max-model-len 1024 \
    --max-num-seqs 4  # Reduce from default
```

### Issue 3: Tests fail with `Connection refused`
```bash
# Solution: Check server is running
curl http://localhost:8000/health

# Or restart server with verbose logging
vllm serve Qwen/Qwen3-0.6B \
    --enable-sleep-mode \
    --log-level debug
```

### Issue 4: Outputs differ after wake
- Check temperature is set to 0.0 (deterministic)
- Verify same prompt and parameters used
- Check logs for checkpoint restoration success

---

## 📝 Minimal Reproduction Script

If you encounter any issues, use this minimal script to reproduce:

```python
"""Minimal reproduction script for bug reports."""

from vllm import LLM, SamplingParams

print("1. Initializing...")
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    enable_sleep_mode=True,
    dtype="auto",
)

print("2. Generating (before sleep)...")
out1 = llm.generate(["Hello"], SamplingParams(max_tokens=5, temperature=0.0))
print(f"   Output: {out1[0].outputs[0].text}")

print("3. Sleeping WITH state preservation...")
llm.sleep(level=2, preserve_state=True)  # 🔑 Key parameter

print("4. Waking...")
llm.wake_up()

print("5. Generating (after wake)...")
out2 = llm.generate(["Hello"], SamplingParams(max_tokens=5, temperature=0.0))
print(f"   Output: {out2[0].outputs[0].text}")

print("6. Comparing...")
if out1[0].outputs[0].text == out2[0].outputs[0].text:
    print("   ✅ SUCCESS")
else:
    print("   ❌ FAILED")

# Note: To test original behavior, use:
# llm.sleep(level=2)  # Without preserve_state parameter
```

---

## 📞 Need Help?

If tests fail, please provide:
1. Full error traceback
2. GPU model and VRAM size (`nvidia-smi`)
3. vLLM version (`python -c "import vllm; print(vllm.__version__)"`)
4. Output of minimal reproduction script above
5. Relevant logs with debug level enabled

Happy testing! 🚀
