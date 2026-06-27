# LLM Inference Runtime

A production-grade LLM inference serving system built in C++17 with Python tooling.
Implements request queuing, dynamic batching, KV cache, scheduling, and live monitoring.

## Architecture

```
Client → FastAPI (Python) → C++ Runtime → llama.cpp → Qwen2.5-1.5B
                                  ↓
                         Request Queue (thread-safe)
                                  ↓
                         Dynamic Batcher
                                  ↓
                         KV Cache Manager
                                  ↓
                         Scheduler (FCFS / Priority / SJF)
```

## Setup (One Time)

```bash
git clone <your-repo>
cd llm-runtime
chmod +x setup.sh
./setup.sh
```

## Run

```bash
source venv/bin/activate

# Terminal 1
./build/llm_runtime --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf --threads 4

# Terminal 2
python api/main.py

# Terminal 3
streamlit run dashboard/app.py

# Terminal 4
python benchmark/load_test.py --users 1 5 10 20
```

## Results

Benchmark graphs saved to `results/graphs/`






The Problem
You know ChatGPT right?
Millions of people use it every day simultaneously. Have you ever wondered — how does it handle so many people at the same time without breaking?
That's the problem this project solves.

Simple Example
Imagine a customer care helpline.
Without any management system:
Caller 1 calls → agent picks up → talks for 10 mins
Caller 2 calls → waiting...
Caller 3 calls → waiting...
Caller 4 calls → waiting...
Everyone is frustrated. Very slow.
With a smart management system:
All callers call at once
→ System puts them in queue
→ Groups similar queries
→ Reuses previous answers
→ Everyone gets served faster

Your Project Does Exactly This — But for AI
Instead of callers → users asking AI questions
Instead of agents → AI model answering
Your software sits in the middle and manages everything smartly.

What Your Software Does in Plain English
1. Takes requests from multiple users at once
User 1: "What is photosynthesis?"
User 2: "Explain gravity"
User 3: "What is Python?"
All arriving at same time → your system handles all
2. Puts them in a smart waiting line
Not random chaos
Organized queue
Fair system
Nobody waits forever
3. Groups questions together
Instead of asking AI one by one
Ask AI 4 questions at once
AI answers all 4 together
Much faster
4. Remembers previous answers
User 1 asks: "What is AI?"
Your system remembers the answer

User 2 asks: "What is AI?"
Instead of asking AI again
Your system reuses the saved answer
Saves time and resources
5. Shows live statistics
How many users are waiting?
How fast is AI responding?
How much memory is being used?
All visible on a live dashboard
6. Measures performance
With 1 user  → AI responds in 5 seconds
With 5 users → AI responds in 7 seconds
With 10 users → AI responds in 12 seconds
All saved as graphs automatically

Real World Impact
Without your system:
10 users ask AI simultaneously
→ 9 users wait for first user to finish
→ Total time = 10 x 30 seconds = 5 minutes
With your system:
10 users ask AI simultaneously
→ Smart grouping and management
→ Total time = much less
→ Everyone happier

Who Uses This Kind of System in Real World
CompanyTheir VersionOpenAIPowers ChatGPTGooglePowers GeminiNVIDIATensorRT-LLMMetaPowers Llama API
You built a mini version of what these companies use.

One Line Summary
You built a smart traffic management system for an AI model that allows multiple users to use it simultaneously in an efficient and organized way.
