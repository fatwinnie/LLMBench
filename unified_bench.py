import aiohttp
import asyncio
import time
from tqdm import tqdm
import random

questions = [
    "Why do we have kidneys?",
    "Why do we have a pituitary gland?",
    "Why do we have eyebrows?",
    "Why is the ocean salty?",
    "Why do we dream?",
]

async def fetch(session, url, model_name):
    start_time = time.time()
    question = random.choice(questions)

    json_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.7
    }

    try:
        async with session.post(url, json=json_payload, ssl=False) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Request failed with status {response.status}: {error_text}")
            
            response_json = await response.json()
            end_time = time.time()
            request_time = end_time - start_time

            # ✅ 標準 OpenAI 格式
            answer = response_json["choices"][0]["message"]["content"]
            completion_tokens = response_json["usage"]["completion_tokens"]

            return completion_tokens, request_time, question, answer

    except Exception as e:
        print(f"Error during request: {e}")
        return 0, 0, question, "ERROR"

async def bound_fetch(sem, session, url, model_name, pbar):
    async with sem:
        result = await fetch(session, url, model_name)
        pbar.update(1)
        return result

async def run(load_url, model_name, max_concurrent_requests, total_requests):
    sem = asyncio.Semaphore(max_concurrent_requests)
    async with aiohttp.ClientSession() as session:
        tasks = []
        with tqdm(total=total_requests) as pbar:
            for _ in range(total_requests):
                task = asyncio.ensure_future(bound_fetch(sem, session, load_url, model_name, pbar))
                tasks.append(task)
            results = await asyncio.gather(*tasks)

    completion_tokens = sum(r[0] for r in results)
    response_times = [r[1] for r in results]
    prompts_answers = [(r[2], r[3]) for r in results]
    return completion_tokens, response_times, prompts_answers

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python unified_bench.py <C> <N> <model_name>")
        print("Example: python unified_bench.py 2 10 gpt-3.5-turbo")
        sys.exit(1)

    C = int(sys.argv[1])
    N = int(sys.argv[2])
    model_name = sys.argv[3]

    # ✅ 使用 OpenAI 格式的 endpoint
    url = 'http://localhost:8503/v1/chat/completions'  # vLLM 或 Ollama 都可以

    start_time = time.time()
    completion_tokens, response_times, prompt_answers = asyncio.run(run(url, model_name, C, N))
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_request = sum(response_times) / len(response_times) if response_times else 0
    tokens_per_second = completion_tokens / total_time if total_time > 0 else 0

    print(f'\nPerformance Results:')
    print(f'  Total requests           : {N}')
    print(f'  Max concurrent requests  : {C}')
    print(f'  Total time               : {total_time:.2f} seconds')
    print(f'  Average time per request : {avg_time_per_request:.2f} seconds')
    print(f'  Tokens per second        : {tokens_per_second:.2f}')

    print(f'\nDetail prompt and Answer:')
    for i, (q, a) in enumerate(prompt_answers):
        print(f'\n--- Request #{i+1} ---')
        print(f'Prompt : {q}')
        print(f'Answer : {a}')
