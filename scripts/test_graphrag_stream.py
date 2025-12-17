import asyncio
import json
import sys

import httpx


async def main() -> None:
    url = "http://127.0.0.1:8000/graphrag/chat"

    question = sys.argv[1] if len(sys.argv) > 1 else "孙少平和谁关系最密切？给出依据，并在末尾列出引用 chunk_id。"

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
        "stream": True,
        "top_k_chunks": 8,
        "max_hops": 2,
    }

    print(f"[request] {question}")
    print("[stream] start\n")

    event_type = None

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line:
                    continue

                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                    continue

                if line.startswith("data:"):
                    data_str = line[len("data:"):].strip()
                    try:
                        data = json.loads(data_str)
                    except Exception:
                        data = data_str

                    if event_type == "meta":
                        print("[meta]", json.dumps(data, ensure_ascii=False))

                    elif event_type == "token":
                        # 你的后端是 {"id":..., "delta":"..."}
                        if isinstance(data, dict):
                            delta = data.get("delta")
                            if delta:
                                print(delta, end="", flush=True)

                    elif event_type == "done":
                        print("\n\n[done]", json.dumps(data, ensure_ascii=False))
                        return

    print("\n[stream] ended without done event")


if __name__ == "__main__":
    asyncio.run(main())
