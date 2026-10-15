The most common prompting techniques with simple Python examples using the OpenRouter setup from earlier.

**Setup (reuse this helper)**

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL = "openai/gpt-4o-mini"

def ask(messages, **kwargs):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        **kwargs,
    )
    return response.choices[0].message.content
```

**1. Zero-shot prompting**

Give the task with no examples.

```python
prompt = "Classify the sentiment as positive, negative, or neutral: 'The food was cold and the service was slow.'"
print(ask([{"role": "user", "content": prompt}]))
```

**2. Few-shot prompting**

Show a few examples so the model learns the pattern.

```python
prompt = """Classify the sentiment.

Text: I love this phone!
Sentiment: positive

Text: It broke after two days.
Sentiment: negative

Text: The package arrived on Tuesday.
Sentiment: neutral

Text: The battery life is amazing.
Sentiment:"""
print(ask([{"role": "user", "content": prompt}]))
```

**3. Role / system prompting**

Set behavior and persona with a system message.

```python
messages = [
    {"role": "system", "content": "You are a strict Python code reviewer. Be concise."},
    {"role": "user", "content": "Review: def add(a,b): return a+b"},
]
print(ask(messages))
```

**4. Chain-of-thought (CoT)**

Ask the model to reason step by step before answering.

```python
prompt = """A shop sells pens at 12 rupees each. Ravi buys 7 pens and pays with a 100 rupee note.
How much change does he get? Think step by step, then give the final answer."""
print(ask([{"role": "user", "content": prompt}]))
```

**5. Structured output (JSON)**

Force a specific output format.

```python
import json

prompt = """Extract the details from this text and return only valid JSON with keys name, age, city.

Text: Priya is 28 years old and lives in Chennai."""
result = ask(
    [{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
)
print(json.loads(result))
```

Note that `response_format` support varies by model on OpenRouter. If a model doesn't support it, just say "return only JSON, no extra text" in the prompt.

**6. Prompt chaining**

Break a task into steps, feeding one output into the next.

```python
article = "Transformers use self-attention to process sequences in parallel, replacing recurrence..."

summary = ask([{"role": "user", "content": f"Summarize in 2 sentences:\n{article}"}])
tweet = ask([{"role": "user", "content": f"Turn this summary into a tweet:\n{summary}"}])
print(tweet)
```

**7. Self-consistency**

Sample multiple answers and take the most common one.

```python
from collections import Counter

question = "If a train travels 60 km in 45 minutes, what is its speed in km/h? Reply with just the number."
answers = [
    ask([{"role": "user", "content": question}], temperature=0.8).strip()
    for _ in range(5)
]
print(answers)
print("Majority answer:", Counter(answers).most_common(1)[0][0])
```

**8. Multi-turn conversation (memory)**

Keep the message history yourself to give the model context.

```python
history = [{"role": "system", "content": "You are a helpful assistant."}]

def chat(user_input):
    history.append({"role": "user", "content": user_input})
    reply = ask(history)
    history.append({"role": "assistant", "content": reply})
    return reply

print(chat("My name is Arun."))
print(chat("What is my name?"))
```

**Quick summary**

| Technique | When to use |
|---|---|
| Zero-shot | Simple, well-known tasks |
| Few-shot | Need a specific format or style |
| System/role | Control tone and behavior |
| Chain-of-thought | Math, logic, multi-step reasoning |
| Structured output | Feeding results into code |
| Prompt chaining | Complex tasks split into stages |
| Self-consistency | Improving accuracy on reasoning |
| Multi-turn | Chatbots needing context |
