# OpenAI Responses API: `input`, `instructions`, and Useful Examples

The `responses.create()` method lets you send prompts to an OpenAI model
and receive responses. The most important parameters are `model`,
`input`, and `instructions`.

## 1. Understand the main parameters

  ---------------------------------------------------------------------------
  Parameter               Purpose                 Example
  ----------------------- ----------------------- ---------------------------
  `model`                 Specifies which model   `"gpt-4.1"`
                          to use                  

  `input`                 The question or task    `"Explain Python lists"`
                          you want the model to   
                          perform                 

  `instructions`          Defines the model's     `"Act as a Python tutor"`
                          role, behavior, tone,   
                          or rules                

  `temperature`           Controls randomness,    `0.2`
                          when supported by the   
                          selected model/API      

  `max_output_tokens`     Limits the maximum      `300`
                          output length           
  ---------------------------------------------------------------------------

The key difference:

-   **`instructions` = How the model should behave.**
-   **`input` = What the model should do.**

## 2. Use both `input` and `instructions`

This is the most common pattern.

``` python
import os
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model=os.getenv("MODEL_NAME", "gpt-4.1"),
    instructions="You are a friendly Python tutor. Explain concepts simply with examples.",
    input="What is a Python dictionary?"
)

print(response.output_text)
```

**Explanation:** - `instructions` tells the model to behave like a
friendly Python tutor. - `input` contains the actual question. -
`response.output_text` retrieves the generated text. -
`os.getenv("MODEL_NAME", "gpt-4.1")` uses your environment variable if
set, otherwise uses the default model.

## 3. Give the model a specific role

You can change `instructions` depending on the task.

``` python
response = client.responses.create(
    model=os.getenv("MODEL_NAME", "gpt-4.1"),
    instructions="You are a senior Python developer. Give concise, production-quality code.",
    input="Write a function to remove duplicates from a list."
)

print(response.output_text)
```

Possible roles include: - Python tutor - Data analyst - Coding
assistant - Technical interviewer - Writing assistant - SQL expert

The role and rules go in `instructions`; the task goes in `input`.

## 4. Ask for a specific output format

You can use `instructions` to guide the structure of the answer.

``` python
response = client.responses.create(
    model=os.getenv("MODEL_NAME", "gpt-4.1"),
    instructions="""
    Explain programming concepts using this format:
    1. Definition
    2. Simple example
    3. Explanation of the code
    4. Common mistake
    """,
    input="Explain Python lambda functions."
)

print(response.output_text)
```

**Why this is useful:** You get consistent, organized responses, which
is especially helpful when building educational tools or AI
applications.

## 5. Send structured input with roles

Instead of passing a plain string, you can pass a list of messages with
roles.

``` python
response = client.responses.create(
    model=os.getenv("MODEL_NAME", "gpt-4.1"),
    instructions="Answer as a patient programming teacher.",
    input=[
        {
            "role": "user",
            "content": "What is a Python function?"
        }
    ]
)

print(response.output_text)
```

**Explanation:** `input` can contain structured messages. For a simple
question, a string is usually enough. A message list is useful when you
need to explicitly represent roles or provide a more structured
conversation.

Note that `instructions` is a separate parameter; you do not need to put
your instructions inside the `input` list.

## 6. Control the length of the response

Use `max_output_tokens` to limit how much text the model generates.

``` python
response = client.responses.create(
    model=os.getenv("MODEL_NAME", "gpt-4.1"),
    instructions="Answer in simple English.",
    input="Explain machine learning.",
    max_output_tokens=150
)

print(response.output_text)
```

**Explanation:** `max_output_tokens=150` limits the output to 150
tokens. Tokens are not exactly the same as words, and the limit may
cause a response to end before the answer is complete.

## 7. Continue a conversation across multiple API calls

If you want the model to remember the context of a previous response,
you can use `previous_response_id`.

``` python
# First question
response1 = client.responses.create(
    model=os.getenv("MODEL_NAME", "gpt-4.1"),
    instructions="You are a Python tutor.",
    input="Explain Python lists."
)

print("First answer:")
print(response1.output_text)

# Follow-up question
response2 = client.responses.create(
    model=os.getenv("MODEL_NAME", "gpt-4.1"),
    previous_response_id=response1.id,
    input="Give me three practical examples."
)

print("Follow-up answer:")
print(response2.output_text)
```

**Explanation:** - `response1.id` identifies the first response. -
`previous_response_id` links the next request to that response's
conversation context. - The follow-up can refer to the previous answer
without repeating the entire question.

Note: Previous responses may be subject to API data-retention settings.
Also, pass `instructions` again if you want those instructions to apply
explicitly to the next request.

## 8. Use variables and reusable prompts

This is helpful when building applications where the user's question
changes dynamically.

``` python
def ask_ai(question, role="You are a helpful assistant."):
    response = client.responses.create(
        model=os.getenv("MODEL_NAME", "gpt-4.1"),
        instructions=role,
        input=question
    )
    return response.output_text


answer = ask_ai(
    "Explain pandas DataFrames.",
    role="You are a data science teacher. Use simple examples."
)

print(answer)
```

**Explanation:** The function accepts a question and an optional role.
You can reuse the same function for different tasks without rewriting
the API call each time.

## 9. Handle errors properly

API calls can fail because of invalid credentials, network issues, rate
limits, or incorrect model names.

``` python
import os
from openai import OpenAI, APIError

client = OpenAI()

try:
    response = client.responses.create(
        model=os.getenv("MODEL_NAME", "gpt-4.1"),
        instructions="You are a helpful assistant.",
        input="Explain Python loops."
    )

    print(response.output_text)

except APIError as e:
    print(f"OpenAI API error: {e}")
```

**Explanation:** The `try/except` block helps you handle API errors
without unexpectedly terminating your program. You can also handle
specific exceptions such as authentication errors and rate-limit errors
if needed.

## 10. Compare the common patterns

  Use case                            Recommended approach
  ----------------------------------- -------------------------------------
  Simple question                     `input="Your question"`
  Give the model a role               `instructions="Your rules"`
  Ask a question with a role          Use both `instructions` and `input`
  Limit response length               Add `max_output_tokens`
  Continue a conversation             Use `previous_response_id`
  Build a reusable chatbot function   Create a Python function
  Handle API failures                 Use `try/except`

## 11. A reusable template to remember

This is a good starting point for most projects.

``` python
import os
from openai import OpenAI

client = OpenAI()

MODEL = os.getenv("MODEL_NAME", "gpt-4.1")

instructions = """
You are a helpful AI assistant.
Explain concepts clearly.
Use examples when appropriate.
Keep answers concise.
"""

user_prompt = "Explain the difference between a list and a tuple in Python."

response = client.responses.create(
    model=MODEL,
    instructions=instructions,
    input=user_prompt,
    max_output_tokens=300
)

print(response.output_text)
```

Make sure your API key is configured in the `OPENAI_API_KEY` environment
variable, and replace the default model with one available to your
account if necessary.

**The main idea:** Keep your reusable behavior rules in `instructions`,
put the current task or question in `input`, and use additional
parameters only when your application needs them.
