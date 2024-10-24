# Custom LLM Samplers

This is a work in progress repo of a few custom samplers for a language model. More will be added soon. 

## 1. 100% Accurate JSON Schema Sampler
Generate accurate structured outputs 100% of the time by manually injecting JSON tokens and letting the model only
generate the tokens corresponding to the values.

### Usage
```python
class JokeSchema(BaseModel):
    setup: str
    punchline: str

messages = [
    {"role": "system", "content": "You are a funny AI assistant."},
    {"role": "user", "content": "Tell me a joke."},
]

model_id = "meta-llama/Llama-3.2-1B-Instruct"

json_output = json_sampler(
    model_id=model_id,
    messages=messages,
    schema=JokeSchema,
    top_k=50,
    top_p=0.95,
    max_tokens=512,
    temperature=0.7,
)
```
Output:
```json
{
  "setup": "Why couldn't the bicycle stand up by itself?",
  "punchline": "Because it was two-tired."
}
```

### How it works overview

Given a JSON schema:
```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string"
    },
    "age": {
      "type": "integer"
    },
    "is_student": {
      "type": "boolean"
    }
  }
}
```

We will manually inject the following starting tokens:
```
{
  "name": "
```

And let the model generate the rest of the string, cutting it off when it generates the closing quote, then manually 
injecting the next JSON tokens:
```
{
  "name": "John",  // stopped model generation at the closing quote
  "age": "  // manually inject these tokens and resume model generation
```

And so on. This way, we can generate the model generates the correct JSON schema 100% of the time.
