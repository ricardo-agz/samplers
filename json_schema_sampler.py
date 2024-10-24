from typing import List, Dict, Optional, Type
import transformers
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# ANSI color codes
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"


def color_text(text: str, color: str) -> str:
    """
    Wraps the given text with ANSI color codes.
    """
    return f"{color}{text}{RESET}"


if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class TokenStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self, tokenizer, end_tokens):
        super().__init__()
        self.tokenizer = tokenizer
        self.end_tokens = end_tokens
        self.generated_tokens = []

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        self.generated_tokens.append(input_ids[0, -1].item())
        # Check if the last tokens match any of the end token sequences
        for end_token_sequence in self.end_tokens:
            if len(self.generated_tokens) >= len(end_token_sequence):
                if (
                    self.generated_tokens[-len(end_token_sequence) :]
                    == end_token_sequence
                ):
                    return True
        return False


def json_sampler(
    messages: List[Dict[str, str]],
    schema: Type[BaseModel],
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    top_k: int = 50,
    top_p: float = 0.95,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Optional[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    # **Optional Step:** Ensure pad_token_id is set properly
    if (
        tokenizer.pad_token_id is None
        or tokenizer.pad_token_id == tokenizer.eos_token_id
    ):
        # Assign a new pad token
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

    # Construct the schema description
    fields = schema.__fields__
    schema_instructions = (
        "Your response should strictly adhere to the following JSON format:\n{\n"
    )
    for field_name, field_info in fields.items():
        field_type = field_info.annotation.__name__
        schema_instructions += f'  "{field_name}": {field_type},\n'
    schema_instructions = schema_instructions.rstrip(",\n") + "\n}"

    system_message_exists = any(msg["role"] == "system" for msg in messages)
    if system_message_exists:
        # Append the schema description to the existing system message
        for msg in messages:
            if msg["role"] == "system":
                msg["content"] += "\n\n" + schema_instructions
                break
    else:
        # Insert a system message with the schema description at the beginning
        messages.insert(0, {"role": "system", "content": schema_instructions})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Start constructing the JSON output and colored output
    json_output = "{\n"
    colored_output = color_text("{\n", BLUE)
    generated_values = {}
    field_names = list(fields.keys())
    num_fields = len(field_names)

    for idx, field_name in enumerate(field_names):
        field_info = fields[field_name]
        field_type = field_info.annotation
        type_name = field_type.__name__

        # Inject the field name into the prompt and json_output
        injected_syntax = f'  "{field_name}": '
        json_output += injected_syntax
        colored_output += color_text(injected_syntax, BLUE)

        if type_name == "str":
            json_output += '"'
            colored_output += color_text('"', BLUE)
            end_strings = ['"\n', '",', '"}', '"']
        else:
            end_strings = [",\n", "\n", "}"]

        current_field_prompt = prompt + f"Assistant: {json_output}"

        # Convert end tokens to token IDs sequences
        end_token_sequences = [
            tokenizer.encode(es, add_special_tokens=False) for es in end_strings
        ]

        # **Modified Tokenization Step: Include attention_mask**
        inputs = tokenizer(current_field_prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        stopping_criteria = transformers.StoppingCriteriaList(
            [TokenStoppingCriteria(tokenizer, end_token_sequences)]
        )

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stopping_criteria=stopping_criteria,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode and process the generated text
        generated_text = tokenizer.decode(
            generated_ids[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        value = generated_text

        # Find the position of the end strings
        for es in end_strings:
            if es in value:
                value = value.split(es)[0]
                break

        value = value.strip()

        # Append the generated value to the JSON output
        if type_name == "str":
            json_output += f'{value}"'
            colored_output += color_text(f"{value}", RED) + color_text('"', BLUE)
        else:
            json_output += f"{value}"
            colored_output += color_text(f"{value}", RED)

        generated_values[field_name] = value

        # Add comma and newline if not the last field
        if idx < num_fields - 1:
            json_output += ",\n"
            colored_output += color_text(",\n", BLUE)
        else:
            json_output += "\n"
            colored_output += color_text("\n", BLUE)

        # Update the prompt with the generated value for the next field
        prompt = current_field_prompt + value
        if type_name == "str":
            prompt += '"'
        prompt += "\n"

    json_output += "}"
    colored_output += color_text("}", BLUE)

    # Validate the generated JSON
    try:
        data = json.loads(json_output)
        validated_data = schema(**data)
        # Print the colored JSON output
        print("\nGenerated JSON Output:")
        print(colored_output)
        return json_output
    except Exception as e:
        print(f"Invalid JSON generated: {e}")
        print(f"Generated JSON:\n{colored_output}")
        return None


if __name__ == "__main__":
    class JokeSchema(BaseModel):
        setup: str
        punchline: str

    messages = [
        {"role": "system", "content": "You are a funny AI assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]

    # model_id = "ministral/Ministral-3b-instruct"
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

    if json_output:
        print("\nGenerated JSON Output (Raw):")
        print(json_output)
    else:
        print("Failed to generate valid JSON.")
