import json
import os

import requests

API_KEY = os.getenv("HUGGINGFACE_API_KEY")


def ask_hf(
    system_prompt: str,
    user_prompt: str,
    model="meta-llama/Llama-2-70b-chat-hf",
    temperature: float = 0.5,
    max_gen_len: int = 256,
    max_retries: int = 3,
):
    """ Ask Meta's Llama-2 model to generate a caption for a given keypoints dataset sample.

    Args:
        prompt (str): Prompt to ask the LLM.
        timeout (int): Timeout in seconds. Defaults to 30.
        temperature (float): Temperature to use. Defaults to 0.5.
    """
    try:
        user_message = "{}".format(user_prompt)

        headers = {"Authorization": f"Bearer {API_KEY}",
                   "Content-Type": "application/json"}
        API_URL = f"https://api-inference.huggingface.co/models/{model}"

        def query(payload):
            response = requests.request("POST", API_URL, headers=headers, json=payload)
            return json.loads(response.content.decode("utf-8"))

        results = query({
            "inputs": f"<s>[INST] <<SYS>>\n{system_prompt}<</SYS>>{user_message} [/INST]",
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_gen_len,
                "repetition_penalty": 1.0,
                "return_full_text": False,
                "num_return_sequences": 1,
            },
            "options": {
                "wait_for_model": True,
            }
        })

        return results[0]["generated_text"]
    except Exception as e:
        recoverable_errors = [KeyError]
        if type(e) in recoverable_errors and max_retries > 0:
            print('{} Exception. Retrying after 15 seconds ({} retries left)...'.format(
                type(e).__name__, max_retries - 1))
            import time
            time.sleep(15)
            return ask_hf(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
                max_gen_len=max_gen_len,
                max_retries=max_retries - 1,
            )
        else:
            raise
