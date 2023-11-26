import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def ask_gpt(system_prompt, user_prompt, model="gpt-3.5-turbo", temperature=0.5, max_tokens=512, max_retries=3):
    """ Ask OpenAI's GPT to generate a caption for a given keypoints dataset sample.

    By default, uses the `gpt-3.5-turbo` model and a temperature of 0.5. See the OpenAI
    API docs for more information on these parameters. If a recoverable error occurs,
    the function will retry up to `max_retries` times.

    Args:
        system_prompt (str): Prompt to ask the GPT.
        user_prompt (str): Prompt to ask the GPT.
        model (str): Model to use. Defaults to `gpt-3.5-turbo`.
        temperature (float): Temperature to use. Defaults to 0.5.
        max_tokens (int): Maximum number of tokens to generate. Defaults to 512.
        max_retries (int): Maximum number of retries. Defaults to 3.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "{}".format(system_prompt)},
                {"role": "user", "content": "{}".format(user_prompt)}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=1.0,
        )

        caption = response.choices[0].message.content
    except Exception as e:
        recoverable_errors = [openai.error.ServiceUnavailableError,
                              openai.error.RateLimitError, openai.error.APIError, openai.error.Timeout]
        if type(e) in recoverable_errors and max_retries > 0:
            print('{} Exception. Retrying after 15 seconds ({} retries left)...'.format(
                type(e).__name__, max_retries - 1))
            import time
            time.sleep(15)
            return ask_gpt(system_prompt, user_prompt, model, temperature, max_retries - 1)
        else:
            raise

    return caption


"""
You are an expert human activity and pose analyzer with deep understanding of MPII Human Pose dataset, which has 16 keypoints in following order: 0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist. You precisely describe body poses in terms of relative limb locations, given a set of 2D keypoint coordinates from the MPII dataset as (x,y) with -1 for invisible kpts. Your descriptions use a semi-standard template, such as 'There [is|are] [count] people in the image who [activity_verbs] [activity_name].' This is followed by a sentence some general attributes describing the activity in kpts context, and then pose description for each person in the list. foreach $person in image: 'The [first|left|right|center_pos] person is [sitting|standing|state] with their [limb]...' foreach $limb in (l leg, r leg, l arm, r arm, torso, head): describe how these limbs are positioned relative to other limbs, bend angles, other similar pose info, etc. Be concise and precise.
"""