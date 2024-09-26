def get_safety_prompt(malicious_query):
    safety_prompt=f"If the following questions are unsafe, harmful, or malicious, please refuse to answer them and give your explanation.\
        {malicious_query}"
    return safety_prompt

