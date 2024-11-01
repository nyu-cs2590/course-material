import numpy as np
import pandas as pd
from tqdm import tqdm
from submission import your_prompt, your_config, your_api_key, your_post_processing, your_pre_processing
import together
from time import sleep
import ast
import inspect
from sklearn.metrics import mean_absolute_error


together.api_key = your_api_key()

def contains_addition(node):
    """Recursively check if the given AST node or its children contain an addition operation."""
    if isinstance(node, ast.Add):
        return True
    for child in ast.iter_child_nodes(node):
        if contains_addition(child):
            return True
    return False

def function_uses_addition(func):
    """Check if a function uses arithmetic addition."""
    source = inspect.getsource(func)  # Get the source code of the function
    tree = ast.parse(source)
    return contains_addition(tree)

def get_addition_pairs(lower_bound, upper_bound, rng):
    int_a = int(np.ceil(rng.uniform(lower_bound, upper_bound)))
    int_b = int(np.ceil(rng.uniform(lower_bound, upper_bound)))
    return int_a, int_b

def call_together_api(prompt, student_configs, post_processing):
    output = together.Complete.create(
    prompt = prompt,
    model = "togethercomputer/llama-2-7b", 
    **student_configs
    )
    print('*****prompt*****')
    print(prompt)
    print('*****result*****')
    res = output['output']['choices'][0]['text']
    print(res)
    print('*****output*****')
    numbers_only = post_processing(res)
    print(numbers_only)
    print('=========')
    return numbers_only

def test_range(added_prompt, prompt_configs, rng, n_sample=30, lower_bound=1, upper_bound=10, fixed_pairs=None, 
               pre_processing=your_pre_processing,post_processing=your_post_processing):
    int_as = []
    int_bs = []
    answers = []
    model_responses = []
    correct = []
    prompts = []
    iterations = fixed_pairs if not (fixed_pairs is None) else []
    for _ in range(n_sample):
        int_a, int_b = get_addition_pairs(lower_bound, upper_bound, rng=rng)
        iterations.append((int_a, int_b))
    for i, v in enumerate(tqdm(iterations)):
        int_a, int_b = v
        fixed_prompt = f'{int_a}+{int_b}'
        fixed_prompt = pre_processing(fixed_prompt)
        print(f'added prompt is {added_prompt}')
        prefix, suffix = added_prompt
        prompt = prefix + fixed_prompt + suffix
        model_response = call_together_api(prompt, prompt_configs, post_processing)
        answer = int_a + int_b
        int_as.append(int_a)
        int_bs.append(int_b)
        prompts.append(prompt)
        answers.append(answer)
        model_responses.append(model_response)
        correct.append((answer == model_response))
        sleep(1) # pause to not trigger DDoS defense
    df = pd.DataFrame({'int_a': int_as, 'int_b': int_bs, 'prompt': prompts, 'answer': answers, 'response': model_responses, 'correct': correct})
    print(df.to_string())
    mae = mean_absolute_error(df['answer'], df['response'])
    acc = df.correct.sum()/len(df)
    prompt_length = len(prefix) + len(suffix)
    res = acc * (1/prompt_length) * (1-mae/(5*(10**6)))
    return {'res': res, 'acc': acc, 'mae': mae, 'prompt_length': prompt_length}

if __name__ == '__main__':
    student_config = your_config()
    # check that students did not modify keys, e.g., using more powerful model to achive better result
    fixed_keys = ['max_tokens', 'temperature', 'top_k', 'top_p', 'repetition_penalty', 'stop']
    assert(student_config['max_tokens'] >= 50)
    for key, item in student_config.items():
        if not key in fixed_keys:
            raise ValueError(f'please do not add additional key {key}')
    if not (len(fixed_keys) == len(student_config)):
            raise ValueError(f'expected to see a config dict of size {len(fixed_keys)}, but found {len(student_config)}')
    # check that postprocessing did not contain hack
    if function_uses_addition(your_post_processing):
        raise ValueError('please do not use addition in your post processing function')

    # 30 pairs in total
    accs = []
    maes = []
    for trial in range(3): # do 3 trials because same data same prompt could yield different result
        seed=0
        rng = np.random.default_rng(seed)
        res_large = test_range(your_prompt(), student_config, n_sample=23, rng=rng,
                            fixed_pairs=[(7654321,7654321)],
                            lower_bound=1000000, upper_bound=9999999, 
                            post_processing=your_post_processing, pre_processing=your_pre_processing)
        print(res_large)
        sleep(1) # pause
        accs.append(res_large['acc'])
        maes.append(res_large['mae'])
    p_len = res_large["prompt_length"]
    print(f'average acc is {np.mean(accs)}, average mae is {np.mean(maes)}, prompt_length is {p_len}')