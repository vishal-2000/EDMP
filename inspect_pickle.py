
import pickle
import sys
import os

# Adjust path if needed to find mpinets
sys.path.append(os.getcwd())
# Mock mpinets if needed or ensure path is correct
# The file load_test_dataset.py appends motion-policy-networks, let's try to mimic
sys.path.append(os.path.join(os.getcwd(), 'motion-policy-networks'))

try:
    with open('datasets/global_solvable_problems.pkl', 'rb') as f:
        data = pickle.load(f)
    print("Keys:", data.keys())
    
    first_key = list(data.keys())[0]
    print(f"Data['{first_key}'] keys:", data[first_key].keys())
    
    first_sub_key = list(data[first_key].keys())[0]
    problems = data[first_key][first_sub_key]
    print(f"Number of problems: {len(problems)}")
    
    first_problem = problems[0]
    print("Problem object type:", type(first_problem))
    print("Problem object dir:", dir(first_problem))
    if hasattr(first_problem, '__dict__'):
        print("Problem object dict keys:", first_problem.__dict__.keys())
    
    # Check for likely trajectory keys
    potential_keys = ['trajectory', 'path', 'solution', 'demo', 'traj']
    for key in potential_keys:
        if hasattr(first_problem, key):
            val = getattr(first_problem, key)
            print(f"Found '{key}': type={type(val)}")
            if hasattr(val, 'shape'):
                print(f"  Shape: {val.shape}")
            elif isinstance(val, list):
                print(f"  Length: {len(val)}")

except Exception as e:
    print(f"Error: {e}")
