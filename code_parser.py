import re

def extract_code_block(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
        
    # Find the first full Python code block
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None
def execute_code(code):
    try:
        print("Executing code...")
        print("-" * 20)
        namespace = {}
        exec(code, namespace)
    except Exception as e:
        print(f"Error executing code: {e}")
        print(f"Error details: {str(e)}")
        import traceback 
        print("\nFull Traceback:")
        print(traceback.format_exc())
        print("-" * 20)

if __name__ == "__main__":
    filepath = "GLang/ai_response.txt"  # Update path as needed
    code = extract_code_block(filepath)
    if code:
        # Write to a new file
        print("✅Extracted code")
        execute_code(code)