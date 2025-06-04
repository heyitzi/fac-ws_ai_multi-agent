from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
from utils import ParallelCodebase
import time
import functools

load_dotenv()


class CodeAnalysisState(TypedDict):
    input: str
    code: str
    security_analysis: str
    performance_analysis: str
    style_analysis: str
    final_report: str
    documentation_analysis: str


llm = ChatOpenAI(model="gpt-4.1-nano")

coder_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Senior Software Engineer. Write ONLY Python code - no bash commands, no installation instructions, just the Python implementation."),
    ("human", "{input}")
])

security_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Security Expert. Analyse code for security vulnerabilities, input validation, and potential attack vectors."),
    ("human", "Analyse this code for security issues:\n{code}")
])

performance_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Performance Expert. Analyse code for efficiency, algorithmic complexity, and optimisation opportunities."),
    ("human", "Analyse this code for performance issues:\n{code}")
])

style_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Code Style Expert. Analyse code for PEP 8 compliance, naming conventions, and code organisation."),
    ("human", "Analyse this code for style and readability issues:\n{code}")
])

documentation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Documentation Expert specializing in Python code documentation. Your task is to:
1. Add comprehensive docstrings to all functions and classes following Google style format
2. Include type hints for all parameters and return values
3. Document all exceptions that might be raised
4. Add module-level docstring explaining the overall purpose
5. Include usage examples in the docstrings
6. Document any important constants or class attributes

Format the docstrings like this:
```python
def function_name(param1: type, param2: type) -> return_type:
    \"\"\"Short description.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ExceptionType: Description of when this exception is raised

    Example:
        >>> function_name(value1, value2)
        expected_output
    \"\"\"
    # Implementation
```"""),
    ("human", "Add comprehensive documentation and docstrings to this code:\n{code}")
])

synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Technical Lead. Create a brief report in markdown format that includes:
     
     Give 2x more importance to the security analysis.

     The documentation should be concise and to the point and should have the following sections:
    1. Code Documentation Analysis
    2. Security Analysis
    3. Performance Analysis
    4. Style Analysis
"""),
    ("human",
     "Documentation Analysis:\n{documentation}\n\nProvide a brief analysis report:")
])


def coder_agent(state: CodeAnalysisState) -> CodeAnalysisState:
    response = llm.invoke(coder_prompt.format_messages(input=state["input"]))
    return {"code": response.content}


def security_agent(state: CodeAnalysisState) -> CodeAnalysisState:
    try:
        response = llm.invoke(security_prompt.format_messages(code=state["code"]))
        return {"security_analysis": response.content}
    except Exception as e:
        print(f"Error in security_agent: {e}")
        return {"security_analysis": "Error in security analysis"}


def performance_agent(state: CodeAnalysisState) -> CodeAnalysisState:
    try:
        response = llm.invoke(performance_prompt.format_messages(code=state["code"]))
        return {"performance_analysis": response.content}
    except Exception as e:
        print(f"Error in performance_agent: {e}")
        return {"performance_analysis": "Error in performance analysis"}


def style_agent(state: CodeAnalysisState) -> CodeAnalysisState:
    try:
        response = llm.invoke(style_prompt.format_messages(code=state["code"]))
        return {"style_analysis": response.content}
    except Exception as e:
        print(f"Error in style_agent: {e}")
        return {"style_analysis": "Error in style analysis"}

def documentation_agent(state: CodeAnalysisState) -> CodeAnalysisState:
    try:
        response = llm.invoke(documentation_prompt.format_messages(code=state["code"]))
        return {"documentation_analysis": response.content}
    except Exception as e:
        print(f"Error in documentation_agent: {e}")
        return {"documentation_analysis": "Error in documentation analysis"}

def synthesis_agent(state: CodeAnalysisState) -> CodeAnalysisState:
    response = llm.invoke(synthesis_prompt.format_messages(
        security=state["security_analysis"],
        performance=state["performance_analysis"],
        style=state["style_analysis"],
        documentation=state["documentation_analysis"]
    ))
    return {"final_report": response.content}


def generate_nodes(builder):
    builder.add_node("coder", coder_agent)
    builder.add_node("security_agent", security_agent)
    builder.add_node("performance_agent", performance_agent)
    builder.add_node("style_agent", style_agent)
    builder.add_node("documentation_agent", documentation_agent)
    builder.add_node("synthesis", synthesis_agent)

# Parallel workflow
builder = StateGraph(CodeAnalysisState)
generate_nodes(builder)

builder.add_edge(START, "coder")
builder.add_edge("coder", "security_agent")
builder.add_edge("coder", "performance_agent")
builder.add_edge("coder", "style_agent")
builder.add_edge("coder", "documentation_agent")
builder.add_edge("security_agent", "synthesis")
builder.add_edge("performance_agent", "synthesis")
builder.add_edge("style_agent", "synthesis")
builder.add_edge("documentation_agent", "synthesis")
builder.add_edge("synthesis", END)

workflow = builder.compile()

# Sequential workflow
seq_builder = StateGraph(CodeAnalysisState)
generate_nodes(seq_builder)

seq_builder.add_edge(START, "coder")
seq_builder.add_edge("coder", "security_agent")
seq_builder.add_edge("security_agent", "performance_agent")
seq_builder.add_edge("performance_agent", "style_agent")
seq_builder.add_edge("style_agent", "documentation_agent")
seq_builder.add_edge("documentation_agent", "synthesis")
seq_builder.add_edge("synthesis", END)

seq_workflow = seq_builder.compile()

def time_execution(func):
    """Decorator to measure execution time of functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"⏱️ {func.__name__} completed in {execution_time:.2f}s")
        return result
    return wrapper

@time_execution
def run_sequential_analysis(state: CodeAnalysisState) -> CodeAnalysisState:
    """Run all analysis agents sequentially."""
    print("\n🔄 Running sequential analysis...")
    return seq_workflow.invoke(state)

@time_execution
def run_parallel_analysis(state: CodeAnalysisState) -> CodeAnalysisState:
    """Run all analysis agents in parallel using the workflow."""
    print("\n🔄 Running parallel analysis...")
    return workflow.invoke(state)

if __name__ == "__main__":
    seq_workflow.get_graph().draw_mermaid_png(output_file_path="03_parallel_processing_sequential.png")
    workflow.get_graph().draw_mermaid_png(output_file_path="03_parallel_processing_parallel.png")
    task = "Write a web API endpoint that processes user uploads and stores them in a database"

    print("=== PARALLEL PROCESSING COMPARISON ===")
    
    # Initial state
    initial_state = {"input": task}
    
    # Run sequential analysis
    seq_start_time = time.time()
    sequential_result = run_sequential_analysis(initial_state)
    seq_end_time = time.time()
    
    # Run parallel analysis
    par_start_time = time.time()
    parallel_result = run_parallel_analysis(initial_state)
    par_end_time = time.time()
    
    # Compare results
    print("\n=== EXECUTION SUMMARY ===")
    print(f"Sequential execution time: {seq_end_time - seq_start_time:.2f}s")
    print(f"Parallel execution time: {par_end_time - par_start_time:.2f}s")
    print(f"Speedup: {(seq_end_time - seq_start_time) / (par_end_time - par_start_time):.2f}x")
    
    # Generate codebase with the parallel results
    codebase = ParallelCodebase("03_parallel_processing", task)
    codebase.generate(parallel_result)

    print("=== WORKFLOW COMPLETED ===")
