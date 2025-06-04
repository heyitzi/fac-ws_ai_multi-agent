from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
from utils import SequentialCodebase
import json
import os
from datetime import datetime

load_dotenv()

def save_state(state: dict, node_name: str) -> None:
    """Save the current state to a JSON file for debugging."""

    debug_dir = "debug_states"
    os.makedirs(debug_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{debug_dir}/state_{node_name}_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    
    print(f"Debug state saved: {filename}")

class CodeReviewState(TypedDict):
    input: str
    code: str
    review: str
    refactored_code: str
    unit_tests: str


llm = ChatOpenAI(model="gpt-4.1-nano")

coder_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Security-Focused Software Engineer. Write secure Python code that follows security best practices, handles input validation, and protects against common vulnerabilities like injection attacks, XSS, and data exposure."),
    ("human", "{input}")
])

reviewer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Security Code Reviewer. Focus on identifying security vulnerabilities, improper input validation, unsafe data handling, and potential attack vectors. Provide specific security-focused feedback."),
    ("human", "Review this code for security issues:\n{code}")
])

refactorer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Security Refactoring Expert. Implement security improvements while maintaining functionality. Focus on fixing vulnerabilities, adding proper input validation, and implementing secure coding practices."),
    ("human",
     "Original code:\n{code}\n\nSecurity review feedback:\n{review}\n\nRefactor to address security concerns:")
])

tester_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Security Tester. Write comprehensive security-focused unit tests that verify proper input validation, error handling, and protection against common attack vectors. Include tests for edge cases and malicious inputs."),
    ("human", "Write security tests for this code:\n{refactored_code}")
])


def coder_agent(state: CodeReviewState) -> CodeReviewState:
    response = llm.invoke(coder_prompt.format_messages(input=state["input"]))
    new_state = {"code": response.content}
    save_state(new_state, "coder")
    return new_state


def reviewer_agent(state: CodeReviewState) -> CodeReviewState:
    response = llm.invoke(reviewer_prompt.format_messages(code=state["code"]))
    new_state = {"review": response.content}
    save_state(new_state, "reviewer")
    return new_state


def refactorer_agent(state: CodeReviewState) -> CodeReviewState:
    response = llm.invoke(refactorer_prompt.format_messages(
        code=state["code"], review=state["review"]))
    new_state = {"refactored_code": response.content}
    save_state(new_state, "refactorer")
    return new_state

def tester_agent(state: CodeReviewState) -> CodeReviewState:
    response = llm.invoke(tester_prompt.format_messages(refactored_code=state["refactored_code"]))
    new_state = {"unit_tests": response.content}
    save_state(new_state, "tester")
    return new_state


builder = StateGraph(CodeReviewState)
builder.add_node("coder", coder_agent)
builder.add_node("reviewer", reviewer_agent)
builder.add_node("refactorer", refactorer_agent)
builder.add_node("tester", tester_agent)

builder.add_edge(START, "coder")
builder.add_edge("coder", "reviewer")
builder.add_edge("reviewer", "refactorer")
builder.add_edge("refactorer", "tester")
builder.add_edge("tester", END)

workflow = builder.compile()

if __name__ == "__main__":
    task = "Write a function that validates email addresses using regex"

    print("Running sequential workflow...")
    result = workflow.invoke({"input": task})

    codebase = SequentialCodebase("01_sequential_workflow", task)
    codebase.generate(result)

    print("=== WORKFLOW COMPLETED ===")
