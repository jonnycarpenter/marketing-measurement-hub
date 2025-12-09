"""
Main entry point for the Marketing Measurement Multi-Agent System
Run this file to interact with the agents via command line
"""

import asyncio
import os
from typing import Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

from agents import measurement_lead


async def run_agent_conversation():
    """Run an interactive conversation with the measurement lead agent"""
    
    # Check for API key
    if not os.environ.get('GOOGLE_API_KEY'):
        print("=" * 60)
        print("WARNING: GOOGLE_API_KEY environment variable not set!")
        print("Please set it before running:")
        print("  set GOOGLE_API_KEY=your_api_key_here")
        print("=" * 60)
        return
    
    # Initialize session service
    session_service = InMemorySessionService()
    
    # Create a session
    session_id = "cli_session_001"
    user_id = "marketing_user"
    
    # Use the agent's module name as app_name for consistency
    app_name = "agents"
    
    session = session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    
    # Initialize runner with the measurement lead agent
    runner = Runner(
        agent=measurement_lead,
        app_name=app_name,
        session_service=session_service
    )
    
    print("=" * 60)
    print("Marketing Measurement Multi-Agent System")
    print("=" * 60)
    print("\nWelcome! I'm your Marketing Measurement Lead Agent.")
    print("I can help you with:")
    print("  - Designing test and control marketing experiments")
    print("  - Analyzing results using Causal Impact methodology")
    print("  - Managing your test portfolio")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("Type 'help' for example queries.")
    print("=" * 60)
    
    while True:
        # Get user input
        try:
            user_input = input("\n📊 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using Marketing Measurement! Goodbye!")
            break
        
        if user_input.lower() == 'help':
            print_help()
            continue
        
        # Create content for the agent
        content = types.Content(
            role="user",
            parts=[types.Part(text=user_input)]
        )
        
        print("\n🤖 Agent: ", end="", flush=True)
        
        try:
            # Run the agent and stream response
            response_text = ""
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content
            ):
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                print(part.text, end="", flush=True)
                                response_text += part.text
            
            if not response_text:
                print("(Processing completed)")
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again or type 'help' for assistance.")


def print_help():
    """Print help information"""
    print("""
╔════════════════════════════════════════════════════════════╗
║                    EXAMPLE QUERIES                         ║
╠════════════════════════════════════════════════════════════╣
║ TEST DESIGN                                                ║
║  • "Show me all existing tests"                            ║
║  • "What DMAs are available for testing?"                  ║
║  • "Create a new DMA test for our holiday campaign"        ║
║  • "Design a customer-level test for email optimization"   ║
║  • "Check for promo conflicts in December"                 ║
║                                                            ║
║ MEASUREMENT                                                ║
║  • "Analyze the results of test TEST_001"                  ║
║  • "Run causal impact analysis for my completed tests"     ║
║  • "What was the lift from our Q3 campaign?"               ║
║  • "Compare results across all DMA tests"                  ║
║                                                            ║
║ DATA & INSIGHTS                                            ║
║  • "Show me the promotional calendar"                      ║
║  • "How many customers do we have by segment?"             ║
║  • "What products are available for testing?"              ║
║  • "Give me recommendations for my next test"              ║
╚════════════════════════════════════════════════════════════╝
""")


async def run_single_query(query: str) -> str:
    """Run a single query through the agent system
    
    Args:
        query: The user's query
        
    Returns:
        The agent's response
    """
    # Initialize session service
    session_service = InMemorySessionService()
    
    session_id = "single_query_session"
    user_id = "api_user"
    app_name = "agents"
    
    session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    
    runner = Runner(
        agent=measurement_lead,
        app_name=app_name,
        session_service=session_service
    )
    
    content = types.Content(
        role="user",
        parts=[types.Part(text=query)]
    )
    
    response_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content
    ):
        if hasattr(event, 'content') and event.content:
            if hasattr(event.content, 'parts'):
                for part in event.content.parts:
                    if hasattr(part, 'text'):
                        response_text += part.text
    
    return response_text


def run_streamlit():
    """Launch the Streamlit application"""
    import subprocess
    import sys
    
    app_path = os.path.join(os.path.dirname(__file__), 'app.py')
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', app_path])


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--streamlit':
            run_streamlit()
        elif sys.argv[1] == '--query':
            if len(sys.argv) > 2:
                query = ' '.join(sys.argv[2:])
                result = asyncio.run(run_single_query(query))
                print(result)
            else:
                print("Usage: python main.py --query <your question>")
        elif sys.argv[1] == '--help':
            print("""
Marketing Measurement Multi-Agent System

Usage:
    python main.py              # Interactive CLI mode
    python main.py --streamlit  # Launch Streamlit web app
    python main.py --query "your question here"  # Single query mode
    python main.py --help       # Show this help
""")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Default: run interactive conversation
        asyncio.run(run_agent_conversation())
