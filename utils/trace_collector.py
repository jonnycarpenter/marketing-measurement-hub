"""
OpenTelemetry Trace Collector for Streamlit Sidebar
Captures ADK agent spans and displays them in a subtle, always-on live feed.
Designed to be non-distracting but informative.
"""
import streamlit as st
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SimpleSpanProcessor
from typing import List, Optional
from datetime import datetime
import random
import time


class StreamlitSpanCollector(SpanExporter):
    """A custom OTel exporter that collects spans into Streamlit's session state."""
    
    def __init__(self, key: str = "adk_trace_spans"):
        self.session_state_key = key
        if self.session_state_key not in st.session_state:
            st.session_state[self.session_state_key] = []

    def export(self, spans) -> None:
        """Called when a batch of spans is ready to be exported."""
        for span in spans:
            try:
                span_data = {
                    "name": span.name,
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "duration_ms": (span.end_time - span.start_time) / 1_000_000 if span.end_time and span.start_time else None,
                    "status": span.status.status_code.name if hasattr(span.status, 'status_code') else "UNKNOWN",
                    "attributes": dict(span.attributes) if span.attributes else {},
                    "events": [{"name": e.name, "attributes": dict(e.attributes) if e.attributes else {}} for e in span.events] if span.events else [],
                    "parent_id": str(span.parent.span_id) if span.parent else None,
                    "span_id": str(span.context.span_id) if span.context else None,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state[self.session_state_key].append(span_data)
            except Exception as e:
                # Silently handle any span processing errors
                pass
    
    def shutdown(self) -> None:
        """Clear the stored spans on shutdown."""
        if self.session_state_key in st.session_state:
            st.session_state[self.session_state_key] = []

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush - no-op for this collector."""
        return True


_otel_initialized = False


def initialize_otel_tracer(span_collector_key: str = "adk_trace_spans"):
    """Initializes the OTel TracerProvider and custom exporter once."""
    global _otel_initialized
    
    if _otel_initialized:
        return
    
    try:
        # Create the custom collector/exporter
        collector = StreamlitSpanCollector(key=span_collector_key)
        
        # Configure the Tracer Provider
        provider = TracerProvider()
        
        # Add the SimpleSpanProcessor with our custom collector
        provider.add_span_processor(SimpleSpanProcessor(collector))
        
        # Set the global tracer provider
        trace.set_tracer_provider(provider)
        
        _otel_initialized = True
    except Exception as e:
        # If OTel setup fails, continue without tracing
        pass


def clear_trace_data(span_collector_key: str = "adk_trace_spans"):
    """Clears the trace data before a new agent run."""
    if span_collector_key in st.session_state:
        st.session_state[span_collector_key] = []


def get_subtle_css() -> str:
    """Returns the CSS for subtle, low-contrast trace display."""
    return """
    <style>
    .otel-trace-container {
        background-color: #F5F5F5;
        border: 1px solid #E8E8E8;
        border-radius: 6px;
        padding: 10px 12px;
        margin: 12px 0;
        font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
    }
    
    .otel-trace-header {
        color: #9E9E9E;
        font-size: 9px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid #E0E0E0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .otel-status-indicator {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #BDBDBD;
    }
    
    .otel-status-indicator.active {
        background-color: #81C784;
        animation: otel-pulse 1.5s ease-in-out infinite;
    }
    
    .otel-trace-list {
        display: flex;
        flex-direction: column;
        gap: 4px;
        max-height: 180px;
        overflow-y: auto;
    }
    
    .otel-trace-list::-webkit-scrollbar {
        width: 3px;
    }
    
    .otel-trace-list::-webkit-scrollbar-track {
        background: transparent;
    }
    
    .otel-trace-list::-webkit-scrollbar-thumb {
        background-color: #E0E0E0;
        border-radius: 3px;
    }
    
    .otel-trace-item {
        display: flex;
        align-items: center;
        font-size: 10px;
        color: #9E9E9E;
        padding: 3px 0;
        line-height: 1.3;
    }
    
    .otel-trace-item:last-child {
        color: #757575;
    }
    
    .otel-trace-dot {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        margin-right: 8px;
        flex-shrink: 0;
        background-color: #E0E0E0;
    }
    
    .otel-trace-dot.completed {
        background-color: #A5D6A7;
    }
    
    .otel-trace-dot.active {
        background-color: #81C784;
        animation: otel-pulse 1.5s ease-in-out infinite;
    }
    
    .otel-trace-dot.error {
        background-color: #EF9A9A;
    }
    
    .otel-trace-dot.idle {
        background-color: #BDBDBD;
    }
    
    .otel-trace-content {
        flex-grow: 1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-weight: 400;
    }
    
    .otel-trace-time {
        font-size: 9px;
        color: #BDBDBD;
        margin-left: 6px;
        flex-shrink: 0;
    }
    
    .otel-empty-state {
        color: #BDBDBD;
        font-size: 10px;
        text-align: center;
        padding: 8px 0;
        font-style: italic;
    }
    
    @keyframes otel-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    </style>
    """


def render_trace_sidebar(span_collector_key: str = "adk_trace_spans"):
    """
    Renders the captured spans with subtle visualization in the Streamlit sidebar.
    Always visible regardless of which tab is active - provides live agent activity feed.
    """
    
    spans = st.session_state.get(span_collector_key, [])
    is_processing = st.session_state.get("agent_processing", False)
    
    # Use native Streamlit components for reliable rendering
    with st.sidebar:
        # Container styling via markdown
        st.markdown("""
        <style>
        .activity-header {
            color: #9E9E9E;
            font-size: 10px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        </style>
        <div class="activity-header">Agent Activity</div>
        """, unsafe_allow_html=True)
        
        # Create a container with light grey background
        activity_container = st.container()
        
        with activity_container:
            if not spans:
                if is_processing:
                    st.caption("⟳ Processing request...")
                else:
                    st.caption("_Awaiting agent activity_")
            else:
                # Show last 8 items for better context
                recent_spans = spans[-8:]
                
                for i, span in enumerate(recent_spans):
                    name = span.get('name') or 'Unknown'
                    status = span.get('status') or 'UNKNOWN'
                    duration = span.get('duration_ms')
                    
                    # Determine status indicator
                    if status == "OK":
                        indicator = "✓"
                        color = "#A5D6A7"
                    elif status == "ERROR":
                        indicator = "✗"
                        color = "#EF9A9A"
                    elif i == len(recent_spans) - 1 and is_processing:
                        indicator = "●"
                        color = "#81C784"
                    else:
                        indicator = "○"
                        color = "#BDBDBD"
                    
                    # Clean up the name for display
                    display_name = str(name).replace("_", " ").title()
                    if len(display_name) > 25:
                        display_name = display_name[:22] + "..."
                        
                    # Format duration
                    time_str = ""
                    if duration:
                        if duration < 1000:
                            time_str = f" · {duration:.0f}ms"
                        else:
                            time_str = f" · {duration/1000:.1f}s"
                    
                    # Render each trace item
                    st.markdown(
                        f'<span style="color:{color};font-size:10px;">{indicator}</span> '
                        f'<span style="color:#9E9E9E;font-size:11px;">{display_name}</span>'
                        f'<span style="color:#BDBDBD;font-size:9px;">{time_str}</span>',
                        unsafe_allow_html=True
                    )


def add_manual_trace(name: str, status: str = "OK", duration_ms: float = 0, 
                     attributes: dict = None, span_collector_key: str = "adk_trace_spans"):
    """Manually add a trace entry (useful when OTel isn't capturing ADK spans)."""
    if span_collector_key not in st.session_state:
        st.session_state[span_collector_key] = []
    
    span_data = {
        "name": name,
        "start_time": None,
        "end_time": None,
        "duration_ms": duration_ms,
        "status": status,
        "attributes": attributes or {},
        "events": [],
        "parent_id": None,
        "span_id": str(hash(f"{name}{datetime.now().isoformat()}")),
        "timestamp": datetime.now().isoformat()
    }
    st.session_state[span_collector_key].append(span_data)
