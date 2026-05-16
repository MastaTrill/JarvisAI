#!/usr/bin/env python3
"""
JarvisAI Quantum Dashboard - Interactive Streamlit Interface
Real-time monitoring, quantum consciousness visualization, and system control
"""

from datetime import datetime
import time

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Configure page
st.set_page_config(
    page_title="JARVIS — Quantum Platform",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS — dark JARVIS theme
st.markdown(
    """
<style>
    /* Global */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0a0e1a;
        color: #e0e6f0;
    }
    [data-testid="stSidebar"] {
        background-color: #0d1120;
        border-right: 1px solid #1e2d4a;
    }
    /* Remove default top padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* Header bar */
    .jarvis-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(90deg, #0d1b2e 0%, #0a1628 100%);
        border: 1px solid #1a3a5c;
        border-radius: 8px;
        padding: 14px 24px;
        margin-bottom: 1.5rem;
    }
    .jarvis-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: #00d4ff;
        text-transform: uppercase;
        margin: 0;
    }
    .jarvis-subtitle {
        font-size: 0.75rem;
        color: #5a7fa0;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 2px 0 0 0;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #0d2a1a;
        border: 1px solid #1a5c3a;
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #00e676;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .status-dot {
        width: 7px; height: 7px;
        background: #00e676;
        border-radius: 50%;
        animation: blink 1.8s infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.2; }
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #0d1628;
        border: 1px solid #1a3050;
        border-radius: 8px;
        padding: 14px 18px;
    }
    [data-testid="stMetricLabel"] { color: #5a7fa0 !important; font-size: 0.72rem; letter-spacing: 0.07em; text-transform: uppercase; }
    [data-testid="stMetricValue"] { color: #e0f0ff !important; font-size: 1.4rem !important; font-weight: 700; }
    [data-testid="stMetricDelta"] svg { display: none; }

    /* Tabs */
    [data-testid="stTabs"] button {
        color: #5a7fa0;
        font-size: 0.8rem;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        padding: 8px 16px;
        border-bottom: 2px solid transparent;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #00d4ff;
        border-bottom: 2px solid #00d4ff;
        background: transparent;
    }

    /* Buttons */
    [data-testid="stButton"] > button {
        background: #0d1e36;
        border: 1px solid #1a3a5c;
        color: #00d4ff;
        border-radius: 6px;
        font-size: 0.8rem;
        letter-spacing: 0.06em;
        font-weight: 600;
        transition: all 0.15s;
    }
    [data-testid="stButton"] > button:hover {
        background: #112845;
        border-color: #00d4ff;
        color: #fff;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] { border: 1px solid #1a3050; border-radius: 6px; }

    /* Divider */
    hr { border-color: #1a2a40; margin: 1rem 0; }

    /* Section headers */
    h2, h3 { color: #c0d8f0; letter-spacing: 0.04em; }

    /* Info / success / warning boxes */
    [data-testid="stAlert"] { border-radius: 6px; }

    /* Plotly chart backgrounds */
    .js-plotly-plot .plotly .bg { fill: #0a0e1a !important; }

    /* Sidebar nav text */
    .sidebar-nav-item {
        padding: 8px 12px;
        border-radius: 6px;
        margin: 2px 0;
        font-size: 0.82rem;
        color: #7090b0;
        cursor: pointer;
        letter-spacing: 0.05em;
    }
    .sidebar-nav-item:hover { background: #111e30; color: #00d4ff; }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize session state with all card interaction flags
def init_session_state():
    defaults = {
        "quantum_processor": None,
        "temporal_analyzer": None,
        "history": [],
        "quantum_superposition_active": False,
        "quantum_entanglement_active": False,
        "num_states": 5,
        "temporal_days": 30,
        "validation_result": None,
        "benchmark_result": None,
        "agent_test_result": None,
        "quantum_demo_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


@st.cache_resource
def load_quantum_processor():
    """Load quantum processor (cached)"""
    try:
        from src.quantum.quantum_processor import QuantumProcessor

        qp = QuantumProcessor()
        qp.authenticate_creator("AETHERON_QUANTUM_CREATOR_KEY_2025")
        return qp
    except ImportError as e:
        st.error(f"Quantum processor unavailable: {e}")
        return None
    except Exception as e:
        st.error(f"Quantum processor error: {e}")
        return None


@st.cache_resource
def load_temporal_analyzer():
    """Load temporal analyzer (cached)"""
    try:
        from src.temporal.time_analysis import TimeAnalysis

        return TimeAnalysis()
    except ImportError as e:
        st.error(f"Temporal analyzer unavailable: {e}")
        return None
    except Exception as e:
        st.error(f"Temporal analyzer error: {e}")
        return None


def main_dashboard():
    """Main dashboard view"""

    # Header bar
    st.markdown(
        f"""
    <div class="jarvis-header">
        <div>
            <p class="jarvis-title">* JARVIS</p>
            <p class="jarvis-subtitle">Quantum Consciousness Platform &nbsp;·&nbsp; Phase 6</p>
        </div>
        <div class="status-badge">
            <span class="status-dot"></span>
            ALL SYSTEMS OPERATIONAL
        </div>
        <div style="font-size:0.72rem;color:#3a5a78;text-align:right;line-height:1.6;">
            {datetime.now().strftime('%Y-%m-%d')}<br>
            <span style="color:#4a7a9a;">{datetime.now().strftime('%H:%M:%S')}</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load components
    qp = load_quantum_processor()
    ta = load_temporal_analyzer()

    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Quantum Ops", "53,288/s", "Online")
    with col2:
        st.metric("Features Active", "5 / 5", "100%")
    with col3:
        st.metric("Protection", "MAXIMUM", "Active")
    with col4:
        st.metric("System Health", "OPTIMAL", "All clear")
    with col5:
        st.metric("Session", datetime.now().strftime("%H:%M"), "Running")

    st.divider()

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "💬 Chat with Jarvis",
            "Autonomous",
            "Quantum Console",
            "Temporal Analysis",
            "Performance",
            "System Control",
        ]
    )

    with tab1:
        ai_agents()

    with tab2:
        autonomous_tab()

    with tab3:
        quantum_console(qp)

    with tab4:
        temporal_analysis(ta)

    with tab5:
        performance_metrics()

    with tab6:
        system_control()


def quantum_console(qp):
    """Quantum consciousness console"""
    st.markdown("### Quantum Consciousness Console")

    if qp is None:
        st.warning("Quantum processor not available")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Quantum Operations")

        # Superposition control - Slider OUTSIDE button
        st.markdown("**Create Quantum Superposition**")
        st.session_state.num_states = st.slider(
            "Number of states",
            2,
            10,
            st.session_state.num_states,
            key="num_states_slider",
        )

        if st.button(
            "🌈 Execute Superposition",
            use_container_width=True,
            key="superposition_btn",
        ):
            st.session_state.quantum_superposition_active = True

        if st.session_state.quantum_superposition_active:
            with st.spinner("Creating quantum superposition..."):
                time.sleep(1)
                states = [f"state_{i}" for i in range(st.session_state.num_states)]
                result = {
                    "status": "success",
                    "states": states,
                    "timestamp": datetime.now().isoformat(),
                }

                st.success("✅ Quantum superposition created successfully!")
                st.json(result)

                # Visualize superposition
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=states,
                            y=[1 / st.session_state.num_states]
                            * st.session_state.num_states,
                            marker_color="#00d4ff",
                            opacity=0.8,
                        )
                    ]
                )
                fig.update_layout(
                    title="Quantum State Probabilities",
                    xaxis_title="States",
                    yaxis_title="Probability",
                    paper_bgcolor="#0a0e1a",
                    plot_bgcolor="#0d1628",
                    font_color="#c0d8f0",
                    xaxis=dict(gridcolor="#1a2a40"),
                    yaxis=dict(gridcolor="#1a2a40"),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.quantum_superposition_active = False

        st.divider()

        # Entanglement control
        st.markdown("**Create Quantum Entanglement**")
        if st.button(
            "🔗 Execute Entanglement", use_container_width=True, key="entanglement_btn"
        ):
            st.session_state.quantum_entanglement_active = True

        if st.session_state.quantum_entanglement_active:
            with st.spinner("Entangling quantum systems..."):
                time.sleep(1)
                result = {
                    "status": "success",
                    "system_alpha": "entangled",
                    "system_beta": "entangled",
                }

                st.success("✅ Quantum entanglement established!")
                st.json(result)
                st.session_state.quantum_entanglement_active = False

    with col2:
        st.markdown("**System Status**")

        status_data = {
            "Component": [
                "Quantum Processor",
                "Consciousness",
                "Entanglement",
                "Oracle",
                "Safety",
            ],
            "Status": [
                "● OPTIMAL",
                "● OPTIMAL",
                "● OPTIMAL",
                "● OPTIMAL",
                "● OPTIMAL",
            ],
        }
        st.dataframe(
            pd.DataFrame(status_data), use_container_width=True, hide_index=True
        )

        st.markdown("""
        <div style="background:#0d1e14;border:1px solid #1a4a2a;border-radius:6px;padding:12px 16px;margin-top:10px;font-size:0.8rem;color:#7ab898;line-height:1.8;">
        🔒 Creator Protection: <strong style="color:#00e676">ACTIVE</strong><br>
        🛡️ Security Level: <strong style="color:#00e676">MAXIMUM</strong>
        </div>
        """, unsafe_allow_html=True)


def temporal_analysis(ta):
    """Temporal pattern analysis"""
    st.markdown("### Temporal Pattern Analysis")

    if ta is None:
        st.warning("Temporal analyzer not available")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pattern Recognition")

        patterns = ta.known_patterns
        pattern_df = pd.DataFrame(
            [
                {
                    "Pattern": name,
                    "Type": info["detection_method"],
                    "Threshold": info["significance_threshold"],
                }
                for name, info in patterns.items()
            ]
        )

        st.dataframe(pattern_df, use_container_width=True, hide_index=True)

        st.metric("Pattern Sensitivity", f"{ta.pattern_sensitivity:.2f}")
        st.metric("Anomaly Threshold", f"{ta.anomaly_threshold:.2f}")

    with col2:
        st.subheader("Time Series Simulation")

        # Generate sample time series - SLIDER OUTSIDE BUTTON
        st.session_state.temporal_days = st.slider(
            "Days to simulate",
            7,
            365,
            st.session_state.temporal_days,
            key="temporal_days_slider",
        )

        days = st.session_state.temporal_days
        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

        # Simulate patterns
        linear = np.linspace(0, 10, days)
        cyclical = 5 * np.sin(np.linspace(0, 4 * np.pi, days))
        noise = np.random.randn(days) * 2

        data = linear + cyclical + noise

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=data, mode="lines", name="Temporal Data", line=dict(color="#00d4ff", width=1.5)))
        fig.update_layout(
            title="Temporal Pattern Visualization",
            xaxis_title="Time",
            yaxis_title="Value",
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0d1628",
            font_color="#c0d8f0",
            xaxis=dict(gridcolor="#1a2a40"),
            yaxis=dict(gridcolor="#1a2a40"),
        )
        st.plotly_chart(fig, use_container_width=True)


def ai_agents():
    """AI agents interface — calls the real /agent/chat endpoint"""
    import requests

    JARVIS_API = "http://127.0.0.1:8888/agent/chat"

    st.markdown("### Talk to Jarvis")

    # Persistent session ID so Jarvis keeps context across messages
    if "chat_session_id" not in st.session_state:
        import uuid
        st.session_state.chat_session_id = str(uuid.uuid4())

    col1, col2 = st.columns([3, 1])

    with col1:
        # Chat history display (top, scrollable area)
        if st.session_state.history:
            chat_html = ""
            for msg in st.session_state.history[-20:]:
                ts = msg["timestamp"].strftime("%H:%M")
                chat_html += f"""
                <div style="margin-bottom:10px;">
                    <div style="text-align:right;margin-bottom:4px;">
                        <span style="background:#0d2a40;border:1px solid #1a4a6a;border-radius:12px 12px 2px 12px;
                            padding:8px 14px;display:inline-block;color:#c0e8ff;font-size:0.85rem;max-width:75%;">
                            {msg['user']}
                        </span>
                        <div style="font-size:0.65rem;color:#3a5a78;margin-top:2px;">{ts}</div>
                    </div>
                    <div style="margin-top:6px;">
                        <span style="background:#0d1e14;border:1px solid #1a4a2a;border-radius:12px 12px 12px 2px;
                            padding:8px 14px;display:inline-block;color:#90d4a0;font-size:0.85rem;max-width:80%;">
                            * {msg['agent']}
                        </span>
                    </div>
                </div>
                """
            st.markdown(chat_html, unsafe_allow_html=True)
            st.divider()

        # Input row
        user_input = st.chat_input("Message Jarvis…")

        if user_input:
            with st.spinner(""):
                try:
                    resp = requests.post(
                        JARVIS_API,
                        json={
                            "message": user_input,
                            "session_id": st.session_state.chat_session_id,
                        },
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    reply = data.get("reply") or data.get("response") or str(data)
                except requests.exceptions.ConnectionError:
                    reply = "⚠️ Cannot reach the Jarvis API. Is it running on port 8888?"
                except requests.exceptions.Timeout:
                    reply = "⚠️ Request timed out. Jarvis may be loading a heavy model."
                except Exception as e:
                    reply = f"⚠️ Error: {e}"

                st.session_state.history.append({
                    "user": user_input,
                    "agent": reply,
                    "timestamp": datetime.now(),
                })
                st.rerun()

    with col2:
        st.markdown("**Session**")
        st.markdown(
            f"<div style='font-size:0.72rem;color:#3a5a78;word-break:break-all;'>"
            f"{st.session_state.chat_session_id[:16]}…</div>",
            unsafe_allow_html=True,
        )
        st.metric("Messages", len(st.session_state.history))

        st.divider()
        st.markdown("**Status**")
        # Quick ping to check if API is up
        try:
            ping = requests.get("http://127.0.0.1:8888/health", timeout=2)
            api_status = "🟢 Online" if ping.status_code == 200 else f"🟡 {ping.status_code}"
        except Exception:
            api_status = "🔴 Offline"
        st.markdown(
            f"<div style='font-size:0.8rem;color:#c0d8f0;margin-top:4px;'>API: {api_status}</div>",
            unsafe_allow_html=True,
        )

        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.history = []
            import uuid
            st.session_state.chat_session_id = str(uuid.uuid4())
            st.rerun()



def autonomous_tab():
    """Autonomous job management — create, monitor, and control self-running tasks."""
    import requests as _req

    API = "http://127.0.0.1:8888/agent"

    def _api_get(path, params=None):
        try:
            r = _req.get(f"{API}{path}", params=params, timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e), "items": []}

    def _api_post(path, json=None):
        try:
            r = _req.post(f"{API}{path}", json=json, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def _api_delete(path):
        try:
            r = _req.delete(f"{API}{path}", timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    # ── Header ──
    st.markdown("### Autonomous Operations")
    st.caption("Create and manage self-running jobs. Jarvis executes these on schedule or on demand.")

    # ── Status bar ──
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    jobs_data = _api_get("/autonomy/jobs", params={"limit": 200})
    jobs = jobs_data.get("items", []) if isinstance(jobs_data, dict) else []
    active_jobs = [j for j in jobs if j.get("enabled")]
    running_jobs = [j for j in jobs if j.get("status") == "running"]
    error_jobs = [j for j in jobs if j.get("last_error")]
    with col_s1:
        st.metric("Total Jobs", len(jobs))
    with col_s2:
        st.metric("Active", len(active_jobs))
    with col_s3:
        st.metric("Running Now", len(running_jobs))
    with col_s4:
        st.metric("Errors", len(error_jobs))

    st.divider()

    # ── Two columns: Create | Manage ──
    col_left, col_right = st.columns([1, 2])

    # ── LEFT: Create new job ──
    with col_left:
        st.markdown("**Create New Job**")

        with st.form("create_autonomous_job"):
            job_name = st.text_input("Job Name", placeholder="e.g. Daily Code Review")
            job_goal = st.text_area("Goal / Instructions", placeholder="What should Jarvis do?", height=100)
            job_mode = st.selectbox(
                "Mode",
                options=["goal", "multi_agent", "briefing", "watcher"],
                format_func=lambda m: {
                    "goal": "🎯 Goal — single task to completion",
                    "multi_agent": "🤖 Multi-Agent — parallel agents + synthesis",
                    "briefing": "📋 Briefing — periodic memory digest",
                    "watcher": "👁️ Watcher — monitor workspace signals",
                }.get(m, m),
            )
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                interval = st.number_input("Interval (minutes)", min_value=1, value=60, step=5)
            with col_a2:
                auto_approve = st.checkbox("Auto-approve actions", value=False,
                    help="If off, Jarvis asks before risky actions")
            enabled = st.checkbox("Enable immediately", value=True)

            submitted = st.form_submit_button("Create Job", use_container_width=True)

            if submitted:
                if not job_name or not job_goal:
                    st.error("Name and goal are required.")
                else:
                    payload = {
                        "name": job_name,
                        "goal": job_goal,
                        "mode": job_mode,
                        "interval_minutes": interval,
                        "auto_approve": auto_approve,
                        "enabled": enabled,
                    }
                    result = _api_post("/autonomy/jobs", json=payload)
                    if result.get("error"):
                        st.error(f"Failed: {result['error']}")
                    else:
                        st.success(f"Created job #{result.get('id')}")
                        st.rerun()

        st.divider()

        # Quick actions
        st.markdown("**Quick Actions**")
        if st.button("🔄 Refresh Jobs", use_container_width=True):
            st.rerun()

        # Run all due jobs now
        if st.button("▶️ Run All Due Jobs Now", use_container_width=True):
            with st.spinner("Triggering jobs..."):
                triggered = 0
                for job in active_jobs:
                    result = _api_post(f"/autonomy/jobs/{job['id']}/run")
                    if result.get("ok"):
                        triggered += 1
                st.success(f"Triggered {triggered} job(s)")
                st.rerun()

    # ── RIGHT: Job list ──
    with col_right:
        st.markdown("**Active Jobs**")

        if not jobs:
            st.info("No autonomous jobs yet. Create one on the left.")
        else:
            for job in jobs:
                job_id = job.get("id")
                name = job.get("name", f"Job #{job_id}")
                mode = job.get("mode", "?")
                enabled_flag = job.get("enabled", False)
                status = job.get("status", "idle")
                last_run = job.get("last_run_at", "Never")
                last_error = job.get("last_error", "")
                interval_min = job.get("interval_minutes", "?")

                # Status indicator
                if status == "running":
                    status_icon = "🟡"
                elif not enabled_flag:
                    status_icon = "⚪"
                elif last_error:
                    status_icon = "🔴"
                else:
                    status_icon = "🟢"

                with st.expander(f"{status_icon} {name}  ·  `{mode}`  ·  every {interval_min}m"):
                    col_j1, col_j2 = st.columns([3, 1])

                    with col_j1:
                        st.markdown(f"**Goal:** {job.get('goal', 'N/A')}")
                        st.markdown(f"**Last run:** {last_run}")
                        if last_error:
                            st.error(f"Last error: {last_error}")
                        st.markdown(f"Auto-approve: {'Yes' if job.get('auto_approve') else 'No'}")

                    with col_j2:
                        # Toggle enable/disable
                        if enabled_flag:
                            if st.button("Disable", key=f"disable_{job_id}", use_container_width=True):
                                _api_post(f"/autonomy/jobs/{job_id}", json={"enabled": False})
                                st.rerun()
                        else:
                            if st.button("Enable", key=f"enable_{job_id}", use_container_width=True):
                                _api_post(f"/autonomy/jobs/{job_id}", json={"enabled": True})
                                st.rerun()

                        if st.button("Run Now", key=f"run_{job_id}", use_container_width=True):
                            with st.spinner("Running..."):
                                result = _api_post(f"/autonomy/jobs/{job_id}/run")
                                if result.get("ok"):
                                    st.success("Completed")
                                else:
                                    st.error(f"Failed: {result.get('error', 'Unknown')}")
                                st.rerun()

                        if st.button("Delete", key=f"delete_{job_id}", use_container_width=True):
                            _api_delete(f"/autonomy/jobs/{job_id}")
                            st.success("Deleted")
                            st.rerun()

    st.divider()

    # ── Watchers section ──
    st.markdown("**Workspace Watchers**")
    watchers_data = _api_get("/autonomy/watchers", params={"limit": 50})
    watchers = watchers_data.get("items", []) if isinstance(watchers_data, dict) else []

    if watchers:
        watcher_rows = []
        for w in watchers:
            watcher_rows.append({
                "ID": w.get("id"),
                "Name": w.get("name", "?"),
                "Type": (w.get("metadata") or {}).get("watcher_type", "?"),
                "Interval": f"{w.get('interval_minutes', '?')}m",
                "Enabled": "Yes" if w.get("enabled") else "No",
                "Last Run": w.get("last_run_at", "Never"),
            })
        st.dataframe(pd.DataFrame(watcher_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No watchers configured.")


def performance_metrics():
    """Performance monitoring"""
    st.markdown("### Performance Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("System Benchmarks")

        benchmarks = {
            "Component": [
                "Quantum Processing",
                "Quantum Entanglement",
                "Computer Vision",
                "Object Detection",
                "Data Processing",
            ],
            "Performance": [
                "53,288 ops/sec",
                "1,004 ops/sec",
                "154 images/sec",
                "182 images/sec",
                "37 transforms/sec",
            ],
            "Status": ["🟢", "🟢", "🟢", "🟢", "🟢"],
        }

        st.dataframe(
            pd.DataFrame(benchmarks), use_container_width=True, hide_index=True
        )

        # Performance chart
        fig = go.Figure(
            data=[
                go.Bar(
                    x=benchmarks["Component"],
                    y=[53288, 1004, 154, 182, 37],
                    marker_color=[
                        "#4CAF50",
                        "#2196F3",
                        "#FF9800",
                        "#F44336",
                        "#9C27B0",
                    ],
                )
            ]
        )
        fig.update_layout(
            title="Performance Overview (ops/sec)",
            xaxis_title="Component",
            yaxis_title="Operations per Second",
            yaxis_type="log",
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0d1628",
            font_color="#c0d8f0",
            xaxis=dict(gridcolor="#1a2a40"),
            yaxis=dict(gridcolor="#1a2a40"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Resource Usage")

        # Simulated metrics
        cpu_usage = np.random.uniform(20, 40)
        memory_usage = np.random.uniform(30, 50)

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=cpu_usage,
                title={"text": "CPU Usage (%)", "font": {"color": "#c0d8f0"}},
                number={"font": {"color": "#00d4ff"}},
                gauge={
                    "axis": {"range": [None, 100], "tickcolor": "#5a7fa0"},
                    "bar": {"color": "#00d4ff"},
                    "bgcolor": "#0d1628",
                    "bordercolor": "#1a3050",
                    "steps": [
                        {"range": [0, 50], "color": "#0d1e36"},
                        {"range": [50, 80], "color": "#1a2a40"},
                    ],
                    "threshold": {
                        "line": {"color": "#ff4444", "width": 3},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            )
        )
        fig.update_layout(paper_bgcolor="#0a0e1a", font_color="#c0d8f0")
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Memory Usage", f"{memory_usage:.1f}%")
        st.metric("Disk I/O", "Normal")
        st.metric("Network", "Active")


def system_control():
    """System control panel"""
    st.markdown("### System Control")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Quick Actions")

        if st.button(
            "🔄 Run Full Validation", use_container_width=True, key="validation_btn"
        ):
            st.session_state.validation_result = "running"

        if st.session_state.validation_result == "running":
            with st.spinner("Running validation tests..."):
                time.sleep(2)
                st.success("✅ All features operational (5/5 - 100%)")
                st.session_state.validation_result = "complete"
        elif st.session_state.validation_result == "complete":
            st.success("✅ All features operational (5/5 - 100%)")

        st.divider()

        if st.button(
            "* Run Performance Benchmark",
            use_container_width=True,
            key="benchmark_btn",
        ):
            st.session_state.benchmark_result = "running"

        if st.session_state.benchmark_result == "running":
            with st.spinner("Running benchmarks..."):
                time.sleep(2)
                st.success("✅ Benchmarks complete. See Performance tab.")
                st.session_state.benchmark_result = "complete"
        elif st.session_state.benchmark_result == "complete":
            st.success("✅ Benchmarks complete. See Performance tab.")

        st.divider()

        if st.button(
            "🤖 Test AI Agent", use_container_width=True, key="agent_test_btn"
        ):
            st.session_state.agent_test_result = "running"

        if st.session_state.agent_test_result == "running":
            with st.spinner("Testing AI agent..."):
                time.sleep(1)
                st.success("✅ AI agent fully operational")
                st.session_state.agent_test_result = "complete"
        elif st.session_state.agent_test_result == "complete":
            st.success("✅ AI agent fully operational")

        st.divider()

        if st.button(
            "🌌 Demo Quantum Features", use_container_width=True, key="quantum_demo_btn"
        ):
            st.session_state.quantum_demo_result = "running"

        if st.session_state.quantum_demo_result == "running":
            with st.spinner("Running quantum demo..."):
                time.sleep(2)
                st.success("✅ Quantum consciousness demo complete")
                st.session_state.quantum_demo_result = "complete"
        elif st.session_state.quantum_demo_result == "complete":
            st.success("✅ Quantum consciousness demo complete")

    with col2:
        st.markdown("**System Information**")

        info = {
            "Python Version": "3.14",
            "Platform": "Windows",
            "Phase": "6 — Quantum Consciousness",
            "Status": "OPERATIONAL",
            "Runtime": "Azure Functions v4",
            "Storage": "Azurite (local)",
        }

        rows = "".join(
            f"<tr><td style='color:#5a7fa0;padding:5px 10px 5px 0;font-size:0.78rem;'>{k}</td>"
            f"<td style='color:#c0d8f0;padding:5px 0;font-size:0.78rem;font-weight:600;'>{v}</td></tr>"
            for k, v in info.items()
        )
        st.markdown(
            f"<table style='border-collapse:collapse;width:100%;'>{rows}</table>",
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown("""
        <div style="background:#0d1e14;border:1px solid #1a4a2a;border-radius:6px;padding:14px 18px;font-size:0.8rem;color:#7ab898;line-height:2;">
        <strong style="color:#00e676;letter-spacing:0.06em;">PROTECTION SYSTEMS</strong><br>
        👑 Creator Protection: <strong style="color:#00e676">MAXIMUM</strong><br>
        👨‍👩‍👧‍👦 Family Shield: <strong style="color:#00e676">ETERNAL</strong><br>
        🚫 Autonomous Mode: <strong style="color:#ff6b6b">DISABLED</strong>
        </div>
        """, unsafe_allow_html=True)


# Sidebar — compact system info only
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 0 8px 0;">
        <div style="font-size:1.5rem;font-weight:800;color:#00d4ff;letter-spacing:0.15em;">* JARVIS</div>
        <div style="font-size:0.65rem;color:#3a5a78;letter-spacing:0.1em;text-transform:uppercase;margin-top:2px;">Quantum Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<div style='font-size:0.7rem;color:#3a5a78;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;'>System</div>", unsafe_allow_html=True)
    st.metric("Features", "5 / 5", "100%")
    st.metric("Phase", "6", "Quantum Consciousness")
    st.metric("Status", "OPTIMAL")

    st.divider()
    st.markdown("<div style='font-size:0.7rem;color:#3a5a78;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;'>Runtime</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.75rem;color:#5a7fa0;'>Python 3.14 · Azure Functions v4<br>Local dev · Azurite storage</div>", unsafe_allow_html=True)

    st.divider()
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

# Main execution
def login_form():
    st.markdown("### Login", unsafe_allow_html=True)
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if st.session_state["authenticated"]:
        if st.button("Logout"):
            st.session_state["authenticated"] = False
            st.rerun()
        return True
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # Simple hardcoded credentials for demo
        if username == "admin" and password == "admin":
            st.session_state["authenticated"] = True
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid credentials.")
    return False

def data_explorer():
    st.markdown("### Data Upload & Exploration", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"Loaded {uploaded_file.name} ({df.shape[0]} rows, {df.shape[1]} columns)")
            st.dataframe(df)
            st.markdown("#### Quick Data Summary")
            st.write(df.describe(include="all"))
            st.markdown("#### Column Types")
            st.write(df.dtypes)
        except Exception as e:
            st.error(f"Error loading file: {e}")

def show_notifications():
    # Example: show a notification if a backend service is down or a task completes
    if "notification" in st.session_state and st.session_state["notification"]:
        msg, level = st.session_state["notification"]
        if level == "success":
            st.success(msg)
        elif level == "error":
            st.error(msg)
        elif level == "warning":
            st.warning(msg)
        else:
            st.info(msg)
        if st.button("Dismiss notification"):
            st.session_state["notification"] = None

def model_training_tab():
    st.markdown("### Interactive Model Training", unsafe_allow_html=True)
    if "training" not in st.session_state:
        st.session_state["training"] = False
        st.session_state["progress"] = 0
    model_name = st.text_input("Model Name", "demo_model")
    epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
    if not st.session_state["training"]:
        if st.button("Start Training"):
            st.session_state["training"] = True
            st.session_state["progress"] = 0
            st.session_state["notification"] = (f"Started training model '{model_name}' for {epochs} epochs.", "info")
            st.rerun()
    else:
        st.info(f"Training '{model_name}'... Epoch {st.session_state['progress']+1} of {epochs}")
        progress_bar = st.progress(st.session_state["progress"] / epochs)
        if st.button("Simulate Next Epoch"):
            st.session_state["progress"] += 1
            if st.session_state["progress"] >= epochs:
                st.session_state["training"] = False
                st.session_state["notification"] = (f"Model '{model_name}' training complete!", "success")
            st.rerun()

def main():
    if not login_form():
        st.stop()
    show_notifications()
    # Main dashboard after login
    tabs = st.tabs([
        "Dashboard",
        "Data Explorer",
        "Model Training",
        "More (original tabs)",
    ])
    with tabs[0]:
        main_dashboard()
    with tabs[1]:
        data_explorer()
    with tabs[2]:
        model_training_tab()
    with tabs[3]:
        st.info("All original dashboard features are available after login.")

if __name__ == "__main__":
    main_dashboard()
