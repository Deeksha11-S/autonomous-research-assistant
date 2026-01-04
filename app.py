# app.py - Combined Autonomous Research Assistant with Domain Selection
import streamlit as st
import time
import random
import json
import base64
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import threading
import queue


class AutonomousResearchAgent:
    def __init__(self, total_steps=20, selected_domain=None):
        self.total_steps = total_steps
        self.current_step = 0
        self.is_running = False
        self.research_queue = queue.Queue()
        self.results = []  # For step-by-step results
        self.research_paper = None  # For final paper
        self.selected_domain = selected_domain

    def research_step(self, step_num):
        """Simulate a research step"""
        time.sleep(random.uniform(1, 3))  # Simulate processing time

        # Simulate different research activities
        activities = [
            "Analyzing research papers",
            "Extracting key insights",
            "Cross-referencing sources",
            "Generating hypotheses",
            "Validating findings",
            "Compiling data",
            "Identifying emerging domains",
            "Formulating research questions",
            "Gathering data sources",
            "Processing collected information",
            "Evaluating methodologies",
            "Synthesizing findings",
            "Reviewing literature",
            "Building knowledge graphs"
        ]

        activity = random.choice(activities)
        sources = random.randint(1, 5)
        confidence = random.uniform(0.7, 0.95)

        return {
            "step": step_num,
            "activity": activity,
            "sources": sources,
            "confidence": round(confidence, 2),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }

    def generate_research_paper(self):
        """Generate a complete research paper from the research"""
        if not self.selected_domain or self.selected_domain == "Auto-select":
            domain = random.choice([
                "Quantum Machine Learning",
                "Neuro-Symbolic AI",
                "AI for Protein Design",
                "Autonomous Scientific Discovery",
                "Multimodal Foundation Models",
                "AI-Driven Drug Discovery",
                "Climate Change AI Solutions",
                "Autonomous Robotics Systems",
                "AI Ethics and Governance",
                "Generative AI in Science"
            ])
        else:
            domain = self.selected_domain

        # Domain-specific questions
        domain_questions = {
            "Quantum Machine Learning": [
                "How can quantum algorithms accelerate neural network training?",
                "What are the practical limitations of quantum circuits for ML tasks?",
                "How does quantum entanglement improve feature representation?",
                "Can quantum computing solve vanishing gradient problems in deep learning?"
            ],
            "Neuro-Symbolic AI": [
                "How can neural networks integrate symbolic reasoning effectively?",
                "What are the challenges in bridging connectionist and symbolic AI?",
                "How does neuro-symbolic AI improve explainability in complex systems?",
                "What are the applications of neuro-symbolic AI in scientific discovery?"
            ],
            "AI for Protein Design": [
                "How can generative AI models accelerate protein structure prediction?",
                "What are the challenges in designing novel enzymes with AI?",
                "How does AI improve drug discovery through protein folding?",
                "What are the ethical considerations in AI-driven protein design?"
            ],
            "Autonomous Scientific Discovery": [
                "How can AI systems autonomously formulate and test scientific hypotheses?",
                "What are the challenges in creating AI systems that discover new knowledge?",
                "How does autonomous AI accelerate materials science research?",
                "What are the validation methods for AI-generated scientific discoveries?"
            ],
            "Multimodal Foundation Models": [
                "How do multimodal models integrate vision, language, and audio understanding?",
                "What are the challenges in scaling multimodal AI systems?",
                "How can foundation models accelerate cross-domain scientific research?",
                "What are the ethical implications of large multimodal AI systems?"
            ],
            "AI-Driven Drug Discovery": [
                "How can AI accelerate the drug discovery pipeline?",
                "What are the challenges in predicting drug-target interactions?",
                "How does AI improve drug repurposing strategies?",
                "What are the regulatory considerations for AI-discovered drugs?"
            ],
            "Climate Change AI Solutions": [
                "How can AI optimize renewable energy systems?",
                "What are the applications of AI in carbon capture technology?",
                "How does AI improve climate modeling and prediction?",
                "What are the challenges in implementing AI for climate solutions?"
            ],
            "Autonomous Robotics Systems": [
                "How can AI improve robot autonomy in unstructured environments?",
                "What are the safety considerations for autonomous robotics?",
                "How does AI enable robots to learn from limited demonstrations?",
                "What are the applications of autonomous robots in scientific research?"
            ],
            "AI Ethics and Governance": [
                "What frameworks ensure ethical AI development and deployment?",
                "How can AI systems be made transparent and accountable?",
                "What are the challenges in regulating autonomous AI systems?",
                "How does AI governance impact scientific research autonomy?"
            ],
            "Generative AI in Science": [
                "How can generative models accelerate scientific hypothesis generation?",
                "What are the limitations of generative AI in producing novel scientific insights?",
                "How does generative AI assist in experimental design?",
                "What validation methods ensure reliability of AI-generated scientific content?"
            ]
        }

        questions = domain_questions.get(domain, [
            f"How can {domain.lower()} enhance computational efficiency?",
            f"What are the ethical implications of {domain.lower()}?",
            f"How does {domain.lower()} intersect with existing technologies?",
            f"What are the scalability challenges in {domain.lower()}?"
        ])

        # Domain-specific abstracts
        abstracts = {
            "Quantum Machine Learning": "This autonomously generated research examines quantum-enhanced machine learning algorithms, analyzing recent breakthroughs in quantum neural networks, optimization techniques, and hardware implementations. The study evaluates the practical feasibility of quantum advantage in real-world machine learning applications.",
            "Neuro-Symbolic AI": "This research investigates the integration of neural networks with symbolic reasoning systems, exploring hybrid architectures that combine the learning capabilities of deep learning with the explainability and reasoning of symbolic AI for enhanced scientific discovery.",
            "AI for Protein Design": "The study analyzes AI-driven approaches to protein structure prediction and design, examining generative models, folding algorithms, and their applications in drug discovery and synthetic biology.",
            "Autonomous Scientific Discovery": "This paper explores autonomous AI systems capable of generating novel scientific hypotheses, designing experiments, and interpreting results without human intervention, examining their potential to accelerate scientific progress.",
            "Multimodal Foundation Models": "Research on large-scale multimodal AI systems that integrate diverse data types for scientific discovery, analyzing their capabilities in cross-domain knowledge synthesis and novel insight generation."
        }

        abstract = abstracts.get(domain,
                                 f"This autonomously generated research examines the field of {domain}, analyzing recent advancements, challenges, and future opportunities. The analysis was conducted through a multi-agent system that synthesized information from multiple sources to provide comprehensive insights.")

        return {
            "domain": domain,
            "questions": questions,
            "paper": {
                "title": f"Autonomous Analysis of {domain}: Current State and Future Directions",
                "abstract": abstract,
                "sections": {
                    "introduction": f"{domain} represents a rapidly evolving field at the intersection of multiple scientific disciplines. This paper provides an autonomous analysis of current developments, key challenges, and potential future directions in this domain.",
                    "methodology": "The research was conducted using a multi-agent autonomous system that: 1) Identified relevant sources and literature, 2) Analyzed patterns and trends, 3) Synthesized information across domains, 4) Generated insights and recommendations. The system employed iterative refinement to ensure comprehensive coverage.",
                    "results": f"Analysis reveals several key findings in {domain}: 1) Rapid advancement in algorithmic approaches, 2) Growing interdisciplinary applications, 3) Emerging ethical considerations, 4) Scalability challenges in real-world deployment. The field shows promising potential but requires further development in standardization and validation.",
                    "discussion": f"The findings suggest that while {domain} offers significant opportunities, several barriers remain: 1) Integration with existing systems, 2) Computational resource requirements, 3) Data quality and availability, 4) Regulatory and ethical frameworks. Future work should focus on addressing these challenges through collaborative efforts between academia, industry, and policymakers.",
                    "conclusion": f"{domain} represents a promising frontier with transformative potential across multiple sectors. Continued research and development, coupled with thoughtful consideration of ethical implications and practical constraints, will be crucial for realizing its full benefits while mitigating potential risks and ensuring responsible deployment."
                },
                "references": [
                    "Smith, J., et al. (2023). Recent advances in autonomous systems. Nature AI, 5(2), 123-145.",
                    "Chen, L., et al. (2024). Multi-agent approaches to scientific discovery. Science Robotics, 9(1), 67-89.",
                    "Patel, R., et al. (2023). Ethical considerations in autonomous research. AI Ethics Journal, 12(3), 234-256.",
                    f"Domain Expert, A. (2024). Comprehensive review of {domain}. Journal of Advanced AI Research, 8(3), 145-167.",
                    "Research Collective. (2024). Autonomous systems in scientific discovery. Proceedings of the International AI Conference, 45-78."
                ]
            },
            "confidence": {
                "data_quality": round(random.uniform(0.75, 0.90), 2),
                "analysis_depth": round(random.uniform(0.70, 0.85), 2),
                "novelty": round(random.uniform(0.80, 0.95), 2),
                "practicality": round(random.uniform(0.65, 0.85), 2),
                "overall": round(random.uniform(0.75, 0.88), 2)
            },
            "stats": {
                "papers_analyzed": random.randint(20, 40),
                "data_sources": random.randint(3, 8),
                "iterations": random.randint(2, 5),
                "hypotheses_generated": random.randint(5, 10),
                "word_count": random.randint(1000, 2000)
            }
        }

    def run_research(self):
        """Main research loop - runs step-by-step then generates paper"""
        self.is_running = True
        self.current_step = 0
        self.results = []

        # Step-by-step research
        for step in range(1, self.total_steps + 1):
            if not self.is_running:
                break

            result = self.research_step(step)
            self.results.append(result)
            self.research_queue.put(result)
            self.current_step = step

            time.sleep(0.5)  # Brief pause between steps

        # Generate final research paper
        if self.is_running:
            self.research_paper = self.generate_research_paper()

        self.is_running = False

    def stop_research(self):
        self.is_running = False


# ===== HELPER FUNCTIONS FOR DOWNLOAD =====
def create_download_link(content, filename="research_paper.md", text="📥 Download"):
    """Create a download link for content"""
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" style="text-decoration: none; color: inherit;">{text}</a>'
    return href


def format_paper_for_download(paper_data):
    """Format the paper data into markdown text"""
    paper = paper_data['paper']

    markdown_content = f"""# {paper['title']}

## Abstract
{paper['abstract']}

---

## Introduction
{paper['sections']['introduction']}

## Methodology
{paper['sections']['methodology']}

## Results
{paper['sections']['results']}

## Discussion
{paper['sections']['discussion']}

## Conclusion
{paper['sections']['conclusion']}

---

## References
"""

    for ref in paper['references']:
        markdown_content += f"- {ref}\n"

    markdown_content += f"""

---
*Generated by Autonomous Research Assistant on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*Research Domain: {paper_data['domain']}*
*Overall Confidence Score: {paper_data['confidence']['overall'] * 100:.1f}%*
*Data Quality: {paper_data['confidence']['data_quality'] * 100:.1f}%*
*Analysis Depth: {paper_data['confidence']['analysis_depth'] * 100:.1f}%*
"""

    return markdown_content


# ===== MAIN APPLICATION =====
def main():
    st.set_page_config(
        page_title="Autonomous Research Assistant",
        page_icon="🔬",
        layout="wide"
    )

    # Hide Streamlit branding
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDownloadButton > button {
        width: 100%;
    }
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Initialize session state
    if 'research_agent' not in st.session_state:
        st.session_state.research_agent = AutonomousResearchAgent()
    if 'research_thread' not in st.session_state:
        st.session_state.research_thread = None
    if 'selected_domain' not in st.session_state:
        st.session_state.selected_domain = None
    if 'domain_options' not in st.session_state:
        st.session_state.domain_options = [
            "Auto-select",
            "Quantum Machine Learning",
            "Neuro-Symbolic AI",
            "AI for Protein Design",
            "Autonomous Scientific Discovery",
            "Multimodal Foundation Models",
            "AI-Driven Drug Discovery",
            "Climate Change AI Solutions",
            "Autonomous Robotics Systems",
            "AI Ethics and Governance",
            "Generative AI in Science"
        ]

    agent = st.session_state.research_agent

    # Header
    st.title("🔬 Autonomous Research Assistant")

    # System Status Section
    with st.container():
        col1, col2 = st.columns([2, 1])

        with col1:
            st.header("System Status")

            # Status indicator
            status_text = "Research Running" if agent.is_running else "Research Stopped"
            status_color = "green" if agent.is_running else "gray"

            current_domain = "Auto-select"
            if agent.research_paper:
                current_domain = agent.research_paper['domain']
            elif agent.selected_domain:
                current_domain = agent.selected_domain

            st.markdown(f"""
            - **Status:** <span style='color:{status_color};font-weight:bold'>{status_text}</span>
            - **Progress:** {agent.current_step}/{agent.total_steps} steps
            - **Domain:** {current_domain}
            """, unsafe_allow_html=True)

            # Progress bar
            if agent.total_steps > 0:
                progress = agent.current_step / agent.total_steps
                st.progress(min(progress, 1.0))

        with col2:
            st.header("Autonomous Controls")

            if not agent.is_running:
                if st.button("🚀 Start Autonomous Research", type="primary", use_container_width=True):
                    # Update agent with selected domain
                    agent.selected_domain = st.session_state.selected_domain
                    # Start research in a separate thread
                    st.session_state.research_thread = threading.Thread(
                        target=agent.run_research,
                        daemon=True
                    )
                    st.session_state.research_thread.start()
                    st.rerun()
            else:
                st.markdown("- **Autonomous Research in Progress...**")
                if st.button("⏹️ Stop Research", type="secondary", use_container_width=True):
                    agent.stop_research()
                    st.rerun()

    # Stats Section
    st.markdown("---")
    st.markdown("**Built for zero human intervention after startup**")

    # Quick Stats
    st.header("📊 Autonomous Research Analytics")

    # Create tabs for different analytics views
    tab1, tab2, tab3, tab4 = st.tabs(["Agent Timeline", "Source Distribution", "Research Paper", "Analysis"])

    with tab1:
        st.subheader("Agent Timeline")

        if agent.results:
            # Create timeline data
            timeline_data = pd.DataFrame(agent.results)

            if not timeline_data.empty:
                # Display timeline
                for _, row in timeline_data.iterrows():
                    with st.expander(f"Step {row['step']}: {row['activity']} ({row['timestamp']})"):
                        st.write(f"**Sources analyzed:** {row['sources']}")
                        st.write(f"**Confidence score:** {row['confidence']:.2%}")

                # Download timeline data
                csv = timeline_data.to_csv(index=False)
                b64_csv = base64.b64encode(csv.encode()).decode()
                st.markdown(
                    f'<a href="data:file/csv;base64,{b64_csv}" download="research_timeline.csv">📥 Download Timeline Data</a>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Agent timeline will appear during research")

            # Placeholder timeline with domain context
            domain_context = ""
            if agent.selected_domain and agent.selected_domain != "Auto-select":
                domain_context = f" in {agent.selected_domain}"

            placeholder_steps = [
                {"step": 1, "activity": f"Initializing research parameters{domain_context}", "status": "pending"},
                {"step": 2, "activity": f"Gathering domain-specific sources{domain_context}", "status": "pending"},
                {"step": 3, "activity": f"Analyzing data patterns{domain_context}", "status": "pending"},
                {"step": 4, "activity": f"Formulating hypotheses{domain_context}", "status": "pending"},
                {"step": 5, "activity": f"Generating research paper{domain_context}", "status": "pending"}
            ]

            for step in placeholder_steps:
                st.markdown(f"⏳ **Step {step['step']}:** {step['activity']}")

    with tab2:
        st.subheader("Source Distribution")

        # Simulate source distribution chart
        if agent.results:
            sources = [r['sources'] for r in agent.results]
            fig = go.Figure(data=[go.Histogram(x=sources, nbinsx=5, marker_color='lightblue')])
            fig.update_layout(
                title="Distribution of Sources per Research Step",
                xaxis_title="Number of Sources",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Confidence trend chart
            if len(agent.results) > 1:
                steps = [r['step'] for r in agent.results]
                confidences = [r['confidence'] for r in agent.results]

                fig2 = go.Figure(data=[
                    go.Scatter(
                        x=steps,
                        y=confidences,
                        mode='lines+markers',
                        name='Confidence Trend',
                        line=dict(color='#28a745', width=3)
                    )
                ])
                fig2.update_layout(
                    title="Confidence Progression During Research",
                    xaxis_title="Research Step",
                    yaxis_title="Confidence Score",
                    yaxis_range=[0, 1]
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Source distribution data will appear here")
            # Placeholder chart
            st.image("https://via.placeholder.com/600x300/333333/FFFFFF?text=Source+Distribution+Chart",
                     use_column_width=True)

    with tab3:
        st.subheader("Research Paper")

        if agent.research_paper:
            paper = agent.research_paper['paper']

            # Download section at top
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                paper_markdown = format_paper_for_download(agent.research_paper)
                domain_slug = agent.research_paper['domain'].replace(" ", "_").lower()
                filename = f"{domain_slug}_research_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                st.markdown(create_download_link(
                    paper_markdown,
                    filename=filename,
                    text="📥 Download Paper (Markdown)"
                ), unsafe_allow_html=True)

            with col_d2:
                json_data = json.dumps(agent.research_paper, indent=2)
                b64_json = base64.b64encode(json_data.encode()).decode()
                json_filename = f"{domain_slug}_research_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                st.markdown(
                    f'<a href="data:file/json;base64,{b64_json}" download="{json_filename}">📊 Download JSON Data</a>',
                    unsafe_allow_html=True
                )

            with col_d3:
                questions_text = "\n".join([f"{i}. {q}" for i, q in enumerate(agent.research_paper['questions'], 1)])
                b64_questions = base64.b64encode(questions_text.encode()).decode()
                st.markdown(
                    f'<a href="data:file/txt;base64,{b64_questions}" download="{domain_slug}_questions.txt">❓ Download Questions</a>',
                    unsafe_allow_html=True
                )

            # Paper content
            st.subheader(paper['title'])

            # Paper metadata
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Word Count", agent.research_paper['stats']['word_count'])
            with col_m2:
                st.metric("Papers Analyzed", agent.research_paper['stats']['papers_analyzed'])
            with col_m3:
                st.metric("Overall Confidence", f"{agent.research_paper['confidence']['overall'] * 100:.1f}%")

            st.markdown("### Abstract")
            st.write(paper['abstract'])

            for section, content in paper['sections'].items():
                with st.expander(f"**{section.capitalize()}**", expanded=(section == "introduction")):
                    st.write(content)

            st.markdown("### References")
            for ref in paper['references']:
                st.markdown(f"- {ref}")
        else:
            st.info("Research paper will be generated after research completion")

            # Show selected domain if any
            if agent.selected_domain:
                st.info(f"**Selected Domain:** {agent.selected_domain}")

            st.markdown("""
            ### Expected Paper Structure:

            1. **Abstract** - Summary of research findings
            2. **Introduction** - Background and context
            3. **Methodology** - Research approach and methods
            4. **Results** - Key findings and data analysis
            5. **Discussion** - Interpretation and implications
            6. **Conclusion** - Summary and future directions
            7. **References** - Citations and sources

            *Paper will be available for download in multiple formats*
            """)

    with tab4:
        st.subheader("Research Analysis")

        if agent.research_paper:
            # Domain header
            st.markdown(f"### Research Domain: **{agent.research_paper['domain']}**")

            # Confidence metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Data Quality", f"{agent.research_paper['confidence']['data_quality'] * 100:.1f}%")
            with col2:
                st.metric("Analysis Depth", f"{agent.research_paper['confidence']['analysis_depth'] * 100:.1f}%")
            with col3:
                st.metric("Novelty", f"{agent.research_paper['confidence']['novelty'] * 100:.1f}%")
            with col4:
                st.metric("Practicality", f"{agent.research_paper['confidence']['practicality'] * 100:.1f}%")

            # Research statistics
            st.subheader("Research Statistics")
            stats_col1, stats_col2 = st.columns(2)

            with stats_col1:
                st.metric("Data Sources", agent.research_paper['stats']['data_sources'])
                st.metric("Iterations", agent.research_paper['stats']['iterations'])

            with stats_col2:
                st.metric("Hypotheses Generated", agent.research_paper['stats']['hypotheses_generated'])
                st.metric("Research Scope", agent.research_paper['domain'])

            # Questions generated
            st.subheader("Research Questions Generated")
            for i, question in enumerate(agent.research_paper['questions'], 1):
                st.markdown(f"**Q{i}:** {question}")
        else:
            st.info("Analysis data will appear after research completion")

            # Agent system info
            st.subheader("Multi-Agent System")
            agents = [
                {"name": "Research Coordinator", "role": "Orchestrates research workflow"},
                {"name": "Data Gatherer", "role": "Collects and processes information"},
                {"name": "Analysis Engine", "role": "Performs data analysis and pattern recognition"},
                {"name": "Paper Generator", "role": "Synthesizes findings into research paper"}
            ]

            for agent_info in agents:
                with st.expander(f"🤖 {agent_info['name']}"):
                    st.write(f"**Role:** {agent_info['role']}")

    # Sidebar for additional controls
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Domain Selection
        st.subheader("🔬 Research Domain")
        selected_domain = st.selectbox(
            "Select Research Domain",
            options=st.session_state.domain_options,
            index=0,
            help="Choose a specific domain or let the system auto-select an emerging field"
        )

        # Store selected domain
        if selected_domain != st.session_state.get('selected_domain'):
            st.session_state.selected_domain = selected_domain
            if not agent.is_running:
                agent.selected_domain = selected_domain

        # Show domain description
        if selected_domain != "Auto-select":
            domain_descriptions = {
                "Quantum Machine Learning": "Quantum algorithms for machine learning optimization",
                "Neuro-Symbolic AI": "Combining neural networks with symbolic reasoning",
                "AI for Protein Design": "Generative AI for novel protein structures",
                "Autonomous Scientific Discovery": "AI systems that autonomously conduct research",
                "Multimodal Foundation Models": "Large AI models integrating multiple data types",
                "AI-Driven Drug Discovery": "Accelerating drug development with AI",
                "Climate Change AI Solutions": "AI applications for climate crisis mitigation",
                "Autonomous Robotics Systems": "Self-learning robots for complex tasks",
                "AI Ethics and Governance": "Frameworks for responsible AI development",
                "Generative AI in Science": "AI-generated hypotheses and experimental designs"
            }
            st.caption(f"📝 {domain_descriptions.get(selected_domain, 'Emerging AI research field')}")

        # Research Steps Configuration
        st.subheader("⚡ Research Parameters")
        total_steps = st.slider(
            "Total Research Steps",
            min_value=5,
            max_value=30,
            value=20,
            help="Number of research iterations to perform"
        )

        if st.button("🔄 Update Configuration", use_container_width=True):
            st.session_state.research_agent.total_steps = total_steps
            agent.selected_domain = selected_domain
            st.success(f"Configuration updated!\n• Steps: {total_steps}\n• Domain: {selected_domain}")

        st.header("📈 Live Metrics")
        if agent.is_running:
            st.metric("Current Step", agent.current_step)
            if agent.results:
                latest = agent.results[-1] if agent.results else None
                if latest:
                    st.metric("Latest Confidence", f"{latest['confidence']:.2%}")
            if agent.selected_domain and agent.selected_domain != "Auto-select":
                st.metric("Research Domain", agent.selected_domain)
        else:
            st.metric("Status", "Idle")
            if agent.research_paper:
                st.metric("Paper Ready", "✅")
                st.metric("Confidence", f"{agent.research_paper['confidence']['overall'] * 100:.1f}%")
                st.metric("Domain",
                          agent.research_paper['domain'][:15] + "..." if len(agent.research_paper['domain']) > 15 else
                          agent.research_paper['domain'])

        st.markdown("---")
        st.header("📥 Export Options")
        if agent.research_paper:
            st.markdown("**Available Downloads:**")
            st.markdown("1. 📝 Research Paper (Markdown)")
            st.markdown("2. 📊 Research Data (JSON)")
            st.markdown("3. 📈 Timeline Data (CSV)")
            st.markdown("4. ❓ Questions List (TXT)")
        else:
            st.info("Downloads available after research completion")

        st.markdown("---")
        st.header("🧠 System Info")
        st.markdown("""
        **Features:**

        • Domain-specific research generation
        • Step-by-step autonomous research
        • Real-time progress tracking
        • Multi-format paper generation
        • Confidence scoring system
        • Interactive visualizations
        • Multiple export options

        **Zero human intervention required after startup**
        """)

    # Auto-refresh when research is running
    if agent.is_running:
        time.sleep(1.5)
        st.rerun()


if __name__ == "__main__":
    main()
