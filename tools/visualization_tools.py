import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json


class VisualizationTools:
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3

    def create_confidence_chart(self, confidence_scores: Dict[str, float]) -> go.Figure:
        """Create bar chart of confidence scores"""
        components = list(confidence_scores.keys())
        scores = list(confidence_scores.values())

        fig = go.Figure(data=[
            go.Bar(
                x=components,
                y=scores,
                marker_color=[self._get_confidence_color(score) for score in scores],
                text=[f"{score:.1%}" for score in scores],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title="Agent Confidence Scores",
            xaxis_title="Component",
            yaxis_title="Confidence Score",
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            height=400
        )

        return fig

    def create_timeline_chart(self, events: List[Dict[str, Any]]) -> go.Figure:
        """Create timeline of research process"""
        df = pd.DataFrame(events)

        fig = px.timeline(
            df,
            x_start="start",
            x_end="end",
            y="agent",
            color="status",
            title="Research Process Timeline",
            hover_data=["description"]
        )

        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=300)

        return fig

    def create_data_source_chart(self, data_sources: List[Dict]) -> go.Figure:
        """Create pie chart of data sources"""
        source_types = {}
        for source in data_sources:
            source_type = source.get("type", "Unknown")
            source_types[source_type] = source_types.get(source_type, 0) + 1

        fig = go.Figure(data=[go.Pie(
            labels=list(source_types.keys()),
            values=list(source_types.values()),
            hole=0.3,
            marker_colors=self.color_palette
        )])

        fig.update_layout(
            title="Data Sources Distribution",
            height=400
        )

        return fig

    def create_experiment_results_chart(self, experiments: List[Dict]) -> go.Figure:
        """Create visualization of experiment results"""
        if not experiments:
            return self._create_empty_chart("No experiments to visualize")

        # Prepare data
        exp_names = []
        confidence_scores = []
        sample_sizes = []

        for exp in experiments:
            exp_names.append(exp.get("name", "Unknown"))
            confidence_scores.append(exp.get("confidence", 0.5))
            sample_sizes.append(exp.get("sample_size", 0))

        # Create grouped bar chart
        fig = go.Figure(data=[
            go.Bar(name='Confidence', x=exp_names, y=confidence_scores,
                   marker_color='lightblue', yaxis='y'),
            go.Bar(name='Sample Size', x=exp_names, y=sample_sizes,
                   marker_color='lightcoral', yaxis='y2')
        ])

        fig.update_layout(
            title="Experiment Metrics",
            xaxis_title="Experiment",
            yaxis=dict(
                title="Confidence",
                tickformat=".0%",
                range=[0, 1]
            ),
            yaxis2=dict(
                title="Sample Size",
                overlaying='y',
                side='right'
            ),
            barmode='group',
            height=400
        )

        return fig

    def create_uncertainty_heatmap(self, uncertainties: List[Dict]) -> go.Figure:
        """Create heatmap of uncertainty sources"""
        if not uncertainties:
            return self._create_empty_chart("No uncertainty data")

        components = [unc["component"] for unc in uncertainties]
        levels = [unc["level"] for unc in uncertainties]
        scores = [unc["score"] for unc in uncertainties]

        # Convert levels to numerical values for heatmap
        level_map = {"Very Low": 0.9, "Low": 0.7, "Medium": 0.5, "High": 0.3}
        level_values = [level_map.get(level, 0.5) for level in levels]

        fig = go.Figure(data=go.Heatmap(
            z=[level_values, scores],
            x=components,
            y=['Uncertainty Level', 'Confidence Score'],
            colorscale='RdYlGn_r',  # Red (high uncertainty) to Green (low uncertainty)
            showscale=True,
            text=[[str(l) for l in levels], [f"{s:.1%}" for s in scores]],
            texttemplate="%{text}",
            textfont={"size": 12}
        ))

        fig.update_layout(
            title="Uncertainty Analysis Heatmap",
            height=300
        )

        return fig

    def create_progress_radar(self, iteration_data: Dict[int, Dict]) -> go.Figure:
        """Create radar chart showing progress across iterations"""
        if not iteration_data:
            return self._create_empty_chart("No iteration data")

        iterations = list(iteration_data.keys())
        metrics = ["confidence", "data_sources", "experiments"]

        # Prepare data for radar chart
        radar_data = []

        for metric in metrics:
            values = []
            for iteration in iterations:
                value = iteration_data[iteration].get(metric, 0)
                if isinstance(value, (int, float)):
                    values.append(value)
                else:
                    values.append(0)

            # Normalize values for radar chart
            if values:
                max_val = max(values) if max(values) > 0 else 1
                normalized = [v / max_val for v in values]
            else:
                normalized = [0] * len(iterations)

            radar_data.append(go.Scatterpolar(
                r=normalized,
                theta=[f"Iter {i}" for i in iterations],
                fill='toself',
                name=metric.replace('_', ' ').title()
            ))

        fig = go.Figure(data=radar_data)

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Progress Across Iterations",
            height=500
        )

        return fig

    def _get_confidence_color(self, score: float) -> str:
        """Get color based on confidence score"""
        if score >= 0.8:
            return "#2ecc71"  # Green
        elif score >= 0.6:
            return "#f39c12"  # Orange
        elif score >= 0.4:
            return "#e74c3c"  # Red
        else:
            return "#95a5a6"  # Gray

    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            height=200,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig

    def export_visualizations(self, charts: Dict[str, go.Figure],
                              format: str = "html") -> Dict[str, str]:
        """Export visualizations in specified format"""
        exported = {}

        for name, fig in charts.items():
            if format == "html":
                exported[name] = fig.to_html(full_html=False, include_plotlyjs='cdn')
            elif format == "json":
                exported[name] = fig.to_json()
            elif format == "png":
                # Note: Plotly requires kaleido for static image export
                try:
                    exported[name] = fig.to_image(format="png", engine="kaleido")
                except:
                    exported[name] = "PNG export not available"

        return exported