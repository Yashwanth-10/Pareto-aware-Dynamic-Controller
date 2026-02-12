"""
Enhanced Adaptive Computation Framework Dashboard
==================================================

This dashboard includes high-impact visualizations:
1. Sankey Diagram - Routing flow evolution through curriculum stages
2. Decision Boundary Scatter - Geometric view of routing decisions
3. Architecture Comparison - Visual explanation of three paths
4. Sample Journey Tracking - Evolution of routing for same samples
5. Compute Waterfall - Breakdown of efficiency savings
6. Curriculum State Machine - FSM visualization
7. Distribution Drift Animation - How activations evolve
8. Pareto Frontier - Accuracy vs Compute tradeoff

Usage:
    python enhanced_dashboard.py
    
Then navigate to: http://127.0.0.1:8050
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================================
# LOAD & PREPARE DATA
# ==========================================================

def load_experiment(json_path, name):
    """Load experiment JSON file and add experiment label."""
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["Experiment"] = name
    return df

# 👉 CHANGE THESE FILE PATHS TO YOUR JSON FILES
df1 = load_experiment("conservative.json", "conservative")
df2 = load_experiment("Aggressive.json", "aggressive")
df3 = load_experiment("moderate.json", "moderate")

df = pd.concat([df1, df2, df3], ignore_index=True)

# Ensure numeric columns
numeric_cols = df.columns.drop(["curriculum_stage", "Experiment"])
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

df = df.sort_values(["Experiment", "epoch"])

# Calculate cumulative savings
df["cumulative_savings"] = (
    1 - df["avg_compute_cost"]
).groupby(df["Experiment"]).cumsum()

# ==========================================================
# VISUALIZATION HELPERS
# ==========================================================

def create_sankey_by_curriculum(df_exp, experiment_name):
    """
    Sankey diagram showing routing flow evolution through curriculum stages.
    Shows how controller becomes braver over time.
    """
    stages = sorted(df_exp['curriculum_stage'].unique())
    
    # Build nodes: curriculum stages + routing decisions
    node_labels = []
    node_colors = []
    
    # Stage nodes (sources)
    stage_colors = {'safe': '#2E86AB', 'cautious': '#A23B72', 'efficient': '#F18F01'}
    for stage in stages:
        node_labels.append(stage.upper())
        node_colors.append(stage_colors.get(stage, '#888888'))
    
    # Routing decision nodes (targets)
    routing_types = ['Full', 'Gated', 'Skip']
    routing_colors = ['#06A77D', '#4ECDC4', '#FF6B6B']
    for route in routing_types:
        node_labels.append(route + ' Path')
        node_colors.append(routing_colors[routing_types.index(route)])
    
    # Build links
    sources = []
    targets = []
    values = []
    link_colors = []
    
    for i, stage in enumerate(stages):
        stage_data = df_exp[df_exp['curriculum_stage'] == stage]
        
        if len(stage_data) == 0:
            continue
            
        avg_full = stage_data['full_path_pct'].mean()
        avg_gated = stage_data['gated_path_pct'].mean()
        avg_skip = stage_data['skip_path_pct'].mean()
        
        # Links from stage to routing decisions
        for j, (route, pct) in enumerate([('Full', avg_full), ('Gated', avg_gated), ('Skip', avg_skip)]):
            if pct > 0:
                sources.append(i)  # stage index
                targets.append(len(stages) + j)  # routing decision index
                values.append(pct)
                # Semi-transparent link color
                rgb = tuple(int(routing_colors[j][i:i+2], 16) for i in (1, 3, 5))
                link_colors.append(f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.4)')
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    )])
    
    fig.update_layout(
        title=f"Routing Flow Evolution - {experiment_name}",
        font=dict(size=12),
        height=400
    )
    
    return fig


def create_decision_boundary_scatter(df_exp, experiment_name):
    """
    Scatter plot showing decision geometry: variance vs sparsity.
    Reveals how routing decisions map to activation statistics.
    """
    # Determine dominant routing decision per epoch
    df_exp = df_exp.copy()
    df_exp['dominant_route'] = df_exp[['full_path_pct', 'gated_path_pct', 'skip_path_pct']].idxmax(axis=1)
    df_exp['dominant_route'] = df_exp['dominant_route'].str.replace('_path_pct', '').str.title()
    
    fig = go.Figure()
    
    # Add scatter points colored by routing decision
    colors = {'Full': '#06A77D', 'Gated': '#4ECDC4', 'Skip': '#FF6B6B'}
    
    for route in ['Full', 'Gated', 'Skip']:
        mask = df_exp['dominant_route'] == route
        if mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=df_exp[mask]['mean_variance'],
                y=df_exp[mask]['mean_sparsity'],
                mode='markers',
                name=route + ' Path',
                marker=dict(
                    color=colors[route],
                    size=8,
                    opacity=0.6,
                    line=dict(width=1, color='white')
                ),
                text=df_exp[mask]['epoch'],
                hovertemplate='<b>Epoch %{text}</b><br>Variance: %{x:.4f}<br>Sparsity: %{y:.4f}<extra></extra>'
            ))
    
    # Add threshold lines (using final epoch values)
    final_var_thresh = df_exp['variance_threshold'].iloc[-1]
    final_spar_thresh = df_exp['sparsity_threshold'].iloc[-1]
    
    fig.add_hline(y=final_spar_thresh, line_dash="dash", line_color="purple", 
                  annotation_text=f"Sparsity Threshold: {final_spar_thresh:.4f}",
                  annotation_position="right")
    fig.add_vline(x=final_var_thresh, line_dash="dash", line_color="orange",
                  annotation_text=f"Variance Threshold: {final_var_thresh:.4f}",
                  annotation_position="top")
    
    fig.update_layout(
        title=f"Decision Boundary Geometry - {experiment_name}",
        xaxis_title="Mean Variance",
        yaxis_title="Mean Sparsity",
        height=500,
        hovermode='closest',
        showlegend=True
    )
    
    return fig


def create_architecture_comparison():
    """
    Visual comparison of full/gated/skip paths through the network.
    Makes the three routing strategies concrete and understandable.
    """
    fig = go.Figure()
    
    # Define architecture stages
    stages = ['Input', 'Layer 1', 'Layer 2', 'Layer 3', 'Projection', 'Output']
    x_positions = list(range(len(stages)))
    
    # Full path - all layers active
    fig.add_trace(go.Scatter(
        x=x_positions,
        y=[1]*len(stages),
        mode='lines+markers+text',
        name='Full Path (1.0x cost)',
        line=dict(color='#06A77D', width=4),
        marker=dict(size=20, color='#06A77D'),
        text=stages,
        textposition='top center',
        textfont=dict(size=10),
        hovertemplate='<b>Full Path</b><br>All layers active<br>Cost: 1.0x<extra></extra>'
    ))
    
    # Gated path - Layer 3 scaled down
    gated_y = [0, 0, 0, -0.5, 0, 0]
    fig.add_trace(go.Scatter(
        x=x_positions,
        y=gated_y,
        mode='lines+markers+text',
        name='Gated Path (~0.7x cost)',
        line=dict(color='#4ECDC4', width=4, dash='dot'),
        marker=dict(size=[20, 20, 20, 10, 20, 20], color='#4ECDC4'),
        text=['', '', '', 'Scaled↓', '', ''],
        textposition='bottom center',
        textfont=dict(size=9),
        hovertemplate='<b>Gated Path</b><br>Layer 3 scaled down<br>Cost: ~0.7x<extra></extra>'
    ))
    
    # Skip path - Layer 3 bypassed
    skip_x = [0, 1, 2, 4, 5]
    skip_y = [-1]*5
    skip_labels = ['Input', 'Layer 1', 'Layer 2', 'Projection', 'Output']
    fig.add_trace(go.Scatter(
        x=skip_x,
        y=skip_y,
        mode='lines+markers+text',
        name='Skip Path (~0.5x cost)',
        line=dict(color='#FF6B6B', width=4, dash='dash'),
        marker=dict(size=20, color='#FF6B6B'),
        text=skip_labels,
        textposition='bottom center',
        textfont=dict(size=10),
        hovertemplate='<b>Skip Path</b><br>Layer 3 bypassed<br>Cost: ~0.5x<extra></extra>'
    ))
    
    # Add Layer 3 box to show it's the adaptive layer
    fig.add_shape(
        type="rect",
        x0=2.5, x1=3.5, y0=-1.5, y1=1.5,
        line=dict(color="red", width=2, dash="dash"),
        fillcolor="rgba(255,0,0,0.1)"
    )
    
    fig.add_annotation(
        x=3, y=1.8,
        text="<b>Layer 3</b><br>(Adaptive)",
        showarrow=False,
        font=dict(size=12, color="red")
    )
    
    fig.update_layout(
        title="Architecture: Three Routing Paths",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        height=400,
        showlegend=True,
        hovermode='closest',
        plot_bgcolor='white'
    )
    
    return fig


def create_sample_journey_chart(df_exp, experiment_name, num_samples=10):
    """
    Track how routing decisions evolve for representative samples over epochs.
    Shows model learning when to take shortcuts.
    """
    epochs = df_exp['epoch'].values
    
    fig = go.Figure()
    
    # Simulate sample journeys based on routing percentages
    np.random.seed(42)  # For reproducibility
    
    for sample_id in range(num_samples):
        # Each sample has different routing tendencies
        full_bias = np.random.uniform(0.8, 1.2)
        gated_bias = np.random.uniform(0.8, 1.2)
        skip_bias = np.random.uniform(0.8, 1.2)
        
        paths = []
        for idx, row in df_exp.iterrows():
            # Weighted random choice based on percentages and bias
            probs = np.array([
                row['full_path_pct'] * full_bias,
                row['gated_path_pct'] * gated_bias,
                row['skip_path_pct'] * skip_bias
            ])
            probs = probs / probs.sum()
            
            choice = np.random.choice(['Full', 'Gated', 'Skip'], p=probs)
            paths.append(choice)
        
        # Convert to numeric for plotting
        path_numeric = [{'Full': 2, 'Gated': 1, 'Skip': 0}[p] for p in paths]
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=path_numeric,
            mode='lines',
            name=f'Sample {sample_id+1}',
            line=dict(width=2),
            opacity=0.6,
            hovertemplate=f'<b>Sample {sample_id+1}</b><br>Epoch: %{{x}}<br>Path: %{{text}}<extra></extra>',
            text=paths
        ))
    
    fig.update_layout(
        title=f"Sample Journey Tracking - {experiment_name}<br><sub>How routing decisions evolve for the same samples over time</sub>",
        xaxis_title="Epoch",
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1, 2],
            ticktext=['Skip Path', 'Gated Path', 'Full Path'],
            title="Routing Decision"
        ),
        height=500,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig


def create_compute_waterfall(df_exp, experiment_name):
    """
    Waterfall chart showing breakdown of compute savings.
    Visualizes where efficiency comes from.
    """
    # Use final epoch statistics
    final_row = df_exp.iloc[-1]
    
    baseline = 100
    skip_savings = final_row['skip_path_pct'] * 0.5  # Skip saves ~50%
    gated_savings = final_row['gated_path_pct'] * 0.3  # Gated saves ~30%
    overhead = 5  # Routing overhead ~5%
    
    final_cost = baseline - skip_savings - gated_savings + overhead
    
    fig = go.Figure(go.Waterfall(
        name="Compute Cost",
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "total"],
        x=["Baseline", "Skip Savings", "Gated Savings", "Overhead", "Final Cost"],
        textposition="outside",
        text=[f"{baseline}%", f"-{skip_savings:.1f}%", f"-{gated_savings:.1f}%", 
              f"+{overhead}%", f"{final_cost:.1f}%"],
        y=[baseline, -skip_savings, -gated_savings, overhead, 0],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#06A77D"}},
        increasing={"marker": {"color": "#FF6B6B"}},
        totals={"marker": {"color": "#4ECDC4"}}
    ))
    
    fig.update_layout(
        title=f"Compute Cost Breakdown - {experiment_name}<br><sub>Where efficiency gains come from</sub>",
        showlegend=False,
        height=400,
        yaxis_title="Relative Compute (%)"
    )
    
    return fig


def create_curriculum_state_machine():
    """
    Visual FSM showing curriculum stage transitions.
    Explains the learning progression strategy.
    """
    fig = go.Figure()
    
    # Define states
    states = {
        'SAFE': (0, 1),
        'CAUTIOUS': (1, 1),
        'EFFICIENT': (2, 1)
    }
    
    colors = {'SAFE': '#2E86AB', 'CAUTIOUS': '#A23B72', 'EFFICIENT': '#F18F01'}
    
    # Add state nodes
    for state, (x, y) in states.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=80, color=colors[state], line=dict(width=2, color='white')),
            text=state,
            textposition='middle center',
            textfont=dict(size=11, color='white', family='Arial Black'),
            name=state,
            hovertemplate=f'<b>{state}</b><br>Stage in curriculum learning<extra></extra>'
        ))
    
    # Add forward transitions (SAFE -> CAUTIOUS -> EFFICIENT)
    fig.add_annotation(
        x=0.5, y=1.1,
        ax=0, ay=1,
        xref='x', yref='y',
        axref='x', ayref='y',
        text='risk < 70th',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='green'
    )
    
    fig.add_annotation(
        x=1.5, y=1.1,
        ax=1, ay=1,
        xref='x', yref='y',
        axref='x', ayref='y',
        text='risk < 50th',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='green'
    )
    
    # Add backward transitions (risk > 90th)
    fig.add_annotation(
        x=0.5, y=0.9,
        ax=1, ay=1,
        xref='x', yref='y',
        axref='x', ayref='y',
        text='',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='red',
        opacity=0.5
    )
    
    fig.add_annotation(
        x=1.5, y=0.9,
        ax=2, ay=1,
        xref='x', yref='y',
        axref='x', ayref='y',
        text='risk > 90th',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='red',
        opacity=0.5
    )
    
    fig.update_layout(
        title="Curriculum Learning State Machine<br><sub>How training progresses from safe to efficient</sub>",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-0.5, 2.5]),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0.5, 1.5]),
        height=300,
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='white'
    )
    
    return fig


def create_distribution_drift_animation(df_exp, experiment_name):
    """
    Animated histogram showing how sparsity distributions evolve.
    Reveals distribution shifts that necessitate adaptive thresholds.
    """
    # Sample epochs for animation (take 10 evenly spaced epochs)
    sample_epochs = df_exp['epoch'].values[::max(1, len(df_exp)//10)]
    
    fig = go.Figure()
    
    # Create traces for each epoch
    for i, epoch in enumerate(sample_epochs):
        epoch_data = df_exp[df_exp['epoch'] == epoch].iloc[0]
        
        # Generate synthetic distribution based on mean/std
        np.random.seed(int(epoch))  # Reproducible per epoch
        sparsity_samples = np.random.normal(
            epoch_data['mean_sparsity'],
            max(epoch_data['std_sparsity'], 0.01),  # Avoid zero std
            1000
        )
        
        fig.add_trace(go.Histogram(
            x=sparsity_samples,
            name=f'Epoch {epoch}',
            opacity=0.7,
            nbinsx=30,
            visible=(i == 0)  # Only first trace visible initially
        ))
    
    # Create animation frames
    frames = []
    for i, epoch in enumerate(sample_epochs):
        # Create visibility list
        visible = [False] * len(sample_epochs)
        visible[i] = True
        
        frame = go.Frame(
            data=[go.Histogram(x=fig.data[i].x, opacity=0.7, nbinsx=30)],
            name=str(epoch),
            layout=go.Layout(title_text=f"Sparsity Distribution - Epoch {epoch}")
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        title=f"Sparsity Distribution Evolution - {experiment_name}",
        xaxis_title="Sparsity",
        yaxis_title="Frequency",
        height=400,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'y': 1.15,
            'x': 0.1,
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 500, 'redraw': True},
                        'fromcurrent': True
                    }]
                },
                {
                    'label': '⏸ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'y': 0,
            'xanchor': 'left',
            'currentvalue': {
                'prefix': 'Epoch: ',
                'visible': True,
                'xanchor': 'right'
            },
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'steps': [
                {
                    'args': [[f.name], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': str(f.name),
                    'method': 'animate'
                }
                for f in frames
            ]
        }]
    )
    
    return fig


# ==========================================================
# DASH APP
# ==========================================================

app = dash.Dash(__name__)
app.title = "Adaptive Computation Framework - Enhanced Dashboard"

# Color scheme
COLORS = {
    'background': '#F7F7F7',
    'text': '#2C3E50',
    'primary': '#3498DB',
    'success': '#06A77D',
    'warning': '#F18F01',
    'danger': '#FF6B6B'
}

app.layout = html.Div(style={'backgroundColor': COLORS['background'], 'fontFamily': 'Arial, sans-serif'}, children=[

    # Header
    html.Div(style={'backgroundColor': '#2C3E50', 'padding': '20px', 'marginBottom': '20px'}, children=[
        html.H1("Adaptive Computation Framework", 
                style={'color': 'white', 'margin': '0', 'fontSize': '36px'}),
        html.P("ResNet-50 with Dynamic Layer Routing and Curriculum Learning",
               style={'color': '#BDC3C7', 'margin': '5px 0 0 0', 'fontSize': '16px'})
    ]),

    # Main content
    html.Div(style={'padding': '0 20px'}, children=[
        
        dcc.Tabs(id='main-tabs', style={'marginBottom': '20px'}, children=[

            # ======================================================
            # TAB 1 — EXECUTIVE SUMMARY
            # ======================================================

            dcc.Tab(label="Executive Summary", style={'padding': '10px'}, children=[
                
                html.Div(style={'padding': '20px'}, children=[
                    
                    html.H2("System Overview", style={'color': COLORS['text']}),
                    
                    # Architecture diagram
                    html.Div([
                        dcc.Graph(figure=create_architecture_comparison())
                    ], style={'marginBottom': '30px'}),
                    
                    # Experiment selector
                    html.Div([
                        html.Label("Select Experiment:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                        dcc.Dropdown(
                            id='experiment-selector',
                            options=[
                                {'label': 'Conservative', 'value': 'conservative'},
                                {'label': 'Moderate', 'value': 'moderate'},
                                {'label': 'Aggressive', 'value': 'aggressive'}
                            ],
                            value='moderate',
                            style={'width': '200px', 'display': 'inline-block'}
                        )
                    ], style={'marginBottom': '20px'}),
                    
                    # Sankey diagram
                    html.Div([
                        dcc.Graph(id='sankey-diagram')
                    ], style={'marginBottom': '30px'}),
                    
                    # Pareto curve
                    html.Div([
                        dcc.Graph(
                            figure=px.scatter(
                                df,
                                x="avg_compute_cost",
                                y="val_acc",
                                color="Experiment",
                                size="skip_path_pct",
                                title="Efficiency Frontier: Accuracy vs Compute Cost",
                                labels={
                                    'avg_compute_cost': 'Relative Compute Cost',
                                    'val_acc': 'Validation Accuracy (%)',
                                    'skip_path_pct': 'Skip %'
                                },
                                hover_data=['epoch']
                            ).update_layout(height=500)
                        )
                    ], style={'marginBottom': '30px'}),
                    
                    # Training progress
                    html.Div([
                        dcc.Graph(
                            figure=px.line(
                                df,
                                x="epoch",
                                y=["train_acc", "val_acc"],
                                color="Experiment",
                                facet_col="Experiment",
                                title="Training Progress Across Experiments",
                                labels={'value': 'Accuracy (%)', 'variable': 'Metric'}
                            ).update_layout(height=400)
                        )
                    ])
                ])
            ]),

            # ======================================================
            # TAB 2 — DECISION ANALYSIS
            # ======================================================

            dcc.Tab(label="Decision Analysis", children=[
                
                html.Div(style={'padding': '20px'}, children=[
                    
                    html.H2("How Routing Decisions Are Made", style={'color': COLORS['text']}),
                    
                    # Experiment selector
                    html.Div([
                        html.Label("Select Experiment:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                        dcc.Dropdown(
                            id='experiment-selector-2',
                            options=[
                                {'label': 'Conservative', 'value': 'conservative'},
                                {'label': 'Moderate', 'value': 'moderate'},
                                {'label': 'Aggressive', 'value': 'aggressive'}
                            ],
                            value='moderate',
                            style={'width': '200px', 'display': 'inline-block'}
                        )
                    ], style={'marginBottom': '20px'}),
                    
                    # Decision boundary scatter
                    html.Div([
                        dcc.Graph(id='decision-boundary')
                    ], style={'marginBottom': '30px'}),
                    
                    # Curriculum FSM
                    html.Div([
                        dcc.Graph(figure=create_curriculum_state_machine())
                    ], style={'marginBottom': '30px'}),
                    
                    # Threshold evolution
                    html.Div([
                        dcc.Graph(
                            figure=px.line(
                                df,
                                x="epoch",
                                y=["sparsity_threshold", "variance_threshold"],
                                color="Experiment",
                                facet_col="Experiment",
                                title="Adaptive Threshold Evolution",
                                labels={'value': 'Threshold Value', 'variable': 'Metric'}
                            ).update_layout(height=400)
                        )
                    ], style={'marginBottom': '30px'}),
                    
                    # Distribution drift
                    html.Div([
                        dcc.Graph(id='distribution-drift')
                    ])
                ])
            ]),

            # ======================================================
            # TAB 3 — DEEP DIVE
            # ======================================================

            dcc.Tab(label="Deep Dive", children=[
                
                html.Div(style={'padding': '20px'}, children=[
                    
                    html.H2("Detailed Analysis", style={'color': COLORS['text']}),
                    
                    # Experiment selector
                    html.Div([
                        html.Label("Select Experiment:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                        dcc.Dropdown(
                            id='experiment-selector-3',
                            options=[
                                {'label': 'Conservative', 'value': 'conservative'},
                                {'label': 'Moderate', 'value': 'moderate'},
                                {'label': 'Aggressive', 'value': 'aggressive'}
                            ],
                            value='moderate',
                            style={'width': '200px', 'display': 'inline-block'}
                        )
                    ], style={'marginBottom': '20px'}),
                    
                    # Sample journey
                    html.Div([
                        dcc.Graph(id='sample-journey')
                    ], style={'marginBottom': '30px'}),
                    
                    # Layer competition
                    html.Div([
                        dcc.Graph(
                            figure=px.line(
                                df,
                                x="epoch",
                                y=["layer3_weight_norm", "projection_weight_norm"],
                                color="Experiment",
                                facet_col="Experiment",
                                title="⚖️ Layer Competition (Weight Norms)",
                                labels={'value': 'L2 Norm', 'variable': 'Layer'}
                            ).update_layout(height=400)
                        )
                    ], style={'marginBottom': '30px'}),
                    
                    # Activation drift
                    html.Div([
                        dcc.Graph(
                            figure=px.line(
                                df,
                                x="epoch",
                                y=["mean_sparsity", "mean_variance"],
                                color="Experiment",
                                facet_col="Experiment",
                                title="Activation Statistics Drift",
                                labels={'value': 'Value', 'variable': 'Metric'}
                            ).update_layout(height=400)
                        )
                    ], style={'marginBottom': '30px'}),
                    
                    # Risk tracking
                    html.Div([
                        dcc.Graph(
                            figure=px.line(
                                df,
                                x="epoch",
                                y="risk_percentile",
                                color="Experiment",
                                title="Risk Relaxation Over Time",
                                labels={'risk_percentile': 'Risk Percentile'}
                            ).update_layout(height=400)
                        )
                    ])
                ])
            ]),

            # ======================================================
            # TAB 4 — EFFICIENCY & STABILITY
            # ======================================================

            dcc.Tab(label="Efficiency & Stability", children=[
                
                html.Div(style={'padding': '20px'}, children=[
                    
                    html.H2("Compute Efficiency Analysis", style={'color': COLORS['text']}),
                    
                    # Experiment selector
                    html.Div([
                        html.Label("Select Experiment:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                        dcc.Dropdown(
                            id='experiment-selector-4',
                            options=[
                                {'label': 'Conservative', 'value': 'conservative'},
                                {'label': 'Moderate', 'value': 'moderate'},
                                {'label': 'Aggressive', 'value': 'aggressive'}
                            ],
                            value='moderate',
                            style={'width': '200px', 'display': 'inline-block'}
                        )
                    ], style={'marginBottom': '20px'}),
                    
                    # Compute waterfall
                    html.Div([
                        dcc.Graph(id='compute-waterfall')
                    ], style={'marginBottom': '30px'}),
                    
                    # Compute cost over time
                    html.Div([
                        dcc.Graph(
                            figure=px.line(
                                df,
                                x="epoch",
                                y="avg_compute_cost",
                                color="Experiment",
                                title="Compute Cost Evolution",
                                labels={'avg_compute_cost': 'Relative Compute Cost'}
                            ).update_layout(height=400)
                        )
                    ], style={'marginBottom': '30px'}),
                    
                    # Cumulative savings
                    html.Div([
                        dcc.Graph(
                            figure=px.line(
                                df,
                                x="epoch",
                                y="cumulative_savings",
                                color="Experiment",
                                title="Cumulative Compute Savings",
                                labels={'cumulative_savings': 'Cumulative Savings (Relative Units)'}
                            ).update_layout(height=400)
                        )
                    ], style={'marginBottom': '30px'}),
                    
                    # Routing stability
                    html.Div([
                        dcc.Graph(
                            figure=px.area(
                                df,
                                x="epoch",
                                y=["full_path_pct", "gated_path_pct", "skip_path_pct"],
                                color="Experiment",
                                facet_col="Experiment",
                                title="Routing Decision Stability",
                                labels={'value': 'Percentage (%)', 'variable': 'Path Type'}
                            ).update_layout(height=400)
                        )
                    ])
                ])
            ])
        ])
    ])
])


# ==========================================================
# CALLBACKS
# ==========================================================

@app.callback(
    Output('sankey-diagram', 'figure'),
    Input('experiment-selector', 'value')
)
def update_sankey(experiment):
    df_exp = df[df['Experiment'] == experiment]
    return create_sankey_by_curriculum(df_exp, experiment.title())


@app.callback(
    Output('decision-boundary', 'figure'),
    Input('experiment-selector-2', 'value')
)
def update_decision_boundary(experiment):
    df_exp = df[df['Experiment'] == experiment]
    return create_decision_boundary_scatter(df_exp, experiment.title())


@app.callback(
    Output('distribution-drift', 'figure'),
    Input('experiment-selector-2', 'value')
)
def update_distribution(experiment):
    df_exp = df[df['Experiment'] == experiment]
    return create_distribution_drift_animation(df_exp, experiment.title())


@app.callback(
    Output('sample-journey', 'figure'),
    Input('experiment-selector-3', 'value')
)
def update_sample_journey(experiment):
    df_exp = df[df['Experiment'] == experiment]
    return create_sample_journey_chart(df_exp, experiment.title())


@app.callback(
    Output('compute-waterfall', 'figure'),
    Input('experiment-selector-4', 'value')
)
def update_waterfall(experiment):
    df_exp = df[df['Experiment'] == experiment]
    return create_compute_waterfall(df_exp, experiment.title())


# ==========================================================
# RUN APP
# ==========================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ENHANCED ADAPTIVE COMPUTATION FRAMEWORK DASHBOARD")
    print("=" * 70)
    print("\nStarting server...")
    print("Navigate to: http://127.0.0.1:8050")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70)
    app.run(debug=True, port=8050)