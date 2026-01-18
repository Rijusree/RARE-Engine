import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

print("📥 Loading RARE Engine data...")
df = pd.read_csv("fused_rare_engine1_2025.csv")
df = df.fillna(0)

# Ensure required columns exist
if 'ghost_gap_risk_proxy' not in df.columns:
    df['ghost_gap_risk_proxy'] = 0
if 'infrastructure_gap_flag' not in df.columns:
    df['infrastructure_gap_flag'] = 0
if 'Population_2025_Millions' not in df.columns:
    df['Population_2025_Millions'] = 1

# Create risk categories for better visualization
df['risk_category'] = df.apply(lambda x: 
    '🚨 Ghost-Gap Risk' if x['ghost_gap_risk_proxy'] == 1 
    else '⚠️ Infrastructure Gap' if x['infrastructure_gap_flag'] == 1 
    else '✅ Stable', axis=1)

print(f"✅ Loaded {len(df)} states/UTs\n")

# ============================================================================
# INTERACTIVE VISUALIZATION 1: GHOST-GAP RISK BUBBLE CHART
# ============================================================================
print("📊 [1/3] Creating Interactive Ghost-Gap Risk Bubble Chart...")

# Create hover text
df['hover_text'] = df.apply(lambda x: 
    f"<b>{x['state']}</b><br>" +
    f"UPI per Capita: {x['upi_per_capita']:.3f}<br>" +
    f"MBU per Capita: {x['mbu_per_capita']:.6f}<br>" +
    f"Population: {x['Population_2025_Millions']:.2f}M<br>" +
    f"Ghost-Gap Risk: {x['ghost_gap_risk_score']:.3f}<br>" +
    f"Digital Maturity: {x['dms']:.3f}<br>" +
    f"Status: {x['risk_category']}", axis=1)

# Color mapping
color_map = {
    '🚨 Ghost-Gap Risk': '#d32f2f',
    '⚠️ Infrastructure Gap': '#ff9800',
    '✅ Stable': '#4caf50'
}

fig1 = px.scatter(
    df,
    x='upi_per_capita',
    y='mbu_per_capita',
    size='Population_2025_Millions',
    color='risk_category',
    color_discrete_map=color_map,
    hover_name='state',
    hover_data={
        'upi_per_capita': ':.3f',
        'mbu_per_capita': ':.6f',
        'ghost_gap_risk_score': ':.3f',
        'dms': ':.3f',
        'Population_2025_Millions': ':.2f',
        'risk_category': False
    },
    title='<b>🔮 RARE Engine: Interactive Ghost-Gap Risk Analysis</b><br><sub>Hover over bubbles for details | Click legend to filter</sub>',
    labels={
        'upi_per_capita': 'UPI Transactions per Capita (Monthly)',
        'mbu_per_capita': 'MBU Infrastructure Capacity per Capita',
        'risk_category': 'Risk Status'
    },
    size_max=60
)

# Add quadrant lines
upi_median = df['upi_per_capita'].median()
infra_median = df['mbu_per_capita'].median()

fig1.add_hline(y=infra_median, line_dash="dash", line_color="gray", 
               annotation_text="Infrastructure Median", annotation_position="right")
fig1.add_vline(x=upi_median, line_dash="dash", line_color="gray",
               annotation_text="UPI Median", annotation_position="top")

# Add quadrant annotations
fig1.add_annotation(
    x=df['upi_per_capita'].max() * 0.8,
    y=df['mbu_per_capita'].max() * 0.9,
    text="<b>✅ DIGITAL LEADERS</b><br>High UPI + Strong Infra",
    showarrow=False,
    bgcolor="#c8e6c9",
    bordercolor="#4caf50",
    borderwidth=2,
    font=dict(size=12, color="black")
)

fig1.add_annotation(
    x=df['upi_per_capita'].max() * 0.8,
    y=df['mbu_per_capita'].min() * 2,
    text="<b>🚨 GHOST-GAP ZONE</b><br>High UPI + Weak Infra<br>⚡ PRIORITY",
    showarrow=False,
    bgcolor="#ffcdd2",
    bordercolor="#d32f2f",
    borderwidth=2,
    font=dict(size=12, color="black")
)

fig1.update_layout(
    height=700,
    font=dict(size=12),
    hovermode='closest',
    plot_bgcolor='white',
    paper_bgcolor='white',
    legend=dict(
        title=dict(text='<b>Risk Status</b>', font=dict(size=14)),
        font=dict(size=12),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='black',
        borderwidth=2
    )
)

fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

fig1.write_html("01_interactive_ghost_gap_analysis.html")
print("   ✅ Saved: 01_interactive_ghost_gap_analysis.html\n")

# ============================================================================
# INTERACTIVE VISUALIZATION 2: COMPREHENSIVE DASHBOARD
# ============================================================================
print("📊 [2/3] Creating Interactive Comprehensive Dashboard...")

# Create subplot figure
fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        '<b>Top 15 Infrastructure Gaps</b>',
        '<b>Digital Maturity Distribution</b>',
        '<b>Intervention Priority (Top 20)</b>',
        '<b>Risk Flags Summary</b>'
    ),
    specs=[
        [{"type": "bar"}, {"type": "pie"}],
        [{"type": "bar"}, {"type": "bar"}]
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.15
)

# SUBPLOT 1: Infrastructure Gap Ranking
top_gaps = df.nlargest(15, 'infrastructure_gap_score').sort_values('infrastructure_gap_score')
colors_gaps = ['#d32f2f' if x == 1 else '#ff9800' for x in top_gaps['infrastructure_gap_flag']]

fig2.add_trace(
    go.Bar(
        y=top_gaps['state'],
        x=top_gaps['infrastructure_gap_score'],
        orientation='h',
        marker=dict(color=colors_gaps, line=dict(color='black', width=1)),
        text=top_gaps['infrastructure_gap_score'].round(3),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Gap Score: %{x:.3f}<extra></extra>',
        name='Infrastructure Gap'
    ),
    row=1, col=1
)

# SUBPLOT 2: Digital Maturity Pie Chart
bins = [0, 0.3, 0.5, 0.7, 1.0]
labels_dms = ['Low (0-0.3)', 'Medium (0.3-0.5)', 'High (0.5-0.7)', 'Very High (0.7-1.0)']
df['dms_category'] = pd.cut(df['dms'], bins=bins, labels=labels_dms, include_lowest=True)
dms_counts = df['dms_category'].value_counts().sort_index()

fig2.add_trace(
    go.Pie(
        labels=dms_counts.index,
        values=dms_counts.values,
        marker=dict(colors=['#f44336', '#ff9800', '#4caf50', '#2196f3']),
        hovertemplate='<b>%{label}</b><br>States: %{value}<br>Percentage: %{percent}<extra></extra>',
        textinfo='percent+label',
        textfont=dict(size=11)
    ),
    row=1, col=2
)

# SUBPLOT 3: Intervention Priority
top_priority = df.nlargest(20, 'intervention_priority').sort_values('intervention_priority')
priority_colors = px.colors.sequential.Reds[2:] * 4  # Gradient

fig2.add_trace(
    go.Bar(
        y=top_priority['state'],
        x=top_priority['intervention_priority'],
        orientation='h',
        marker=dict(
            color=top_priority['intervention_priority'],
            colorscale='Reds',
            line=dict(color='black', width=1),
            showscale=False
        ),
        text=top_priority['intervention_priority'].round(3),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Priority: %{x:.3f}<extra></extra>',
        name='Priority'
    ),
    row=2, col=1
)

# SUBPLOT 4: Risk Flags Comparison
ghost_count = df['ghost_gap_risk_proxy'].sum()
infra_count = df['infrastructure_gap_flag'].sum()
total = len(df)

categories = ['Ghost-Gap<br>Risk', 'Infrastructure<br>Gap']
flagged_counts = [ghost_count, infra_count]
normal_counts = [total - ghost_count, total - infra_count]

fig2.add_trace(
    go.Bar(
        x=categories,
        y=flagged_counts,
        name='⚠️ Flagged',
        marker=dict(color='#d32f2f', line=dict(color='black', width=1.5)),
        text=[f'{x}<br>({x/total*100:.1f}%)' for x in flagged_counts],
        textposition='inside',
        hovertemplate='<b>%{x}</b><br>Flagged: %{y}<extra></extra>'
    ),
    row=2, col=2
)

fig2.add_trace(
    go.Bar(
        x=categories,
        y=normal_counts,
        name='✅ Normal',
        marker=dict(color='#4caf50', line=dict(color='black', width=1.5)),
        text=[f'{x}<br>({x/total*100:.1f}%)' for x in normal_counts],
        textposition='inside',
        hovertemplate='<b>%{x}</b><br>Normal: %{y}<extra></extra>'
    ),
    row=2, col=2
)

# Update layout
fig2.update_layout(
    height=900,
    showlegend=True,
    title_text='<b>🎯 RARE Engine: Comprehensive Analytics Dashboard</b><br><sub>Interactive multi-metric analysis</sub>',
    title_font=dict(size=20),
    font=dict(size=11),
    plot_bgcolor='white',
    paper_bgcolor='white',
    barmode='stack',
    legend=dict(
        font=dict(size=11),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='black',
        borderwidth=1
    )
)

# Update axes
fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

fig2.write_html("02_interactive_comprehensive_dashboard.html")
print("   ✅ Saved: 02_interactive_comprehensive_dashboard.html\n")

# ============================================================================
# INTERACTIVE VISUALIZATION 3: STATE-BY-STATE COMPARISON
# ============================================================================
print("📊 [3/3] Creating Interactive State Comparison Tool...")

# Prepare data for parallel coordinates
df_sorted = df.sort_values('intervention_priority', ascending=False).head(25)

# Normalize scores for better visualization
df_sorted['upi_norm'] = (df_sorted['upi_per_capita'] - df_sorted['upi_per_capita'].min()) / \
                         (df_sorted['upi_per_capita'].max() - df_sorted['upi_per_capita'].min())
df_sorted['infra_norm'] = (df_sorted['mbu_per_capita'] - df_sorted['mbu_per_capita'].min()) / \
                           (df_sorted['mbu_per_capita'].max() - df_sorted['mbu_per_capita'].min())

fig3 = go.Figure(data=
    go.Parcoords(
        line=dict(
            color=df_sorted['intervention_priority'],
            colorscale='RdYlGn_r',
            showscale=True,
            cmin=0,
            cmax=1,
            colorbar=dict(
                title="Intervention<br>Priority",
                thickness=20,
                len=0.7
            )
        ),
        dimensions=[
            dict(
                range=[0, 1],
                label='<b>UPI Adoption</b>',
                values=df_sorted['upi_norm'],
                ticktext=['Low', 'Medium', 'High'],
                tickvals=[0, 0.5, 1]
            ),
            dict(
                range=[0, 1],
                label='<b>Infrastructure</b>',
                values=df_sorted['infra_norm'],
                ticktext=['Weak', 'Moderate', 'Strong'],
                tickvals=[0, 0.5, 1]
            ),
            dict(
                range=[0, 1],
                label='<b>Ghost-Gap Risk</b>',
                values=df_sorted['ghost_gap_risk_score'],
                ticktext=['Low', 'Medium', 'High'],
                tickvals=[0, 0.5, 1]
            ),
            dict(
                range=[0, 1],
                label='<b>Infrastructure Gap</b>',
                values=df_sorted['infrastructure_gap_score'],
                ticktext=['Low', 'Medium', 'High'],
                tickvals=[0, 0.5, 1]
            ),
            dict(
                range=[0, 1],
                label='<b>Digital Maturity</b>',
                values=df_sorted['dms'],
                ticktext=['Low', 'Medium', 'High'],
                tickvals=[0, 0.5, 1]
            ),
            dict(
                range=[0, 1],
                label='<b>Priority</b>',
                values=df_sorted['intervention_priority'],
                ticktext=['Low', 'Medium', 'High'],
                tickvals=[0, 0.5, 1]
            )
        ],
        labelfont=dict(size=13, color='black'),
        tickfont=dict(size=11, color='black'),
        rangefont=dict(size=11, color='black')
    )
)

fig3.update_layout(
    title='<b>🔍 Interactive State-by-State Comparison: Top 25 Priority States</b><br>' +
          '<sub>Drag the vertical axes to filter | Each line represents one state | Color = Intervention Priority</sub>',
    title_font=dict(size=18),
    height=600,
    font=dict(size=12),
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=100, r=100, t=120, b=50)
)

fig3.write_html("03_interactive_state_comparison.html")
print("   ✅ Saved: 03_interactive_state_comparison.html\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ ALL INTERACTIVE VISUALIZATIONS COMPLETED!")
print("="*80)
print("\n📁 Generated Interactive HTML Files:")
print("   1. 01_interactive_ghost_gap_analysis.html")
print("      → Interactive bubble chart with hover details and filtering")
print("   2. 02_interactive_comprehensive_dashboard.html")
print("      → 4-panel dashboard with rankings, distributions, and comparisons")
print("   3. 03_interactive_state_comparison.html")
print("      → Parallel coordinates plot for multi-metric state comparison")
print("\n🎯 Features:")
print("   ✓ Hover for detailed information")
print("   ✓ Click legend items to show/hide categories")
print("   ✓ Zoom and pan capabilities")
print("   ✓ Export to PNG option (camera icon)")
print("   ✓ Fully interactive in web browser")
print("\n💡 Open the HTML files in any web browser for full interactivity!")
print("="*80)