# ============================================================
#  app.py — Business Intelligence Dashboard
#  Run: python app.py → http://localhost:8050
# ============================================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
from sklearn.linear_model import LinearRegression
import warnings; warnings.filterwarnings('ignore')

# ── Load data ─────────────────────────────────────────────
df = pd.read_csv('data/processed/sales_clean.csv', parse_dates=['date'])
REGIONS = sorted(df['region'].unique())
COLORS = {'bg':'#0a0c14', 'card':'#111420', 'accent':'#4f8ef7'}

# ── App init ──────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "BI Dashboard"

def kpi_card(title, value, delta=None, color='#4f8ef7'):
    return html.Div([
        html.P(title, style={'color':'#6b7280', 'fontSize':'12px', 'margin':'0'}),
        html.H2(value, style={'color':color, 'margin':'6px 0 4px', 'fontSize':'28px'}),
        html.Span(delta if delta else "",
                  style={'color': '#10b981' if delta and '+' in str(delta) else '#ef4444',
                         'fontSize':'12px'})
    ], style={
        'background':'#111420', 'border':'1px solid #1e2438',
        'borderRadius':'10px', 'padding':'20px', 'flex':'1', 'minWidth':'180px'
    })

# ── Layout ────────────────────────────────────────────────
app.layout = html.Div(style={'backgroundColor':'#0a0c14','minHeight':'100vh','fontFamily':'monospace'},
children=[
    # Header
    html.Div([
        html.H1("📊 Business Intelligence Dashboard",
                style={'color':'#e8eaf6','margin':'0','fontSize':'22px'}),
        html.P("Executive Analytics Platform",
               style={'color':'#6b7280','margin':'4px 0 0','fontSize':'12px'})
    ], style={'padding':'24px 40px','borderBottom':'1px solid #1e2438'}),

    # Controls
    html.Div([
        html.Div([
            html.Label("Filter by Region", style={'color':'#6b7280','fontSize':'11px'}),
            dcc.Dropdown(
                id='region-filter',
                options=[{'label':'All Regions','value':'ALL'}] +
                        [{'label':r, 'value':r} for r in REGIONS],
                value='ALL', clearable=False,
                style={'backgroundColor':'#111420', 'color':'white'}
            )
        ], style={'width':'220px'}),
        html.Div([
            html.Label("Filter by Year", style={'color':'#6b7280','fontSize':'11px'}),
            dcc.Dropdown(
                id='year-filter',
                options=[{'label':'All Years','value':0}] +
                        [{'label':str(y),'value':y} for y in sorted(df['year'].unique())],
                value=0, clearable=False,
                style={'backgroundColor':'#111420', 'color':'white'}
            )
        ], style={'width':'160px'}),
    ], style={'display':'flex','gap':'20px','padding':'20px 40px','alignItems':'flex-end'}),

    # KPI Cards
    html.Div(id='kpi-section',
             style={'display':'flex','gap':'16px','padding':'0 40px 24px','flexWrap':'wrap'}),

    # Charts row 1
    html.Div([
        html.Div(dcc.Graph(id='revenue-trend'),
                 style={'flex':'2','background':'#111420','borderRadius':'10px','padding':'10px','border':'1px solid #1e2438'}),
        html.Div(dcc.Graph(id='category-pie'),
                 style={'flex':'1','background':'#111420','borderRadius':'10px','padding':'10px','border':'1px solid #1e2438'}),
    ], style={'display':'flex','gap':'16px','padding':'0 40px 16px'}),

    # Charts row 2
    html.Div([
        html.Div(dcc.Graph(id='region-bar'),
                 style={'flex':'1','background':'#111420','borderRadius':'10px','padding':'10px','border':'1px solid #1e2438'}),
        html.Div(dcc.Graph(id='forecast-chart'),
                 style={'flex':'2','background':'#111420','borderRadius':'10px','padding':'10px','border':'1px solid #1e2438'}),
    ], style={'display':'flex','gap':'16px','padding':'0 40px 40px'}),
])

# ── Callbacks ─────────────────────────────────────────────
@app.callback(
    Output('kpi-section','children'), Output('revenue-trend','figure'),
    Output('category-pie','figure'), Output('region-bar','figure'),
    Output('forecast-chart','figure'),
    Input('region-filter','value'), Input('year-filter','value')
)
def update_dashboard(region, year):
    filtered = df.copy()
    if region != 'ALL': filtered = filtered[filtered['region']==region]
    if year: filtered = filtered[filtered['year']==year]

    # KPIs
    tot_rev = filtered['revenue'].sum()
    tot_pft = filtered['profit'].sum()
    margin  = tot_pft / tot_rev * 100 if tot_rev > 0 else 0
    orders  = len(filtered)
    custs   = filtered['customer_id'].nunique()
    
    kpis = html.Div([
        kpi_card("Total Revenue", f"₹{tot_rev:,.0f}", "+8.2% YoY", '#4f8ef7'),
        kpi_card("Total Profit",  f"₹{tot_pft:,.0f}", color='#10b981'),
        kpi_card("Profit Margin", f"{margin:.1f}%", color='#f59e0b'),
        kpi_card("Total Orders", f"{orders:,}", color='#a78bfa'),
        kpi_card("Unique Customers", f"{custs:,}", color='#ef4444'),
    ], style={'display':'flex','gap':'16px','flexWrap':'wrap','width':'100%'})

    # Revenue trend chart
    monthly = filtered.groupby(pd.Grouper(key='date',freq='M'))['revenue'].sum().reset_index()
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly['date'], y=monthly['revenue'],
        fill='tozeroy', fillcolor='rgba(79,142,247,0.1)',
        line=dict(color='#4f8ef7', width=2), name='Revenue'
    ))
    fig_trend.update_layout(
        title='Monthly Revenue Trend', template='plotly_dark',
        height=320, margin=dict(l=20,r=20,t=40,b=20),
        paper_bgcolor='#111420', plot_bgcolor='#111420'
    )

    # Category pie
    cat_data = filtered.groupby('category')['revenue'].sum().reset_index()
    fig_pie  = px.pie(cat_data, names='category', values='revenue',
                      template='plotly_dark', title='Revenue by Category',
                      hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
    fig_pie.update_layout(height=320, margin=dict(l=20,r=20,t=40,b=20),
                         paper_bgcolor='#111420')

    # Region bar
    reg_data = filtered.groupby('region')['revenue'].sum().reset_index().sort_values('revenue')
    fig_reg  = px.bar(reg_data, x='revenue', y='region', orientation='h',
                      template='plotly_dark', title='Revenue by Region',
                      color='revenue', color_continuous_scale='Blues')
    fig_reg.update_layout(height=320, margin=dict(l=20,r=20,t=40,b=20),
                         paper_bgcolor='#111420', plot_bgcolor='#111420')

    # Forecast
    m2 = df.groupby(pd.Grouper(key='date',freq='MS'))['revenue'].sum().reset_index()
    m2['t'] = np.arange(len(m2))
    m2['s'] = np.sin(2*np.pi*m2['date'].dt.month/12)
    m2['c'] = np.cos(2*np.pi*m2['date'].dt.month/12)
    mdl = LinearRegression().fit(m2[['t','s','c']], m2['revenue'])
    fut = pd.date_range(m2['date'].max() + pd.DateOffset(months=1), periods=6, freq='MS')
    Xi  = pd.DataFrame({'t':np.arange(len(m2),len(m2)+6),'s':np.sin(2*np.pi*fut.month/12),'c':np.cos(2*np.pi*fut.month/12)})
    fcast = mdl.predict(Xi)
    std   = (m2['revenue'] - mdl.predict(m2[['t','s','c']])).std()

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=m2['date'],y=m2['revenue'],name='Actual',line=dict(color='#4f8ef7',width=2)))
    fig_fc.add_trace(go.Scatter(x=fut,y=fcast,name='Forecast',line=dict(color='#10b981',width=2,dash='dash')))
    fig_fc.add_trace(go.Scatter(
        x=list(fut)+list(fut[::-1]), y=list(fcast+1.96*std)+list((fcast-1.96*std)[::-1]),
        fill='toself', fillcolor='rgba(16,185,129,0.1)', line=dict(color='rgba(0,0,0,0)'), name='95% CI'
    ))
    fig_fc.update_layout(
        title='6-Month Revenue Forecast', template='plotly_dark',
        height=320, margin=dict(l=20,r=20,t=40,b=20),
        paper_bgcolor='#111420', plot_bgcolor='#111420',
        legend=dict(orientation='h', y=-0.2)
    )

    return kpis, fig_trend, fig_pie, fig_reg, fig_fc

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)