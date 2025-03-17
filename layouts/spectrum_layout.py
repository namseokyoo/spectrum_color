from dash import html, dcc


def create_spectrum_tab():
    """스펙트럼 분석 탭 레이아웃 생성"""
    return dcc.Tab(
        label='스펙트럼 분석',
        children=[
            html.Div([
                html.H3("스펙트럼 분석",
                        style={'textAlign': 'center', 'margin': '20px 0', 'color': '#2c3e50', 'fontWeight': 'bold'}),
                
                # 스펙트럼 데이터 입력 섹션
                html.Div([
                    html.Div([
                        html.Label("스펙트럼 데이터:",
                                  style={'fontWeight': 'bold', 'marginTop': '10px'}),
                        dcc.Textarea(
                            id='spectrum-data',
                            placeholder='파장(nm), 강도 형식으로 데이터를 입력하세요...',
                            style={'width': '100%', 'height': '200px', 'marginBottom': '10px'}
                        ),
                        html.Button('적용',
                                   id='apply-spectrum',
                                   n_clicks=0,
                                   style={'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                                          'padding': '10px 20px', 'marginTop': '10px', 'borderRadius': '4px'})
                    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    # 스펙트럼 그래프
                    html.Div([
                        dcc.Graph(
                            id='spectrum-graph',
                            style={'height': '500px'}
                        )
                    ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '5%'})
                ], style={'margin': '20px 0'})
            ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'})
        ],
        style={'padding': '0px'},
        selected_style={'padding': '0px', 'borderTop': '2px solid #3498db'}
    ) 