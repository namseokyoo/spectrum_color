from dash import html, dcc, dash_table


def create_doe_tab():
    """DOE 탭 레이아웃 생성"""
    return dcc.Tab(
        label='DOE 분석',
        children=[
            html.Div([
                # 좌측 영역 - 색좌표 범위 입력 및 결과 테이블
                html.Div([
                    create_doe_input_section(),
                    create_results_section()
                ], style={'width': '56%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                # 우측 영역 - 통계, CIE 다이어그램, 3D 그래프
                html.Div([
                    create_stats_section(),
                    create_doe_cie_diagram_section(),
                    create_doe_3d_graph_section()  # 새로운 3D 그래프 섹션 추가
                ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
            ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'})
        ],
        style={'padding': '0px'},
        selected_style={'padding': '0px', 'borderTop': '2px solid #3498db'}
    )


def create_doe_input_section():
    """DOE 입력 섹션 생성"""
    return html.Div([

        html.Div([
            html.H3("RGB 색좌표 범위 입력",
                    style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'color': '#2c3e50', 'marginBottom': '20px'}),

            # 초기화 버튼 추가
            html.Button('입력 초기화',
                        id='reset-doe-inputs',
                        n_clicks=0,
                        style={'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                               'padding': '8px 8px', 'margin': '0 0 10px 20px', 'borderRadius': '4px',
                               'display': 'inline-block', 'fontSize': '10px'})
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between'}),





        # 입력 테이블 형식으로 변경
        html.Table([
            # 헤더 행
            html.Tr([
                html.Th("색상", style={'width': '10%', 'textAlign': 'center',
                        'padding': '8px', 'backgroundColor': '#f2f2f2'}),
                html.Th("최소값 (x,y)", style={
                        'width': '35%', 'textAlign': 'center', 'padding': '8px', 'backgroundColor': '#f2f2f2'}),
                html.Th("최대값 (x,y)", style={
                        'width': '35%', 'textAlign': 'center', 'padding': '8px', 'backgroundColor': '#f2f2f2'}),
                html.Th("분할 수", style={
                        'width': '20%', 'textAlign': 'center', 'padding': '8px', 'backgroundColor': '#f2f2f2'})
            ]),

            # R 행
            html.Tr([
                html.Td("R", style={'textAlign': 'center', 'padding': '8px',
                        'backgroundColor': '#ffebee', 'color': '#e74c3c', 'fontWeight': 'bold'}),
                html.Td([
                    dcc.Input(
                        id='r-min-x',
                        type='number',
                        placeholder='x 최소값',
                        min=0,
                        max=1,
                        step=0.001,
                        style={'width': '48%', 'marginRight': '4%'}
                    ),
                    dcc.Input(
                        id='r-min-y',
                        type='number',
                        placeholder='y 최소값',
                        min=0,
                        max=1,
                        step=0.001,
                        style={'width': '48%'}
                    )
                ], style={'padding': '8px'}),
                html.Td([
                    dcc.Input(
                        id='r-max-x',
                        type='number',
                        placeholder='x 최대값',
                        min=0,
                        max=1,
                        step=0.001,
                        style={'width': '48%', 'marginRight': '4%'}
                    ),
                    dcc.Input(
                        id='r-max-y',
                        type='number',
                        placeholder='y 최대값',
                        min=0,
                        max=1,
                        step=0.001,
                        style={'width': '48%'}
                    )
                ], style={'padding': '8px'}),
                html.Td([
                    dcc.Input(
                        id='r-divisions',
                        type='number',
                        placeholder='분할 수',
                        min=1,
                        value=5,
                        style={'width': '100%', 'height': '36px', 'borderRadius': '4px',
                               'padding': '8px', 'boxSizing': 'border-box'}
                    )
                ], style={'padding': '8px'})
            ]),

            # G 행
            html.Tr([
                html.Td("G", style={'textAlign': 'center', 'padding': '8px',
                        'backgroundColor': '#e8f5e9', 'color': '#27ae60', 'fontWeight': 'bold'}),
                html.Td([
                    dcc.Input(
                        id='g-min-x',
                        type='number',
                        placeholder='x 최소값',
                        min=0,
                        max=1,
                        step=0.001,
                        style={'width': '48%', 'marginRight': '4%'}
                    ),
                    dcc.Input(
                        id='g-min-y',
                        type='number',
                        placeholder='y 최소값',
                        min=0,
                        max=1,
                        step=0.001,
                        style={'width': '48%'}
                    )
                ], style={'padding': '8px'}),
                html.Td([
                    dcc.Input(
                        id='g-max-x',
                        type='number',
                        placeholder='x 최대값',
                        min=0,
                        max=1,
                        step=0.001,
                        style={'width': '48%', 'marginRight': '4%'}
                    ),
                    dcc.Input(
                        id='g-max-y',
                        type='number',
                        placeholder='y 최대값',
                        min=0,
                        max=1,
                        step=0.001,
                        style={'width': '48%'}
                    )
                ], style={'padding': '8px'}),
                html.Td([
                    dcc.Input(
                        id='g-divisions',
                        type='number',
                        placeholder='분할 수',
                        min=1,
                        value=5,
                        style={'width': '100%', 'height': '36px', 'borderRadius': '4px',
                               'padding': '8px', 'boxSizing': 'border-box'}
                    )
                ], style={'padding': '8px'})
            ]),

            # B 행
            html.Tr([
                html.Td("B", style={'textAlign': 'center', 'padding': '8px',
                        'backgroundColor': '#e3f2fd', 'color': '#3498db', 'fontWeight': 'bold'}),
                html.Td([
                    dcc.Input(
                        id='b-min-x',
                        type='number',
                        placeholder='x 최소값',
                        min=0,
                        max=1,
                        step=0.001,
                        style={'width': '48%', 'marginRight': '4%'}
                    ),
                    dcc.Input(
                        id='b-min-y',
                        type='number',
                        placeholder='y 최소값',
                        min=0,
                        max=1,
                        step=0.001,
                        style={'width': '48%'}
                    )
                ], style={'padding': '8px'}),
                html.Td([
                    dcc.Input(
                        id='b-max-x',
                        type='number',
                        placeholder='x 최대값',
                        min=0,
                        max=1,
                        step=0.001,
                        style={'width': '48%', 'marginRight': '4%'}
                    ),
                    dcc.Input(
                        id='b-max-y',
                        type='number',
                        placeholder='y 최대값',
                        min=0,
                        max=1,
                        step=0.001,
                        style={'width': '48%'}
                    )
                ], style={'padding': '8px'}),
                html.Td([
                    dcc.Input(
                        id='b-divisions',
                        type='number',
                        placeholder='분할 수',
                        min=1,
                        value=5,
                        style={'width': '100%', 'height': '36px', 'borderRadius': '4px',
                               'padding': '8px', 'boxSizing': 'border-box'}
                    )
                ], style={'padding': '8px'})
            ])
        ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginBottom': '20px'}),

        html.Button('분석 적용',
                    id='apply-doe',
                    n_clicks=0,
                    style={'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                           'padding': '10px 20px', 'margin': '15px 0', 'borderRadius': '4px', 'width': '100%',
                           'fontSize': '16px', 'fontWeight': 'bold'}),

        html.Div([
            html.P("입력 형식: 각 색상별로 최소/최대 x,y 좌표와 분할 수를 입력하세요.",
                   style={'fontSize': '14px', 'color': '#7f8c8d', 'margin': '5px 0'}),
            html.P("예시: R(0.68,0.32) ~ (0.70,0.30), 분할 수 10",
                   style={'fontSize': '14px', 'color': '#7f8c8d', 'margin': '5px 0'})
        ], style={'backgroundColor': '#f9f9f9', 'padding': '10px', 'borderRadius': '4px'})
    ], style={'padding': '20px', 'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'borderRadius': '8px', 'margin': '0 0 20px 0'})


def create_results_section():
    """결과 섹션 생성"""
    return html.Div([
        html.H3("분석 결과", style={'margin': '20px 0 15px 0'}),

        # 빈 결과 테이블 (초기 상태)
        dash_table.DataTable(
            id='doe-results-table',
            columns=[],
            data=[],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_header={
                'backgroundColor': '#f2f2f2',
                'color': '#666666',
                'fontWeight': 'normal',
                'textAlign': 'center',
                'padding': '5px',
                'height': '30px'
            },
            sort_action='native',
            sort_mode='single',
            editable=True,
            row_selectable=False,
            column_selectable=False,
            page_size=20,  # 페이지당 20개 행 표시
            page_action='native'  # 페이지네이션 활성화
        ),

        html.Div(id='doe-results-info')
    ], style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)'})


def create_stats_section():
    """통계 섹션 생성"""
    return html.Div([
        html.H3("통계 요약",
                style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'color': '#2c3e50'}),

        html.Div(id='doe-stats')
    ], style={'padding': '20px', 'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'borderRadius': '8px', 'margin': '0 0 20px 0'})


def create_doe_cie_diagram_section():
    """DOE CIE 다이어그램 섹션 생성"""
    return html.Div([
        html.H3("CIE 색도 다이어그램",
                style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'color': '#2c3e50'}),

        # 색공간 선택 체크박스
        html.Div([
            html.H4("비교할 색공간 선택", style={'margin': '15px 0 10px 0'}),
            dcc.Checklist(
                id='doe-color-space-checklist',
                options=[
                    {'label': ' DCI-P3', 'value': 'DCI-P3'},
                    {'label': ' Adobe RGB', 'value': 'adobeRGB'},
                    {'label': ' BT.2020', 'value': 'BT2020'}
                ],
                value=['DCI-P3', 'adobeRGB', 'BT2020'],
                inline=True,
                labelStyle={'marginRight': '20px', 'fontSize': '14px'},
                style={'padding': '0 0 15px 0'}
            ),
        ]),

        # 다이어그램 유형 선택
        html.Div([
            html.H4("다이어그램 유형", style={'margin': '15px 0 10px 0'}),
            dcc.RadioItems(
                id='doe-diagram-type',
                options=[
                    {'label': ' CIE 1931 xy', 'value': 'xy'},
                    {'label': ' CIE 1976 u\'v\'', 'value': 'uv'}
                ],
                value='xy',
                inline=True,
                labelStyle={'marginRight': '20px', 'fontSize': '14px'},
                style={'padding': '0 0 15px 0'}
            ),
        ]),

        # 다이어그램 그래프
        dcc.Graph(
            id='doe-cie-diagram',
            style={
                'height': '500px',
                'width': '100%',
                'resize': 'both',
                'overflow': 'auto',
                'minHeight': '400px',
                'minWidth': '300px',
                'maxHeight': '800px',
                'border': '1px dashed #ccc',
                'padding': '10px'
            }
        )
    ], style={'padding': '20px', 'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'borderRadius': '8px'})


def create_doe_3d_graph_section():
    """DOE 3D 그래프 섹션 생성"""
    return html.Div([
        html.H3("3D 색재현율 시각화",
                style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'color': '#2c3e50'}),

        # 시각화 옵션 선택
        html.Div([
            # 색공간 선택
            html.Div([
                html.H4("색공간 선택", style={'margin': '15px 0 10px 0'}),
                dcc.RadioItems(
                    id='doe-3d-colorspace',
                    options=[
                        {'label': ' DCI-P3', 'value': 'DCI-P3'},
                        {'label': ' Adobe RGB', 'value': 'adobeRGB'},
                        {'label': ' BT.2020', 'value': 'BT2020'}
                    ],
                    value='DCI-P3',
                    inline=True,
                    labelStyle={'marginRight': '20px', 'fontSize': '14px'},
                    style={'padding': '0 0 15px 0'}
                ),
            ]),

            # 측정 항목 선택
            html.Div([
                html.H4("측정 항목 선택", style={'margin': '15px 0 10px 0'}),
                dcc.RadioItems(
                    id='doe-3d-metric',
                    options=[
                        {'label': ' 중첩비', 'value': 'coverage'},
                        {'label': ' 면적비', 'value': 'area'}
                    ],
                    value='coverage',
                    inline=True,
                    labelStyle={'marginRight': '20px', 'fontSize': '14px'},
                    style={'padding': '0 0 15px 0'}
                ),
            ]),

            # 좌표계 선택
            html.Div([
                html.H4("좌표계 선택", style={'margin': '15px 0 10px 0'}),
                dcc.RadioItems(
                    id='doe-3d-coordinate',
                    options=[
                        {'label': ' CIE 1931 xy', 'value': 'xy'},
                        {'label': ' CIE 1976 u\'v\'', 'value': 'uv'}
                    ],
                    value='xy',
                    inline=True,
                    labelStyle={'marginRight': '20px', 'fontSize': '14px'},
                    style={'padding': '0 0 15px 0'}
                ),
            ]),
        ]),

        # 3D 그래프
        dcc.Graph(
            id='doe-3d-graph',
            style={
                'height': '500px',
                'width': '100%',
                'resize': 'both',
                'overflow': 'auto',
                'minHeight': '400px',
                'minWidth': '300px',
                'maxHeight': '800px',
                'border': '1px dashed #ccc',
                'padding': '10px'
            }
        )
    ], style={'padding': '20px', 'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'borderRadius': '8px', 'marginTop': '20px'})
