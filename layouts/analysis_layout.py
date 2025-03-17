from dash import html, dcc, dash_table


def create_analysis_tab():
    """다중 샘플 분석 탭 레이아웃 생성"""
    return dcc.Tab(
        label='다중 샘플 분석',
        children=[
            html.Div([
                # 좌측 영역 - 색좌표 데이터 입력 및 결과 테이블
                html.Div([
                    create_multi_sample_input_section(),
                    create_results_section()
                ], style={'width': '56%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                # 우측 영역 - 통계 및 CIE 다이어그램
                html.Div([
                    create_stats_section(),
                    create_multi_cie_diagram_section()
                ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
            ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'})
        ],
        style={'padding': '0px'},
        selected_style={'padding': '0px', 'borderTop': '2px solid #3498db'}
    )


def create_multi_sample_input_section():
    """다중 샘플 입력 섹션 생성"""
    return html.Div([
        html.H3("다중 샘플 RGB 색좌표 입력",
                style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'color': '#2c3e50'}),

        # 초기화 버튼 추가
        html.Button('입력 초기화',
                    id='reset-multi-sample-inputs',
                    n_clicks=0,
                    style={'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                           'padding': '8px 15px', 'margin': '0 0 15px 0', 'borderRadius': '4px',
                           'float': 'right', 'fontSize': '14px'}),

        dcc.Textarea(
            id='multi-sample-data',
            placeholder='샘플 이름, Rx, Ry, Gx, Gy, Bx, By 형식으로 입력하세요. 예:\nSample1, 0.640, 0.330, 0.300, 0.600, 0.150, 0.060\nSample2, 0.625, 0.340, 0.290, 0.610, 0.155, 0.070\n...',
            style={'width': '100%', 'height': '200px', 'resize': 'vertical'}
        ),

        html.Button('분석 적용',
                    id='apply-multi-sample',
                    n_clicks=0,
                    style={'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                           'padding': '10px 20px', 'margin': '15px 0', 'borderRadius': '4px', 'width': '100%',
                           'fontSize': '16px', 'fontWeight': 'bold'}),

        html.Div([
            html.P("입력 형식: 샘플 이름, Rx, Ry, Gx, Gy, Bx, By",
                   style={'fontSize': '14px', 'color': '#7f8c8d', 'margin': '5px 0'}),
            html.P("예시: Sample1, 0.640, 0.330, 0.300, 0.600, 0.150, 0.060",
                   style={'fontSize': '14px', 'color': '#7f8c8d', 'margin': '5px 0'})
        ], style={'backgroundColor': '#f9f9f9', 'padding': '10px', 'borderRadius': '4px'}),
    ], style={'padding': '20px', 'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'borderRadius': '8px', 'margin': '0 0 20px 0'})


def create_results_section():
    """결과 섹션 생성"""
    return html.Div([
        html.H3("분석 결과", style={'margin': '20px 0 15px 0'}),

        # 다중 샘플 분석 테이블 스타일 수정 - DOE 분석 테이블과 일치시킴
        dash_table.DataTable(
            id='results-table',
            columns=[],
            data=[],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center',
                'padding': '5px',
                'minWidth': '60px',
                'width': '60px',
                'maxWidth': '100px',
                'whiteSpace': 'normal',
                'height': 'auto'
            },
            style_header={
                'backgroundColor': '#f8f9fa',
                'fontWeight': 'bold',
                'border': '1px solid #ddd'
            },
            style_data={
                'border': '1px solid #ddd'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f9f9f9'
                }
            ],
            sort_action='native',
            filter_action='native',
            page_action='none',
            editable=True,
            row_selectable=False,
            column_selectable=False
        ),

        html.Div(id='multi-sample-results')
    ], style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)'})


def create_stats_section():
    """통계 섹션 생성"""
    return html.Div([
        html.H3("통계 요약",
                style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'color': '#2c3e50'}),

        html.Div(id='multi-sample-stats')
    ], style={'padding': '20px', 'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'borderRadius': '8px', 'margin': '0 0 20px 0'})


def create_multi_cie_diagram_section():
    """다중 샘플 CIE 다이어그램 섹션 생성"""
    return html.Div([
        html.H3("CIE 색도 다이어그램",
                style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'color': '#2c3e50'}),

        # 색공간 선택 체크박스
        html.Div([
            html.H4("비교할 색공간 선택", style={'margin': '15px 0 10px 0'}),
            dcc.Checklist(
                id='multi-color-space-checklist',
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
                id='cie-diagram-type',
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
            id='multi-sample-cie-diagram',
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
