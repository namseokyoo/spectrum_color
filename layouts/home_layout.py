from dash import html, dcc


def create_spectrum_tab():
    """스펙트럼 분석 탭 레이아웃 생성"""
    return dcc.Tab(
        label='스펙트럼 분석',
        children=[
            # 데이터 저장을 위한 컴포넌트 추가 - storage_type을 명시적으로 'local'로 설정
            dcc.Store(
                id='spectrum-data-store',
                storage_type='local',
                clear_data=False  # 명시적으로 데이터를 지우지 않도록 설정
            ),

            html.Div([
                # 좌측 영역 - 스펙트럼 데이터 입력 및 스펙트럼 그래프
                html.Div([
                    create_spectrum_input_section(),
                    create_spectrum_graph_section()
                ], style={'width': '56%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                # 우측 영역 - 색공간 관련 수치와 그래프
                html.Div([
                    create_gamut_results_section(),
                    create_cie_diagram_section()
                ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
            ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'})
        ],
        style={'padding': '0px'},
        selected_style={'padding': '0px', 'borderTop': '2px solid #3498db'}
    )


def create_spectrum_input_section():
    """스펙트럼 입력 섹션 생성"""
    return html.Div([
        html.Div([
            html.Div([
                html.H3("스펙트럼 데이터 입력",
                        style={'display': 'inline-block', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'color': '#2c3e50', 'margin': '0'}),
                html.Button('입력 초기화',
                            id='reset-spectrum-inputs',
                            n_clicks=0,
                            style={'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                                   'padding': '8px 8px', 'margin': '0 0 10px 20px', 'borderRadius': '4px',
                                   'display': 'inline-block', 'fontSize': '10px'})
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between'}),

            # 스펙트럼 입력 탭 추가
            dcc.Tabs([
                # RGB 스펙트럼 입력 탭
                dcc.Tab(label='EL Spectrum', children=[
                    # RGB 입력 컨테이너 (가로 배열)
                    html.Div([
                        # R 스펙트럼 입력
                        html.Div([
                            html.H4("Red", style={
                                    'color': '#e74c3c', 'marginBottom': '10px', 'textAlign': 'center'}),
                            dcc.Textarea(
                                id='input-r-data',
                                placeholder='파장(nm) 강도(a.u.) 형식으로 입력하세요. 예:\n400 0.1\n450 0.3\n500 0.5\n...',
                                style={'width': '100%', 'height': '200px',
                                       'resize': 'vertical', 'borderColor': '#e74c3c'}
                            ),
                            html.Div(id='r-coordinates-display', style={
                                'marginTop': '10px',
                                'padding': '10px',
                                'backgroundColor': '#f9f9f9',
                                'borderRadius': '4px',
                                'fontSize': '13px',
                                'border': '1px solid #e74c3c'
                            })
                        ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                        # G 스펙트럼 입력
                        html.Div([
                            html.H4("Green", style={
                                    'color': '#27ae60', 'marginBottom': '10px', 'textAlign': 'center'}),
                            dcc.Textarea(
                                id='input-g-data',
                                placeholder='파장(nm) 강도(a.u.) 형식으로 입력하세요. 예:\n400 0.1\n450 0.3\n500 0.5\n...',
                                style={'width': '100%', 'height': '200px',
                                       'resize': 'vertical', 'borderColor': '#27ae60'}
                            ),
                            html.Div(id='g-coordinates-display', style={
                                'marginTop': '10px',
                                'padding': '10px',
                                'backgroundColor': '#f9f9f9',
                                'borderRadius': '4px',
                                'fontSize': '13px',
                                'border': '1px solid #27ae60'
                            })
                        ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'}),

                        # B 스펙트럼 입력
                        html.Div([
                            html.H4("Blue", style={
                                    'color': '#3498db', 'marginBottom': '10px', 'textAlign': 'center'}),
                            dcc.Textarea(
                                id='input-b-data',
                                placeholder='파장(nm) 강도(a.u.) 형식으로 입력하세요. 예:\n400 0.1\n450 0.3\n500 0.5\n...',
                                style={'width': '100%', 'height': '200px',
                                       'resize': 'vertical', 'borderColor': '#3498db'}
                            ),
                            html.Div(id='b-coordinates-display', style={
                                'marginTop': '10px',
                                'padding': '10px',
                                'backgroundColor': '#f9f9f9',
                                'borderRadius': '4px',
                                'fontSize': '13px',
                                'border': '1px solid #3498db'
                            })
                        ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '10px'})
                ]),

                # 컬러 필터 스펙트럼 입력 탭
                dcc.Tab(label='Color Filter Spectrum', children=[
                    # RGB 필터 입력 컨테이너 (가로 배열)
                    html.Div([
                        # R 필터 스펙트럼 입력
                        html.Div([
                            html.H4("Red", style={
                                    'color': '#e74c3c', 'marginBottom': '10px', 'textAlign': 'center'}),
                            dcc.Textarea(
                                id='input-r-filter-data',
                                placeholder='파장(nm) 강도(a.u.) 형식으로 입력하세요. 예:\n400 0.1\n450 0.3\n500 0.5\n...',
                                style={'width': '100%', 'height': '200px',
                                       'resize': 'vertical', 'borderColor': '#e74c3c'}
                            )
                        ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                        # G 필터 스펙트럼 입력
                        html.Div([
                            html.H4("Green", style={
                                    'color': '#27ae60', 'marginBottom': '10px', 'textAlign': 'center'}),
                            dcc.Textarea(
                                id='input-g-filter-data',
                                placeholder='파장(nm) 강도(a.u.) 형식으로 입력하세요. 예:\n400 0.1\n450 0.3\n500 0.5\n...',
                                style={'width': '100%', 'height': '200px',
                                       'resize': 'vertical', 'borderColor': '#27ae60'}
                            )
                        ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'}),

                        # B 필터 스펙트럼 입력
                        html.Div([
                            html.H4("Blue", style={
                                    'color': '#3498db', 'marginBottom': '10px', 'textAlign': 'center'}),
                            dcc.Textarea(
                                id='input-b-filter-data',
                                placeholder='파장(nm) 강도(a.u.) 형식으로 입력하세요. 예:\n400 0.1\n450 0.3\n500 0.5\n...',
                                style={'width': '100%', 'height': '200px',
                                       'resize': 'vertical', 'borderColor': '#3498db'}
                            )
                        ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '10px'})
                ])
            ], id='spectrum-input-tabs', style={'marginTop': '1px',
                                                'height': '20px',  # 탭 높이 증가
                                                'lineHeight': '5px',  # 텍스트 세로 중앙 정렬을 위한 라인 높이 설정
                                                'fontSize': '14px',
                                                })
        ]),



        # 컬러 필터 모드 체크박스
        html.Div([
            dcc.Checklist(
                id='color-filter-mode',
                options=[
                    {'label': ' Color Filter Mode', 'value': 'enabled'}
                ],
                value=['enabled'],  # 기본값은 체크 상태
                labelStyle={'fontSize': '14px',
                            'color': '#2c3e50', 'fontWeight': 'bold'}
            )
        ], style={'marginTop': '5px'})

    ], style={'padding': '20px', 'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'borderRadius': '8px', 'margin': '0 0 20px 0'})


def create_spectrum_graph_section():
    """스펙트럼 그래프 섹션 생성"""
    return html.Div([
        html.H3("RGB 발광 스펙트럼",
                style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'color': '#2c3e50'}),

        # 정규화 체크박스 추가
        html.Div([
            dcc.Checklist(
                id='normalize-spectrum',
                options=[
                    {'label': ' 스펙트럼 정규화 (최대값=1)', 'value': 'normalize'}
                ],
                value=['normalize'],  # 기본값은 체크 상태
                labelStyle={'fontSize': '14px', 'color': '#2c3e50'}
            )
        ], style={'marginBottom': '10px'}),

        # 그래프 컴포넌트 - 다중 샘플 분석 탭과 동일한 방식으로 수정
        dcc.Graph(
            id='spectrum-graph',
            style={
                'height': '400px',
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


def create_gamut_results_section():
    """색재현율 결과 섹션 생성"""
    return html.Div([
        html.H3("색재현율 분석 결과",
                style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'color': '#2c3e50'}),

        html.Div(id='gamut-results', style={'padding': '10px', 'backgroundColor': '#f9f9f9',
                 'borderRadius': '4px', 'minHeight': '120px', 'marginTop': '15px'})
    ], style={'padding': '20px', 'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'borderRadius': '8px', 'margin': '0 0 20px 0'})


def create_cie_diagram_section():
    """CIE 다이어그램 섹션 생성"""
    return html.Div([
        html.H3("CIE 색도 다이어그램",
                style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'color': '#2c3e50'}),

        # 색공간 선택 체크박스
        html.Div([
            html.H4("비교할 색공간 선택", style={'margin': '15px 0 10px 0'}),
            dcc.Checklist(
                id='color-space-checklist',
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

        # 다이어그램 유형 선택 (추가)
        html.Div([
            html.H4("다이어그램 유형", style={'margin': '15px 0 10px 0'}),
            dcc.RadioItems(
                id='single-cie-diagram-type',
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

        # 다이어그램 그래프 - 다중 샘플 분석 탭과 동일한 방식으로 수정
        dcc.Graph(
            id='cie-diagram',
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
