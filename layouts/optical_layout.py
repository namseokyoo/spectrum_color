from dash import html, dcc, dash_table


def create_optical_tab():
    """광학 상수 추출 탭 레이아웃 생성"""
    return dcc.Tab(
        label='광학 상수 추출',
        children=[
            html.Div([
                # html.H3("PL과 EL 스펙트럼으로부터 광학 상수(OC) 추출",
                # style={'textAlign': 'center', 'margin': '20px 0', 'color': '#2c3e50', 'fontWeight': 'bold'}),

                # R, G, B 섹션을 가로로 배치
                html.Div([
                    # R 섹션
                    html.Div([
                        html.H4("R 광학 상수 추출",
                                style={'color': '#e74c3c', 'borderBottom': '2px solid #e74c3c', 'paddingBottom': '10px'}),

                        # PL 데이터 입력 및 그래프
                        html.Div([
                            html.Label("PL 스펙트럼 데이터:",
                                       style={'fontWeight': 'bold', 'marginTop': '10px'}),
                            html.Div([
                                # 왼쪽: 입력 필드
                                html.Div([
                                    dcc.Textarea(
                                        id='r-pl-data',
                                        placeholder='파장(nm), 강도 형식으로 데이터를 입력하세요...',
                                        style={
                                            'width': '100%', 'height': '150px', 'marginBottom': '10px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                                # 오른쪽: 그래프
                                html.Div([
                                    dcc.Graph(
                                        id='r-pl-graph',
                                        style={'height': '150px',
                                               'marginBottom': '10px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                            ], style={'display': 'flex', 'flexWrap': 'wrap'}),

                            # EL 데이터 입력 및 그래프
                            html.Label("EL 스펙트럼 데이터:",
                                       style={'fontWeight': 'bold', 'marginTop': '10px'}),
                            html.Div([
                                # 왼쪽: 입력 필드
                                html.Div([
                                    dcc.Textarea(
                                        id='r-el-data',
                                        placeholder='파장(nm), 강도 형식으로 데이터를 입력하세요...',
                                        style={
                                            'width': '100%', 'height': '150px', 'marginBottom': '10px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                                # 오른쪽: 그래프
                                html.Div([
                                    dcc.Graph(
                                        id='r-el-graph',
                                        style={'height': '150px',
                                               'marginBottom': '10px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                            ], style={'display': 'flex', 'flexWrap': 'wrap'}),

                            # 노이즈 제거 체크박스
                            dcc.Checklist(
                                id='r-noise-reduction',
                                options=[
                                    {'label': '노이즈 제거', 'value': 'enabled'}],
                                value=['enabled'],  # 기본값 선택
                                style={
                                    'display': 'inline-block', 'marginRight': '15px', 'verticalAlign': 'middle'}
                            ),

                            # 계산 버튼
                            html.Button('OC 계산',
                                        id='calculate-r-oc',
                                        n_clicks=0,
                                        style={'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                                               'padding': '10px 20px', 'marginTop': '10px', 'borderRadius': '4px',
                                               'display': 'inline-block', 'verticalAlign': 'middle'}),

                            # 결과 그래프와 데이터 (가로 배치)
                            html.Div([
                                # 왼쪽: OC 데이터 표시
                                html.Div([
                                    html.Label("계산된 OC 데이터 (정규화됨):",
                                               style={'fontWeight': 'bold', 'marginTop': '20px'}),
                                    dcc.Textarea(
                                        id='r-oc-data',
                                        readOnly=True,
                                        placeholder='OC 계산 버튼을 클릭하면 여기에 데이터가 표시됩니다...',
                                        style={
                                            'width': '100%', 'height': '200px', 'marginTop': '10px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                                # 오른쪽: OC 그래프
                                html.Div([
                                    dcc.Graph(
                                        id='r-oc-graph',
                                        style={'height': '300px',
                                               'marginTop': '20px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginTop': '20px'}),

                            # 새로운 섹션: OC와 PL로 EL 예측
                            html.Div([
                                html.H5("OC와 PL로 EL 예측",
                                        style={'color': '#e74c3c', 'borderTop': '1px solid #e74c3c', 'paddingTop': '15px', 'marginTop': '20px'}),

                                # PL 데이터 입력 및 그래프
                                html.Label("새 PL 스펙트럼 데이터:",
                                           style={'fontWeight': 'bold', 'marginTop': '10px'}),
                                html.Div([
                                    # 왼쪽: 입력 필드
                                    html.Div([
                                        dcc.Textarea(
                                            id='r-new-pl-data',
                                            placeholder='파장(nm), 강도 형식으로 데이터를 입력하세요...',
                                            style={
                                                'width': '100%', 'height': '150px', 'marginBottom': '10px'}
                                        ),
                                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                                    # 오른쪽: 그래프
                                    html.Div([
                                        dcc.Graph(
                                            id='r-new-pl-graph',
                                            style={'height': '150px',
                                                   'marginBottom': '10px'}
                                        ),
                                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                                ], style={'display': 'flex', 'flexWrap': 'wrap'}),

                                # 계산 버튼
                                html.Button('EL 예측',
                                            id='calculate-r-predicted-el',
                                            n_clicks=0,
                                            style={'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                                                   'padding': '10px 20px', 'marginTop': '10px', 'borderRadius': '4px',
                                                   'display': 'inline-block', 'verticalAlign': 'middle'}),

                                # 결과 그래프와 데이터 (가로 배치)
                                html.Div([
                                    # 왼쪽: 예측된 EL 데이터 표시
                                    html.Div([
                                        html.Label("예측된 EL 데이터:",
                                                   style={'fontWeight': 'bold', 'marginTop': '20px'}),
                                        dcc.Textarea(
                                            id='r-predicted-el-data',
                                            readOnly=True,
                                            placeholder='EL 예측 버튼을 클릭하면 여기에 데이터가 표시됩니다...',
                                            style={
                                                'width': '100%', 'height': '200px', 'marginTop': '10px'}
                                        ),
                                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                                    # 오른쪽: 예측된 EL 그래프
                                    html.Div([
                                        dcc.Graph(
                                            id='r-predicted-el-graph',
                                            style={'height': '300px',
                                                   'marginTop': '20px'}
                                        ),
                                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                                ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginTop': '20px'})
                            ])
                        ])
                    ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top',
                              'padding': '10px', 'backgroundColor': '#fff', 'borderRadius': '8px',
                              'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),

                    # G 섹션
                    html.Div([
                        html.H4("G 광학 상수 추출",
                                style={'color': '#2ecc71', 'borderBottom': '2px solid #2ecc71', 'paddingBottom': '10px'}),

                        # PL 데이터 입력 및 그래프
                        html.Div([
                            html.Label("PL 스펙트럼 데이터:",
                                       style={'fontWeight': 'bold', 'marginTop': '10px'}),
                            html.Div([
                                # 왼쪽: 입력 필드
                                html.Div([
                                    dcc.Textarea(
                                        id='g-pl-data',
                                        placeholder='파장(nm), 강도 형식으로 데이터를 입력하세요...',
                                        style={
                                            'width': '100%', 'height': '150px', 'marginBottom': '10px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                                # 오른쪽: 그래프
                                html.Div([
                                    dcc.Graph(
                                        id='g-pl-graph',
                                        style={'height': '150px',
                                               'marginBottom': '10px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                            ], style={'display': 'flex', 'flexWrap': 'wrap'}),

                            # EL 데이터 입력 및 그래프
                            html.Label("EL 스펙트럼 데이터:",
                                       style={'fontWeight': 'bold', 'marginTop': '10px'}),
                            html.Div([
                                # 왼쪽: 입력 필드
                                html.Div([
                                    dcc.Textarea(
                                        id='g-el-data',
                                        placeholder='파장(nm), 강도 형식으로 데이터를 입력하세요...',
                                        style={
                                            'width': '100%', 'height': '150px', 'marginBottom': '10px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                                # 오른쪽: 그래프
                                html.Div([
                                    dcc.Graph(
                                        id='g-el-graph',
                                        style={'height': '150px',
                                               'marginBottom': '10px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                            ], style={'display': 'flex', 'flexWrap': 'wrap'}),

                            # 노이즈 제거 체크박스
                            dcc.Checklist(
                                id='g-noise-reduction',
                                options=[
                                    {'label': '노이즈 제거', 'value': 'enabled'}],
                                value=['enabled'],  # 기본값 선택
                                style={
                                    'display': 'inline-block', 'marginRight': '15px', 'verticalAlign': 'middle'}
                            ),

                            # 계산 버튼
                            html.Button('OC 계산',
                                        id='calculate-g-oc',
                                        n_clicks=0,
                                        style={'backgroundColor': '#2ecc71', 'color': 'white', 'border': 'none',
                                               'padding': '10px 20px', 'marginTop': '10px', 'borderRadius': '4px',
                                               'display': 'inline-block', 'verticalAlign': 'middle'}),

                            # 결과 그래프와 데이터 (가로 배치)
                            html.Div([
                                # 왼쪽: OC 데이터 표시
                                html.Div([
                                    html.Label("계산된 OC 데이터 (정규화됨):",
                                               style={'fontWeight': 'bold', 'marginTop': '20px'}),
                                    dcc.Textarea(
                                        id='g-oc-data',
                                        readOnly=True,
                                        placeholder='OC 계산 버튼을 클릭하면 여기에 데이터가 표시됩니다...',
                                        style={
                                            'width': '100%', 'height': '200px', 'marginTop': '10px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                                # 오른쪽: OC 그래프
                                html.Div([
                                    dcc.Graph(
                                        id='g-oc-graph',
                                        style={'height': '300px',
                                               'marginTop': '20px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginTop': '20px'}),

                            # 새로운 섹션: OC와 PL로 EL 예측
                            html.Div([
                                html.H5("OC와 PL로 EL 예측",
                                        style={'color': '#2ecc71', 'borderTop': '1px solid #2ecc71', 'paddingTop': '15px', 'marginTop': '20px'}),

                                # PL 데이터 입력 및 그래프
                                html.Label("새 PL 스펙트럼 데이터:",
                                           style={'fontWeight': 'bold', 'marginTop': '10px'}),
                                html.Div([
                                    # 왼쪽: 입력 필드
                                    html.Div([
                                        dcc.Textarea(
                                            id='g-new-pl-data',
                                            placeholder='파장(nm), 강도 형식으로 데이터를 입력하세요...',
                                            style={
                                                'width': '100%', 'height': '150px', 'marginBottom': '10px'}
                                        ),
                                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                                    # 오른쪽: 그래프
                                    html.Div([
                                        dcc.Graph(
                                            id='g-new-pl-graph',
                                            style={'height': '150px',
                                                   'marginBottom': '10px'}
                                        ),
                                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                                ], style={'display': 'flex', 'flexWrap': 'wrap'}),

                                # 계산 버튼
                                html.Button('EL 예측',
                                            id='calculate-g-predicted-el',
                                            n_clicks=0,
                                            style={'backgroundColor': '#2ecc71', 'color': 'white', 'border': 'none',
                                                   'padding': '10px 20px', 'marginTop': '10px', 'borderRadius': '4px',
                                                   'display': 'inline-block', 'verticalAlign': 'middle'}),

                                # 결과 그래프와 데이터 (가로 배치)
                                html.Div([
                                    # 왼쪽: 예측된 EL 데이터 표시
                                    html.Div([
                                        html.Label("예측된 EL 데이터:",
                                                   style={'fontWeight': 'bold', 'marginTop': '20px'}),
                                        dcc.Textarea(
                                            id='g-predicted-el-data',
                                            readOnly=True,
                                            placeholder='EL 예측 버튼을 클릭하면 여기에 데이터가 표시됩니다...',
                                            style={
                                                'width': '100%', 'height': '200px', 'marginTop': '10px'}
                                        ),
                                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                                    # 오른쪽: 예측된 EL 그래프
                                    html.Div([
                                        dcc.Graph(
                                            id='g-predicted-el-graph',
                                            style={'height': '300px',
                                                   'marginTop': '20px'}
                                        ),
                                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                                ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginTop': '20px'})
                            ])
                        ])
                    ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '1%',
                              'padding': '10px', 'backgroundColor': '#fff', 'borderRadius': '8px',
                              'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),

                    # B 섹션
                    html.Div([
                        html.H4("B 광학 상수 추출",
                                style={'color': '#3498db', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),

                        # PL 데이터 입력 및 그래프
                        html.Div([
                            html.Label("PL 스펙트럼 데이터:",
                                       style={'fontWeight': 'bold', 'marginTop': '10px'}),
                            html.Div([
                                # 왼쪽: 입력 필드
                                html.Div([
                                    dcc.Textarea(
                                        id='b-pl-data',
                                        placeholder='파장(nm), 강도 형식으로 데이터를 입력하세요...',
                                        style={
                                            'width': '100%', 'height': '150px', 'marginBottom': '10px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                                # 오른쪽: 그래프
                                html.Div([
                                    dcc.Graph(
                                        id='b-pl-graph',
                                        style={'height': '150px',
                                               'marginBottom': '10px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                            ], style={'display': 'flex', 'flexWrap': 'wrap'}),

                            # EL 데이터 입력 및 그래프
                            html.Label("EL 스펙트럼 데이터:",
                                       style={'fontWeight': 'bold', 'marginTop': '10px'}),
                            html.Div([
                                # 왼쪽: 입력 필드
                                html.Div([
                                    dcc.Textarea(
                                        id='b-el-data',
                                        placeholder='파장(nm), 강도 형식으로 데이터를 입력하세요...',
                                        style={
                                            'width': '100%', 'height': '150px', 'marginBottom': '10px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                                # 오른쪽: 그래프
                                html.Div([
                                    dcc.Graph(
                                        id='b-el-graph',
                                        style={'height': '150px',
                                               'marginBottom': '10px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                            ], style={'display': 'flex', 'flexWrap': 'wrap'}),

                            # 노이즈 제거 체크박스
                            dcc.Checklist(
                                id='b-noise-reduction',
                                options=[
                                    {'label': '노이즈 제거', 'value': 'enabled'}],
                                value=['enabled'],  # 기본값 선택
                                style={
                                    'display': 'inline-block', 'marginRight': '15px', 'verticalAlign': 'middle'}
                            ),

                            # 계산 버튼
                            html.Button('OC 계산',
                                        id='calculate-b-oc',
                                        n_clicks=0,
                                        style={'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                                               'padding': '10px 20px', 'marginTop': '10px', 'borderRadius': '4px',
                                               'display': 'inline-block', 'verticalAlign': 'middle'}),

                            # 결과 그래프와 데이터 (가로 배치)
                            html.Div([
                                # 왼쪽: OC 데이터 표시
                                html.Div([
                                    html.Label("계산된 OC 데이터 (정규화됨):",
                                               style={'fontWeight': 'bold', 'marginTop': '20px'}),
                                    dcc.Textarea(
                                        id='b-oc-data',
                                        readOnly=True,
                                        placeholder='OC 계산 버튼을 클릭하면 여기에 데이터가 표시됩니다...',
                                        style={
                                            'width': '100%', 'height': '200px', 'marginTop': '10px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                                # 오른쪽: OC 그래프
                                html.Div([
                                    dcc.Graph(
                                        id='b-oc-graph',
                                        style={'height': '300px',
                                               'marginTop': '20px'}
                                    ),
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginTop': '20px'}),

                            # 새로운 섹션: OC와 PL로 EL 예측
                            html.Div([
                                html.H5("OC와 PL로 EL 예측",
                                        style={'color': '#3498db', 'borderTop': '1px solid #3498db', 'paddingTop': '15px', 'marginTop': '20px'}),

                                # PL 데이터 입력 및 그래프
                                html.Label("새 PL 스펙트럼 데이터:",
                                           style={'fontWeight': 'bold', 'marginTop': '10px'}),
                                html.Div([
                                    # 왼쪽: 입력 필드
                                    html.Div([
                                        dcc.Textarea(
                                            id='b-new-pl-data',
                                            placeholder='파장(nm), 강도 형식으로 데이터를 입력하세요...',
                                            style={
                                                'width': '100%', 'height': '150px', 'marginBottom': '10px'}
                                        ),
                                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                                    # 오른쪽: 그래프
                                    html.Div([
                                        dcc.Graph(
                                            id='b-new-pl-graph',
                                            style={'height': '150px',
                                                   'marginBottom': '10px'}
                                        ),
                                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                                ], style={'display': 'flex', 'flexWrap': 'wrap'}),

                                # 계산 버튼
                                html.Button('EL 예측',
                                            id='calculate-b-predicted-el',
                                            n_clicks=0,
                                            style={'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                                                   'padding': '10px 20px', 'marginTop': '10px', 'borderRadius': '4px',
                                                   'display': 'inline-block', 'verticalAlign': 'middle'}),

                                # 결과 그래프와 데이터 (가로 배치)
                                html.Div([
                                    # 왼쪽: 예측된 EL 데이터 표시
                                    html.Div([
                                        html.Label("예측된 EL 데이터:",
                                                   style={'fontWeight': 'bold', 'marginTop': '20px'}),
                                        dcc.Textarea(
                                            id='b-predicted-el-data',
                                            readOnly=True,
                                            placeholder='EL 예측 버튼을 클릭하면 여기에 데이터가 표시됩니다...',
                                            style={
                                                'width': '100%', 'height': '200px', 'marginTop': '10px'}
                                        ),
                                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                                    # 오른쪽: 예측된 EL 그래프
                                    html.Div([
                                        dcc.Graph(
                                            id='b-predicted-el-graph',
                                            style={'height': '300px',
                                                   'marginTop': '20px'}
                                        ),
                                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                                ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginTop': '20px'})
                            ])
                        ])
                    ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '0.5%',
                              'padding': '10px', 'backgroundColor': '#fff', 'borderRadius': '8px',
                              'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
                ], style={'margin': '0px 0'})
            ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'})
        ],
        style={'padding': '0px'},
        selected_style={'padding': '0px', 'borderTop': '2px solid #3498db'}
    )
