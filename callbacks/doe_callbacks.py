from dash import Input, Output, State, html, dash_table, dcc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import color_math
import dash


def register_callbacks(app):
    @app.callback(
        [Output('doe-results-table', 'columns'),
         Output('doe-results-table', 'data'),
         Output('doe-results-info', 'children')],
        [Input('apply-doe', 'n_clicks')],
        [State('r-min-x', 'value'), State('r-min-y', 'value'),
         State('r-max-x', 'value'), State('r-max-y', 'value'),
         State('r-divisions', 'value'),
         State('g-min-x', 'value'), State('g-min-y', 'value'),
         State('g-max-x', 'value'), State('g-max-y', 'value'),
         State('g-divisions', 'value'),
         State('b-min-x', 'value'), State('b-min-y', 'value'),
         State('b-max-x', 'value'), State('b-max-y', 'value'),
         State('b-divisions', 'value')]
    )
    def update_doe_results(n_clicks, r_min_x, r_min_y, r_max_x, r_max_y, r_divisions,
                           g_min_x, g_min_y, g_max_x, g_max_y, g_divisions,
                           b_min_x, b_min_y, b_max_x, b_max_y, b_divisions):
        # 초기 상태 또는 입력값이 없는 경우
        if n_clicks == 0 or None in [r_min_x, r_min_y, r_max_x, r_max_y, r_divisions,
                                     g_min_x, g_min_y, g_max_x, g_max_y, g_divisions,
                                     b_min_x, b_min_y, b_max_x, b_max_y, b_divisions]:
            return [], [], html.Div("모든 입력값을 채우고 분석 버튼을 클릭하세요")

        # 분할 수 유효성 검사 - 최소값과 최대값이 같은 경우 분할 수 1 허용
        error_messages = []

        # R 분할 수 검사
        if r_divisions < 1:
            error_messages.append("R 분할 수는 1 이상이어야 합니다.")
        elif r_divisions == 1 and (abs(r_min_x - r_max_x) > 1e-10 or abs(r_min_y - r_max_y) > 1e-10):
            error_messages.append("R 최소값과 최대값이 다른 경우 분할 수는 2 이상이어야 합니다.")

        # G 분할 수 검사
        if g_divisions < 1:
            error_messages.append("G 분할 수는 1 이상이어야 합니다.")
        elif g_divisions == 1 and (g_min_x != g_max_x or g_min_y != g_max_y):
            error_messages.append("G 최소값과 최대값이 다른 경우 분할 수는 2 이상이어야 합니다.")

        # B 분할 수 검사
        if b_divisions < 1:
            error_messages.append("B 분할 수는 1 이상이어야 합니다.")
        elif b_divisions == 1 and (b_min_x != b_max_x or b_min_y != b_max_y):
            error_messages.append("B 최소값과 최대값이 다른 경우 분할 수는 2 이상이어야 합니다.")

        if error_messages:
            return [], [], html.Div([
                html.P(msg, style={'color': 'red', 'margin': '5px 0'})
                for msg in error_messages
            ])

        # 색좌표 범위 생성
        r_x_values = np.linspace(r_min_x, r_max_x, r_divisions)
        r_y_values = np.linspace(r_min_y, r_max_y, r_divisions)
        g_x_values = np.linspace(g_min_x, g_max_x, g_divisions)
        g_y_values = np.linspace(g_min_y, g_max_y, g_divisions)
        b_x_values = np.linspace(b_min_x, b_max_x, b_divisions)
        b_y_values = np.linspace(b_min_y, b_max_y, b_divisions)

        # 색공간 정보 가져오기
        color_spaces = color_math.get_color_spaces()

        # 테이블 데이터 생성
        table_data = []
        sample_id = 1

        # 모든 조합 생성 및 계산
        total_combinations = r_divisions * g_divisions * b_divisions
        processed_combinations = 0

        for r_idx in range(r_divisions):
            for g_idx in range(g_divisions):
                for b_idx in range(b_divisions):
                    # 샘플 생성
                    sample = {
                        'red': [r_x_values[r_idx], r_y_values[r_idx]],
                        'green': [g_x_values[g_idx], g_y_values[g_idx]],
                        'blue': [b_x_values[b_idx], b_y_values[b_idx]]
                    }

                    # 체크박스 컬럼 추가 (기본값은 빈 문자열)
                    row = {
                        'Sample': f'Sample{sample_id}',
                        'Selected': '',
                        'Rx': round(r_x_values[r_idx], 4),
                        'Ry': round(r_y_values[r_idx], 4),
                        'Gx': round(g_x_values[g_idx], 4),
                        'Gy': round(g_y_values[g_idx], 4),
                        'Bx': round(b_x_values[b_idx], 4),
                        'By': round(b_y_values[b_idx], 4)
                    }

                    # 각 색공간에 대한 색재현율 계산
                    for cs_name in ['DCI-P3', 'adobeRGB', 'BT2020']:
                        if cs_name in color_spaces:
                            try:
                                coverage_xy, area_xy, coverage_uv, area_uv = color_math.calculate_gamut_coverage(
                                    sample, color_spaces[cs_name])

                                row[f'{cs_name}_xy_coverage'] = f"{coverage_xy:.2f}%"
                                row[f'{cs_name}_xy_area'] = f"{area_xy:.2f}%"
                                row[f'{cs_name}_uv_coverage'] = f"{coverage_uv:.2f}%"
                                row[f'{cs_name}_uv_area'] = f"{area_uv:.2f}%"

                            except Exception as e:
                                row[f'{cs_name}_xy_coverage'] = "오류"
                                row[f'{cs_name}_xy_area'] = "오류"
                                row[f'{cs_name}_uv_coverage'] = "오류"
                                row[f'{cs_name}_uv_area'] = "오류"

                    table_data.append(row)
                    sample_id += 1
                    processed_combinations += 1

        # 테이블 컬럼 정의
        columns = [
            {'name': 'Sample', 'id': 'Sample'},
            {'name': 'Rx', 'id': 'Rx'},
            {'name': 'Ry', 'id': 'Ry'},
            {'name': 'Gx', 'id': 'Gx'},
            {'name': 'Gy', 'id': 'Gy'},
            {'name': 'Bx', 'id': 'Bx'},
            {'name': 'By', 'id': 'By'},
        ]

        # xy 중첩비 그룹
        for cs_name in ['DCI-P3', 'adobeRGB', 'BT2020']:
            columns.append({'name': f'{cs_name} xy 중첩비',
                           'id': f'{cs_name}_xy_coverage'})

        # xy 면적비 그룹
        for cs_name in ['DCI-P3', 'adobeRGB', 'BT2020']:
            columns.append({'name': f'{cs_name} xy 면적비',
                           'id': f'{cs_name}_xy_area'})

        # uv 중첩비 그룹
        for cs_name in ['DCI-P3', 'adobeRGB', 'BT2020']:
            columns.append({'name': f'{cs_name} uv 중첩비',
                           'id': f'{cs_name}_uv_coverage'})

        # uv 면적비 그룹
        for cs_name in ['DCI-P3', 'adobeRGB', 'BT2020']:
            columns.append({'name': f'{cs_name} uv 면적비',
                           'id': f'{cs_name}_uv_area'})

        # 그래프 표시 열 추가
        columns.append({
            'name': '그래프 표시 (O/X)',
            'id': 'Selected',
            'type': 'text'
        })

        # 결과 정보 생성
        info = html.Div([
            html.P(f"총 {total_combinations}개의 조합이 생성되었습니다.",
                   style={'fontSize': '14px', 'fontWeight': 'bold', 'margin': '10px 0'}),
            html.P("xy 중첩비: 타겟 색공간 대비 겹치는 영역의 비율 (CIE 1931 xy 좌표계)",
                   style={'fontSize': '12px', 'color': '#7f8c8d', 'margin': '5px 0 0 0'}),
            html.P("xy 면적비: 타겟 색공간 대비 측정 RGB 색역의 면적 비율 (CIE 1931 xy 좌표계)",
                   style={'fontSize': '12px', 'color': '#7f8c8d', 'margin': '5px 0 0 0'}),
            html.P("uv 중첩비: 타겟 색공간 대비 겹치는 영역의 비율 (CIE 1976 u'v' 좌표계)",
                   style={'fontSize': '12px', 'color': '#7f8c8d', 'margin': '5px 0 0 0'}),
            html.P("uv 면적비: 타겟 색공간 대비 측정 RGB 색역의 면적 비율 (CIE 1976 u'v' 좌표계)",
                   style={'fontSize': '12px', 'color': '#7f8c8d', 'margin': '5px 0 0 0'})
        ])

        return columns, table_data, info

    @app.callback(
        Output('doe-stats', 'children'),
        [Input('doe-results-table', 'data')]
    )
    def update_doe_stats(data):
        if not data:
            return html.Div("데이터가 없습니다.")

        # 통계 계산을 위한 데이터 추출
        stats = {
            'DCI-P3': {'xy_coverage': [], 'xy_area': [], 'uv_coverage': [], 'uv_area': []},
            'adobeRGB': {'xy_coverage': [], 'xy_area': [], 'uv_coverage': [], 'uv_area': []},
            'BT2020': {'xy_coverage': [], 'xy_area': [], 'uv_coverage': [], 'uv_area': []}
        }

        for row in data:
            for cs_name in ['DCI-P3', 'adobeRGB', 'BT2020']:
                for metric in ['xy_coverage', 'xy_area', 'uv_coverage', 'uv_area']:
                    key = f'{cs_name}_{metric}'
                    if key in row and row[key] != "오류":
                        try:
                            value = float(row[key].replace('%', ''))
                            stats[cs_name][metric].append(value)
                        except:
                            pass

        # 통계 테이블 데이터 생성
        table_data = []

        # 분석 결과와 동일한 순서로 통계 데이터 생성
        metrics_order = [
            ('xy_coverage', 'xy 중첩비'),
            ('xy_area', 'xy 면적비'),
            ('uv_coverage', 'uv 중첩비'),
            ('uv_area', 'uv 면적비')
        ]

        for metric_id, metric_name in metrics_order:
            for cs_name in ['DCI-P3', 'adobeRGB', 'BT2020']:
                if stats[cs_name][metric_id]:
                    values = stats[cs_name][metric_id]
                    row = {
                        '항목': f'{cs_name} {metric_name}',
                        '평균': f"{sum(values) / len(values):.2f}%",
                        '최소값': f"{min(values):.2f}%",
                        '최대값': f"{max(values):.2f}%",
                        '표준편차': f"{pd.Series(values).std():.2f}%"
                    }
                    table_data.append(row)

        # 통계 테이블 컬럼 정의
        columns = [
            {'name': '항목', 'id': '항목'},
            {'name': '평균', 'id': '평균'},
            {'name': '최소값', 'id': '최소값'},
            {'name': '최대값', 'id': '최대값'},
            {'name': '표준편차', 'id': '표준편차'}
        ]

        # 통계 테이블 생성
        return html.Div([
            dash_table.DataTable(
                columns=columns,
                data=table_data,
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
                style_data_conditional=[
                    {
                        'if': {'column_id': '항목'},
                        'backgroundColor': '#f2f2f2',
                        'fontWeight': 'bold'
                    }
                ]
            )
        ])

    @app.callback(
        Output('doe-cie-diagram', 'figure'),
        [Input('doe-results-table', 'data'),
         Input('doe-color-space-checklist', 'value'),
         Input('doe-diagram-type', 'value')]
    )
    def update_doe_cie_diagram(data, selected_color_spaces, diagram_type):
        # 다이어그램 유형에 따라 설정
        if diagram_type == 'xy':
            title = 'CIE 1931 xy 색도 다이어그램'
            xaxis_title = 'x'
            yaxis_title = 'y'
            xaxis_range = [0, 0.8]
            yaxis_range = [0, 0.9]
        else:  # 'uv'
            title = 'CIE 1976 u\'v\' 색도 다이어그램'
            xaxis_title = 'u\''
            yaxis_title = 'v\''
            xaxis_range = [0, 0.6]
            yaxis_range = [0, 0.6]

        # 기본 다이어그램 생성
        fig = go.Figure()

        # 스펙트럼 궤적 생성
        x_spectral, y_spectral = color_math.generate_spectral_locus(
            start_nm=380, end_nm=780, step=0.5)

        if diagram_type == 'uv':
            # xy 좌표를 u'v' 좌표로 변환
            u_spectral = []
            v_spectral = []
            for i in range(len(x_spectral)):
                u_prime, v_prime = color_math.xy_to_uv(
                    x_spectral[i], y_spectral[i])
                u_spectral.append(u_prime)
                v_spectral.append(v_prime)
            x_spectral, y_spectral = u_spectral, v_spectral

        # 스펙트럼 궤적 그리기
        fig.add_trace(go.Scatter(
            x=x_spectral,
            y=y_spectral,
            mode='lines',
            line=dict(color='black', width=1.5),
            showlegend=False
        ))

        # 스펙트럼 궤적 닫기 (끝점과 시작점 연결)
        fig.add_trace(go.Scatter(
            x=[x_spectral[-1], x_spectral[0]],
            y=[y_spectral[-1], y_spectral[0]],
            mode='lines',
            line=dict(color='black', width=1.5),
            showlegend=False
        ))

        # 색공간 삼각형 그리기
        color_spaces = color_math.get_color_spaces()
        for cs_name in selected_color_spaces:
            if cs_name in color_spaces:
                cs = color_spaces[cs_name]

                if diagram_type == 'xy':
                    # xy 좌표계 사용
                    x_points = [cs['red'][0], cs['green']
                                [0], cs['blue'][0], cs['red'][0]]
                    y_points = [cs['red'][1], cs['green']
                                [1], cs['blue'][1], cs['red'][1]]
                else:  # 'uv'
                    # u'v' 좌표계로 변환
                    x_points = []
                    y_points = []
                    for point in ['red', 'green', 'blue', 'red']:
                        u_prime, v_prime = color_math.xy_to_uv(
                            cs[point][0], cs[point][1])
                        x_points.append(u_prime)
                        y_points.append(v_prime)

                # 색공간 삼각형 그리기
                fig.add_trace(go.Scatter(
                    x=x_points,
                    y=y_points,
                    mode='lines+markers',
                    name=cs_name,
                    line=dict(width=2)
                ))

        # 샘플 데이터가 있는 경우에만 샘플 그리기
        if data:
            # 'O' 또는 '1'이 입력된 샘플만 표시
            selected_samples = [row['Sample'] for row in data
                                if row.get('Selected') and str(row.get('Selected')).upper() in ['O', '1']]

            # 각 샘플의 RGB 좌표 그리기
            for row in data:
                # 선택된 샘플만 그리기
                if row['Sample'] in selected_samples:
                    if diagram_type == 'xy':
                        # xy 좌표계 사용
                        x_points = [row['Rx'], row['Gx'], row['Bx'], row['Rx']]
                        y_points = [row['Ry'], row['Gy'], row['By'], row['Ry']]
                    else:  # 'uv'
                        # u'v' 좌표계로 변환
                        x_points = []
                        y_points = []
                        for x, y in [(row['Rx'], row['Ry']), (row['Gx'], row['Gy']), (row['Bx'], row['By']), (row['Rx'], row['Ry'])]:
                            u_prime, v_prime = color_math.xy_to_uv(x, y)
                            x_points.append(u_prime)
                            y_points.append(v_prime)

                    # 샘플 삼각형 그리기
                    fig.add_trace(go.Scatter(
                        x=x_points,
                        y=y_points,
                        mode='lines+markers',
                        name=row['Sample'],
                        line=dict(width=1.5, dash='dash'),
                        marker=dict(size=8)
                    ))

        # 그래프 레이아웃 설정
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            xaxis=dict(
                range=xaxis_range,
                constrain='domain',
                showgrid=False,
                showline=True,
                linecolor='black',
                linewidth=1,
                mirror=False
            ),
            yaxis=dict(
                range=yaxis_range,
                scaleanchor='x',
                scaleratio=1,
                showgrid=False,
                showline=True,
                linecolor='black',
                linewidth=1,
                mirror=False
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(orientation="h", yanchor="top",
                    y=-0.15, xanchor="center", x=0.5),
            autosize=True,
            margin=dict(l=50, r=30, t=50, b=50),
            height=500
        )

        return fig

    @app.callback(
        Output('doe-3d-graph', 'figure'),
        [Input('doe-results-table', 'data'),
         Input('doe-3d-colorspace', 'value'),
         Input('doe-3d-metric', 'value'),
         Input('doe-3d-coordinate', 'value')]
    )
    def update_doe_3d_graph(data, colorspace, metric, coordinate):
        if not data:
            # 데이터가 없는 경우 빈 그래프 반환
            return go.Figure().update_layout(
                title="데이터가 없습니다. 분석을 실행하세요.",
                scene=dict(
                    xaxis_title="Rx",
                    yaxis_title="Gx",
                    zaxis_title="Bx"
                )
            )

        # 데이터 추출
        x_values = []  # Rx 값
        y_values = []  # Gx 값
        z_values = []  # Bx 값
        color_values = []  # 색상값 (중첩비 또는 면적비)

        # 측정 항목 및 좌표계에 따른 컬럼 ID 결정
        column_id = f"{colorspace}_{coordinate}_{metric}"

        for row in data:
            try:
                # RGB x 좌표 추출
                x = float(row['Rx'])
                y = float(row['Gx'])
                z = float(row['Bx'])

                # 색재현율 값 추출 (% 기호 제거)
                if column_id in row and row[column_id] != "오류":
                    color_value = float(row[column_id].replace('%', ''))

                    x_values.append(x)
                    y_values.append(y)
                    z_values.append(z)
                    color_values.append(color_value)
            except:
                continue

        # 측정 항목 이름 설정
        metric_name = "중첩비" if metric == "coverage" else "면적비"
        coordinate_name = "CIE 1931 xy" if coordinate == "xy" else "CIE 1976 u'v'"

        # 3D 산점도 생성
        fig = go.Figure(data=[go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,
            mode='markers',
            marker=dict(
                size=5,
                color=color_values,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(
                    title=f"{metric_name} (%)"
                )
            ),
            text=[f"Rx: {x:.4f}, Gx: {y:.4f}, Bx: {z:.4f}<br>{metric_name}: {c:.2f}%" for x, y, z, c in zip(
                x_values, y_values, z_values, color_values)],
            hoverinfo='text'
        )])

        # 그래프 레이아웃 설정
        fig.update_layout(
            title=f"{colorspace} {coordinate_name} {metric_name} 3D 시각화",
            scene=dict(
                xaxis_title="Rx",
                yaxis_title="Gx",
                zaxis_title="Bx",
                xaxis=dict(range=[min(x_values), max(x_values)]),
                yaxis=dict(range=[min(y_values), max(y_values)]),
                zaxis=dict(range=[min(z_values), max(z_values)])
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        return fig

    @app.callback(
        Output('doe-data-store', 'data'),
        [Input('r-min-x', 'value'), Input('r-min-y', 'value'),
         Input('r-max-x', 'value'), Input('r-max-y', 'value'),
         Input('r-divisions', 'value'),
         Input('g-min-x', 'value'), Input('g-min-y', 'value'),
         Input('g-max-x', 'value'), Input('g-max-y', 'value'),
         Input('g-divisions', 'value'),
         Input('b-min-x', 'value'), Input('b-min-y', 'value'),
         Input('b-max-x', 'value'), Input('b-max-y', 'value'),
         Input('b-divisions', 'value')]
    )
    def store_doe_data(r_min_x, r_min_y, r_max_x, r_max_y, r_divisions,
                       g_min_x, g_min_y, g_max_x, g_max_y, g_divisions,
                       b_min_x, b_min_y, b_max_x, b_max_y, b_divisions):
        return {
            'r_min_x': r_min_x, 'r_min_y': r_min_y,
            'r_max_x': r_max_x, 'r_max_y': r_max_y,
            'r_divisions': r_divisions,
            'g_min_x': g_min_x, 'g_min_y': g_min_y,
            'g_max_x': g_max_x, 'g_max_y': g_max_y,
            'g_divisions': g_divisions,
            'b_min_x': b_min_x, 'b_min_y': b_min_y,
            'b_max_x': b_max_x, 'b_max_y': b_max_y,
            'b_divisions': b_divisions
        }

    @app.callback(
        [Output('r-min-x', 'value'), Output('r-min-y', 'value'),
         Output('r-max-x', 'value'), Output('r-max-y', 'value'),
         Output('r-divisions', 'value'),
         Output('g-min-x', 'value'), Output('g-min-y', 'value'),
         Output('g-max-x', 'value'), Output('g-max-y', 'value'),
         Output('g-divisions', 'value'),
         Output('b-min-x', 'value'), Output('b-min-y', 'value'),
         Output('b-max-x', 'value'), Output('b-max-y', 'value'),
         Output('b-divisions', 'value')],
        [Input('doe-data-store', 'modified_timestamp')],
        [State('doe-data-store', 'data')]
    )
    def load_doe_data(ts, data):
        if ts is None or data is None:
            return [None] * 15
        return [
            data.get('r_min_x'), data.get('r_min_y'),
            data.get('r_max_x'), data.get('r_max_y'),
            data.get('r_divisions'),
            data.get('g_min_x'), data.get('g_min_y'),
            data.get('g_max_x'), data.get('g_max_y'),
            data.get('g_divisions'),
            data.get('b_min_x'), data.get('b_min_y'),
            data.get('b_max_x'), data.get('b_max_y'),
            data.get('b_divisions')
        ]

    @app.callback(
        [Output('r-min-x', 'value', allow_duplicate=True),
         Output('r-min-y', 'value', allow_duplicate=True),
         Output('r-max-x', 'value', allow_duplicate=True),
         Output('r-max-y', 'value', allow_duplicate=True),
         Output('r-divisions', 'value', allow_duplicate=True),
         Output('g-min-x', 'value', allow_duplicate=True),
         Output('g-min-y', 'value', allow_duplicate=True),
         Output('g-max-x', 'value', allow_duplicate=True),
         Output('g-max-y', 'value', allow_duplicate=True),
         Output('g-divisions', 'value', allow_duplicate=True),
         Output('b-min-x', 'value', allow_duplicate=True),
         Output('b-min-y', 'value', allow_duplicate=True),
         Output('b-max-x', 'value', allow_duplicate=True),
         Output('b-max-y', 'value', allow_duplicate=True),
         Output('b-divisions', 'value', allow_duplicate=True)],
        [Input('reset-doe-inputs', 'n_clicks')],
        prevent_initial_call=True
    )
    def reset_doe_inputs(n_clicks):
        if n_clicks > 0:
            return [None] * 15  # 분할 수만 기본값 5로 설정
        return [dash.no_update] * 15
