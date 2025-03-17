from dash import Input, Output, State, html, dash_table, dcc
import pandas as pd
from utils import parsers, color_math
import dash  # dash 모듈 추가


def register_callbacks(app):
    # 다중 샘플 분석 탭 콜백 함수들
    @app.callback(
        Output('multi-sample-results', 'children'),
        [Input('apply-multi-sample', 'n_clicks')],
        [State('multi-sample-data', 'value')]
    )
    def update_multi_sample_results(n_clicks, data):
        if n_clicks == 0 or not data:
            return html.Div("데이터를 입력하고 분석 버튼을 클릭하세요")

        # 데이터 파싱
        samples = parsers.parse_multi_sample_data(data)
        if not samples:
            return html.Div("데이터 형식이 올바르지 않습니다")

        # 여기서는 테이블을 직접 생성하지 않고, 다른 콜백이 처리하도록 함
        return html.Div([
            html.Div(id='results-table-container'),  # 테이블을 위한 컨테이너
            html.Div([
                html.P("xy 중첩비: 타겟 색공간 대비 겹치는 영역의 비율 (CIE 1931 xy 좌표계)",
                       style={'fontSize': '12px', 'color': '#7f8c8d', 'margin': '10px 0 0 0'}),
                html.P("xy 면적비: 타겟 색공간 대비 측정 RGB 색역의 면적 비율 (CIE 1931 xy 좌표계)",
                       style={'fontSize': '12px', 'color': '#7f8c8d', 'margin': '5px 0 0 0'}),
                html.P("uv 중첩비: 타겟 색공간 대비 겹치는 영역의 비율 (CIE 1976 u'v' 좌표계)",
                       style={'fontSize': '12px', 'color': '#7f8c8d', 'margin': '5px 0 0 0'}),
                html.P("uv 면적비: 타겟 색공간 대비 측정 RGB 색역의 면적 비율 (CIE 1976 u'v' 좌표계)",
                       style={'fontSize': '12px', 'color': '#7f8c8d', 'margin': '5px 0 0 0'})
            ], style={'marginTop': '15px'})
        ])

    @app.callback(
        Output('results-table-container', 'children'),  # 컨테이너 업데이트
        [Input('apply-multi-sample', 'n_clicks')],
        [State('multi-sample-data', 'value')]
    )
    def update_results_table(n_clicks, data):
        if n_clicks == 0 or not data:
            return html.Div()  # 빈 div 반환

        # 데이터 파싱
        samples = parsers.parse_multi_sample_data(data)
        if not samples:
            return html.Div("데이터 형식이 올바르지 않습니다")

        # 색공간 정보 가져오기
        color_spaces = color_math.get_color_spaces()

        # 테이블 데이터 생성
        table_data = []

        # 각 샘플에 대한 색재현율 계산 및 테이블 데이터 수집
        for sample in samples:
            # 체크박스 컬럼 추가 (기본값은 빈 문자열)
            row = {'Sample': sample['name'], 'Selected': ''}

            # 각 색공간에 대한 색재현율 계산
            for cs_name in ['DCI-P3', 'adobeRGB', 'BT2020']:
                if cs_name in color_spaces:
                    try:
                        coverage_xy, area_xy, coverage_uv, area_uv = color_math.calculate_gamut_coverage(
                            sample, color_spaces[cs_name])
                        row[f'{cs_name}_xy_coverage'] = f"{coverage_xy:.2f}%"
                        row[f'{cs_name}_xy_coverage_style'] = 'good' if coverage_xy >= 90 else (
                            'bad' if coverage_xy < 70 else 'normal')

                        row[f'{cs_name}_xy_area'] = f"{area_xy:.2f}%"
                        row[f'{cs_name}_xy_area_style'] = 'good' if area_xy >= 90 else (
                            'bad' if area_xy < 70 else 'normal')

                        row[f'{cs_name}_uv_coverage'] = f"{coverage_uv:.2f}%"
                        row[f'{cs_name}_uv_coverage_style'] = 'good' if coverage_uv >= 90 else (
                            'bad' if coverage_uv < 70 else 'normal')

                        row[f'{cs_name}_uv_area'] = f"{area_uv:.2f}%"
                        row[f'{cs_name}_uv_area_style'] = 'good' if area_uv >= 90 else (
                            'bad' if area_uv < 70 else 'normal')

                    except Exception as e:
                        row[f'{cs_name}_xy_coverage'] = "오류"
                        row[f'{cs_name}_xy_area'] = "오류"
                        row[f'{cs_name}_uv_coverage'] = "오류"
                        row[f'{cs_name}_uv_area'] = "오류"

            table_data.append(row)

        # 테이블 컬럼 정의 수정
        columns = [
            {'name': 'Sample', 'id': 'Sample'},
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
            'type': 'text'  # 텍스트 입력으로 변경
        })

        # 테이블 반환
        return dash_table.DataTable(
            id='results-table',
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
                'backgroundColor': '#3498db',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'padding': '10px'
            },
            style_data_conditional=[
                # 각 셀에 대한 조건부 스타일링
                {
                    'if': {'column_id': [f'{cs}_{metric}_style' for cs in ['DCI-P3', 'adobeRGB', 'BT2020']
                                         for metric in ['xy_coverage', 'xy_area', 'uv_coverage', 'uv_area']]},
                    'backgroundColor': 'white'  # 기본 배경색
                },
                # 샘플 열은 회색 배경
                {
                    'if': {'column_id': 'Sample'},
                    'backgroundColor': '#f2f2f2',
                    'fontWeight': 'bold'
                }
            ],
            # 정렬 기능 추가
            sort_action='native',  # 네이티브 정렬 활성화
            sort_mode='single',    # 단일 열 정렬 모드
            # 체크박스 상태 저장
            row_selectable=False,  # 행 선택 비활성화 (체크박스만 사용)
            editable=True,         # 체크박스 편집 가능
            # 체크박스 열만 편집 가능하도록 설정
            column_selectable=False
        )

    @app.callback(
        Output('multi-sample-stats', 'children'),
        [Input('apply-multi-sample', 'n_clicks')],
        [State('multi-sample-data', 'value')]
    )
    def update_multi_sample_stats(n_clicks, data):
        if n_clicks == 0 or not data:
            return html.Div("통계 데이터가 없습니다")

        # 데이터 파싱
        samples = parsers.parse_multi_sample_data(data)
        if not samples or len(samples) < 2:
            return html.Div("통계를 계산하기 위해서는 최소 2개 이상의 샘플이 필요합니다")

        # 색공간 정보 가져오기
        color_spaces = color_math.get_color_spaces()

        # 통계 데이터 저장용 변수
        stats = {
            'DCI-P3': {'xy_coverage': [], 'xy_area': [], 'uv_coverage': [], 'uv_area': []},
            'adobeRGB': {'xy_coverage': [], 'xy_area': [], 'uv_coverage': [], 'uv_area': []},
            'BT2020': {'xy_coverage': [], 'xy_area': [], 'uv_coverage': [], 'uv_area': []}
        }

        # 각 샘플에 대한 색재현율 계산 및 통계 데이터 수집
        for sample in samples:
            for cs_name in ['DCI-P3', 'adobeRGB', 'BT2020']:
                if cs_name in color_spaces:
                    try:
                        coverage_xy, area_xy, coverage_uv, area_uv = color_math.calculate_gamut_coverage(
                            sample, color_spaces[cs_name])

                        stats[cs_name]['xy_coverage'].append(coverage_xy)
                        stats[cs_name]['xy_area'].append(area_xy)
                        stats[cs_name]['uv_coverage'].append(coverage_uv)
                        stats[cs_name]['uv_area'].append(area_uv)
                    except Exception:
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
                        '항목': f'{cs_name} {metric_name}',  # 색공간과 측정항목을 하나로 합침
                        '평균': f"{sum(values) / len(values):.2f}%",
                        '최소값': f"{min(values):.2f}%",
                        '최대값': f"{max(values):.2f}%",
                        '표준편차': f"{pd.Series(values).std():.2f}%"
                    }
                    table_data.append(row)

        # 통계 테이블 컬럼 정의 수정
        columns = [
            {'name': '항목', 'id': '항목'},  # 색공간과 측정항목을 합친 열
            {'name': '평균', 'id': '평균'},
            {'name': '최소값', 'id': '최소값'},
            {'name': '최대값', 'id': '최대값'},
            {'name': '표준편차', 'id': '표준편차'}
        ]

        # 결과 테이블 생성
        return html.Div([
            dash_table.DataTable(
                id='stats-table',
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
                    'backgroundColor': '#3498db',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'padding': '10px'
                },
                style_data_conditional=[
                    # 각 셀에 대한 조건부 스타일링
                    {
                        'if': {'column_id': [f'{cs}_{metric}' for cs in ['DCI-P3', 'adobeRGB', 'BT2020']
                                             for metric in ['xy_coverage', 'xy_area', 'uv_coverage', 'uv_area']]},
                        'backgroundColor': 'white'  # 기본 배경색
                    },
                    # 샘플 열은 회색 배경
                    {
                        'if': {'column_id': '항목'},
                        'backgroundColor': '#f2f2f2',
                        'fontWeight': 'bold'
                    }
                ],
                # 정렬 기능 추가
                sort_action='native',
                sort_mode='single'
            )
        ])

    @app.callback(
        Output('multi-sample-cie-diagram', 'figure'),
        [Input('apply-multi-sample', 'n_clicks'),
         Input('multi-color-space-checklist', 'value'),
         Input('cie-diagram-type', 'value'),
         Input('results-table', 'data_timestamp')],
        [State('results-table', 'data'),
         State('multi-sample-data', 'value')]
    )
    def update_multi_sample_cie_diagram(n_clicks, selected_color_spaces, diagram_type, data_timestamp, table_data, data):
        import plotly.graph_objects as go
        import numpy as np

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
        if n_clicks > 0 and data and table_data:
            # 데이터 파싱
            samples = parsers.parse_multi_sample_data(data)
            if samples:
                # 'O' 또는 '1'이 입력된 샘플만 표시
                selected_samples = [row['Sample'] for row in table_data
                                    if row.get('Selected') and str(row.get('Selected')).upper() in ['O', '1']]

                # 각 샘플의 RGB 좌표 그리기
                for i, sample in enumerate(samples):
                    # 선택된 샘플만 그리기
                    if sample['name'] in selected_samples:
                        if diagram_type == 'xy':
                            # xy 좌표계 사용
                            x_points = [sample['red'][0], sample['green']
                                        [0], sample['blue'][0], sample['red'][0]]
                            y_points = [sample['red'][1], sample['green']
                                        [1], sample['blue'][1], sample['red'][1]]
                        else:  # 'uv'
                            # u'v' 좌표계로 변환
                            x_points = []
                            y_points = []
                            for point in ['red', 'green', 'blue', 'red']:
                                u_prime, v_prime = color_math.xy_to_uv(
                                    sample[point][0], sample[point][1])
                                x_points.append(u_prime)
                                y_points.append(v_prime)

                        # 샘플 삼각형 그리기
                        fig.add_trace(go.Scatter(
                            x=x_points,
                            y=y_points,
                            mode='lines+markers',
                            name=sample['name'],
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
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5

            ),
            autosize=True,
            margin=dict(l=50, r=30, t=50, b=50),
            height=500
        )

        return fig

    # 데이터 저장 콜백
    @app.callback(
        Output('multi-sample-data-store', 'data'),
        [Input('multi-sample-data', 'value')]
    )
    def store_multi_sample_data(data):
        return {'multi_sample_data': data}

    # 저장된 데이터 로드 콜백
    @app.callback(
        Output('multi-sample-data', 'value'),
        [Input('multi-sample-data-store', 'modified_timestamp')],
        [State('multi-sample-data-store', 'data')]
    )
    def load_multi_sample_data(ts, data):
        if ts is None or data is None:
            return ''
        return data.get('multi_sample_data', '')

    # 초기화 버튼 콜백
    @app.callback(
        Output('multi-sample-data', 'value', allow_duplicate=True),
        [Input('reset-multi-sample-inputs', 'n_clicks')],
        prevent_initial_call=True
    )
    def reset_multi_sample_inputs(n_clicks):
        if n_clicks > 0:
            return ''
        return dash.no_update
