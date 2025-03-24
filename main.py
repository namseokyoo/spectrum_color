import dash
from dash import html, dcc, Output, Input, State
from layouts import home_layout, analysis_layout, doe_layout, optical_layout, spectrum_layout
from callbacks import home_callbacks, analysis_callbacks, doe_callbacks, optical_callbacks
from dash import dash_table
from utils import color_math, parsers
import math
import dash_html_components as html
import webbrowser
import threading
import os
import sys
from flask import request

# 앱 생성
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# 브라우저 자동 실행을 위한 함수


def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050")

    # 서버 종료 함수


def shutdown_server():
    os._exit(0)


# 레이아웃 정의
app.layout = html.Div([
    # 데이터 저장을 위한 컴포넌트 추가
    dcc.Store(id='multi-sample-data-store', storage_type='local'),
    dcc.Store(id='doe-data-store', storage_type='local'),
    dcc.Store(id='optical-data-store', storage_type='local'),  # 광학 데이터 저장소 추가

    html.H1("OLED 발광 스펙트럼 분석 및 색재현율 계산 대시보드",
            style={'textAlign': 'center', 'margin': '20px 0', 'color': '#2c3e50', 'fontWeight': 'bold'}),

    # 메인 탭 컴포넌트
    dcc.Tabs([
        # 첫 번째 탭 - 스펙트럼 분석
        home_layout.create_spectrum_tab(),

        # 두 번째 탭 - 다중 샘플 분석
        analysis_layout.create_analysis_tab(),

        # 세 번째 탭 - DOE 분석
        doe_layout.create_doe_tab(),

        # 네 번째 탭 - 광학 상수 추출
        optical_layout.create_optical_tab()
    ], style={
        'margin': '0 20px',
        'height': '40px',  # 탭 높이 증가
        'lineHeight': '40px',  # 텍스트 세로 중앙 정렬을 위한 라인 높이 설정
        'verticalAlign': 'middle',  # 세로 중앙 정렬
    })
])

# 콜백 등록
home_callbacks.register_callbacks(app)
analysis_callbacks.register_callbacks(app)
doe_callbacks.register_callbacks(app)
optical_callbacks.register_callbacks(app)

# 다중 샘플 분석 결과 테이블 생성 함수


def create_results_table(samples, color_spaces):
    # 테이블 컬럼 정의
    columns = [
        {'name': '선택', 'id': 'select', 'type': 'text',
            'presentation': 'markdown', 'sort_action': 'none'},
        {'name': '샘플 번호', 'id': 'sample_no', 'type': 'numeric'},
        {'name': 'Rx', 'id': 'rx', 'type': 'numeric',
            'format': {'specifier': '.4f'}},
        {'name': 'Ry', 'id': 'ry', 'type': 'numeric',
            'format': {'specifier': '.4f'}},
        {'name': 'Gx', 'id': 'gx', 'type': 'numeric',
            'format': {'specifier': '.4f'}},
        {'name': 'Gy', 'id': 'gy', 'type': 'numeric',
            'format': {'specifier': '.4f'}},
        {'name': 'Bx', 'id': 'bx', 'type': 'numeric',
            'format': {'specifier': '.4f'}},
        {'name': 'By', 'id': 'by', 'type': 'numeric', 'format': {'specifier': '.4f'}}
    ]

    # 각 색공간에 대한 컬럼 추가
    for cs_name in color_spaces:
        columns.extend([
            {'name': f'{cs_name} xy 중첩(%)', 'id': f'{cs_name}_xy_coverage', 'type': 'numeric', 'format': {
                'specifier': '.2f'}},
            {'name': f'{cs_name} xy 면적(%)', 'id': f'{cs_name}_xy_area', 'type': 'numeric', 'format': {
                'specifier': '.2f'}},
            {'name': f'{cs_name} uv 중첩(%)', 'id': f'{cs_name}_uv_coverage', 'type': 'numeric', 'format': {
                'specifier': '.2f'}},
            {'name': f'{cs_name} uv 면적(%)', 'id': f'{cs_name}_uv_area', 'type': 'numeric', 'format': {
                'specifier': '.2f'}}
        ])

    # 데이터 생성
    data = []
    for sample in samples:
        row = {
            'select': '- [x] 선택',  # 마크다운 형식의 체크박스 (기본 선택)
            'sample_no': sample['no'],
            'rx': sample['red'][0],
            'ry': sample['red'][1],
            'gx': sample['green'][0],
            'gy': sample['green'][1],
            'bx': sample['blue'][0],
            'by': sample['blue'][1]
        }

        # 각 색공간에 대한 색재현율 계산 및 추가
        for cs_name, cs in color_spaces.items():
            sample_data = {
                'red': sample['red'],
                'green': sample['green'],
                'blue': sample['blue']
            }
            coverage_xy, area_xy, coverage_uv, area_uv = color_math.calculate_gamut_coverage(
                sample_data, cs)
            row[f'{cs_name}_xy_coverage'] = coverage_xy
            row[f'{cs_name}_xy_area'] = area_xy
            row[f'{cs_name}_uv_coverage'] = coverage_uv
            row[f'{cs_name}_uv_area'] = area_uv

        data.append(row)

    # 테이블 생성
    table = dash_table.DataTable(
        id='results-table',
        columns=columns,
        data=data,
        style_table={
            'overflowX': 'auto',
            'overflowY': 'auto',
            'height': '500px',  # 고정 높이 설정
            'minWidth': '100%'
        },
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
        sort_action='native',  # 정렬 기능 활성화
        filter_action='native',  # 필터 기능 활성화
        page_action='none',  # 페이지네이션 비활성화
        markdown_options={'html': True},  # 마크다운 HTML 지원
        css=[{
            'selector': '.dash-spreadsheet-container',
            'rule': 'max-height: 500px; overflow-y: auto;'
        }],
        sort_by=[]  # 초기 정렬 없음
    )

    return table, data


@app.callback(
    Output('results-table-container', 'children', allow_duplicate=True),
    [Input('apply-multi-sample', 'n_clicks')],
    [State('multi-sample-data', 'value')],
    prevent_initial_call=True
)
def update_results_table(n_clicks, data):
    if n_clicks == 0 or not data:
        return html.Div()

    # 데이터 파싱
    samples = parsers.parse_multi_sample_data(data)
    if not samples:
        return html.Div("데이터 형식이 올바르지 않습니다. 샘플 이름, Rx, Ry, Gx, Gy, Bx, By 형식으로 입력하세요.")

    # 색공간 정보 가져오기
    color_spaces = color_math.get_color_spaces()

    # 테이블 데이터 생성
    table_data = []

    # 각 샘플에 대한 색재현율 계산 및 테이블 데이터 수집
    for sample in samples:
        try:
            row = {'Sample': sample['name'], 'Selected': ''}

            for cs_name in ['DCI-P3', 'adobeRGB', 'BT2020']:
                if cs_name in color_spaces:
                    try:
                        sample_data = {
                            'red': sample['red'],
                            'green': sample['green'],
                            'blue': sample['blue']
                        }
                        coverage_xy, area_xy, coverage_uv, area_uv = color_math.calculate_gamut_coverage(
                            sample_data, color_spaces[cs_name])

                        # 결과가 NaN이나 무한대인 경우 처리
                        if (math.isnan(coverage_xy) or math.isinf(coverage_xy) or
                            math.isnan(area_xy) or math.isinf(area_xy) or
                            math.isnan(coverage_uv) or math.isinf(coverage_uv) or
                                math.isnan(area_uv) or math.isinf(area_uv)):
                            row[f'{cs_name}_xy_coverage'] = '계산 불가'
                            row[f'{cs_name}_xy_area'] = '계산 불가'
                            row[f'{cs_name}_uv_coverage'] = '계산 불가'
                            row[f'{cs_name}_uv_area'] = '계산 불가'
                        else:
                            row[f'{cs_name}_xy_coverage'] = f'{coverage_xy:.2%}'
                            row[f'{cs_name}_xy_area'] = f'{area_xy:.2%}'
                            row[f'{cs_name}_uv_coverage'] = f'{coverage_uv:.2%}'
                            row[f'{cs_name}_uv_area'] = f'{area_uv:.2%}'
                    except Exception as e:
                        # 오류 발생 시 메시지 대신 '계산 불가' 표시
                        row[f'{cs_name}_xy_coverage'] = '계산 불가'
                        row[f'{cs_name}_xy_area'] = '계산 불가'
                        row[f'{cs_name}_uv_coverage'] = '계산 불가'
                        row[f'{cs_name}_uv_area'] = '계산 불가'

            table_data.append(row)
        except Exception as e:
            # 개별 샘플 처리 중 오류 발생 시 해당 샘플 건너뛰기
            continue

    # 결과 테이블 생성
    # ... 나머지 코드 ...

    @app.server.route('/shutdown', methods=['POST'])
    def shutdown():
        shutdown_server()
        return "Server shutting down..."


# 앱 실행
if __name__ == '__main__':
    threading.Timer(1.5, open_browser).start()
    try:
        app.run_server(debug=False, host="127.0.0.1", port=8050)
    except KeyboardInterrupt:
        shutdown_server()

# version 1.0.1
# 모드에 따라 스펙트럼 그래프 표시 추가
