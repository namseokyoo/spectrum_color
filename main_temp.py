import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import base64
import io
from scipy.spatial import ConvexHull
import colorsys

# colour-science 라이브러리 임포트
import colour
from colour.plotting import plot_chromaticity_diagram_CIE1931
from shapely.geometry import Polygon

# CIE 1931 표준 관찰자 색매칭 함수 로드 (colour 라이브러리 사용)


def load_cie_cmf():
    # colour 라이브러리에서 CIE 1931 2도 관찰자 색매칭 함수 가져오기
    cmfs = colour.colorimetry.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    wavelengths = cmfs.wavelengths
    x_bar = cmfs.values[:, 0]
    y_bar = cmfs.values[:, 1]
    z_bar = cmfs.values[:, 2]

    return wavelengths, x_bar, y_bar, z_bar

# 표준 색공간 색좌표 (colour 라이브러리 사용)


def get_color_spaces():
    # colour 라이브러리에서 표준 RGB 색공간 정보 가져오기
    color_spaces = {
        'DCI-P3': {
            'red': colour.models.RGB_COLOURSPACES['DCI-P3'].primaries[0],
            'green': colour.models.RGB_COLOURSPACES['DCI-P3'].primaries[1],
            'blue': colour.models.RGB_COLOURSPACES['DCI-P3'].primaries[2],
            'white': colour.models.RGB_COLOURSPACES['DCI-P3'].whitepoint
        },
        'adobeRGB': {
            'red': colour.models.RGB_COLOURSPACES['Adobe RGB (1998)'].primaries[0],
            'green': colour.models.RGB_COLOURSPACES['Adobe RGB (1998)'].primaries[1],
            'blue': colour.models.RGB_COLOURSPACES['Adobe RGB (1998)'].primaries[2],
            'white': colour.models.RGB_COLOURSPACES['Adobe RGB (1998)'].whitepoint
        },
        'BT2020': {
            'red': colour.models.RGB_COLOURSPACES['ITU-R BT.2020'].primaries[0],
            'green': colour.models.RGB_COLOURSPACES['ITU-R BT.2020'].primaries[1],
            'blue': colour.models.RGB_COLOURSPACES['ITU-R BT.2020'].primaries[2],
            'white': colour.models.RGB_COLOURSPACES['ITU-R BT.2020'].whitepoint
        }
    }
    return color_spaces

# 스펙트럼에서 CIE XYZ 및 xy 좌표 계산 (colour 라이브러리 사용)


def spectrum_to_cie(wavelengths, intensities):
    # 스펙트럼 데이터를 colour 라이브러리 형식으로 변환
    spd_data = dict(zip(wavelengths, intensities))
    spd = colour.SpectralDistribution(spd_data)

    # CIE XYZ 계산
    XYZ = colour.sd_to_XYZ(spd)

    # CIE xy 계산
    xy = colour.XYZ_to_xy(XYZ)

    return XYZ[0], XYZ[1], XYZ[2], xy[0], xy[1]

# 삼각형 면적 계산


def triangle_area(vertices):
    # 삼각형의 세 꼭지점 좌표
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]

    # 삼각형 면적 계산 (행렬식 이용)
    area = 0.5 * abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)))
    return area

# 삼각형 겹침 면적 계산 (ConvexHull 교집합 면적)


def overlap_area(vertices1, vertices2):
    from shapely.geometry import Polygon

    # 두 삼각형을 Polygon 객체로 변환
    poly1 = Polygon(vertices1)
    poly2 = Polygon(vertices2)

    # 두 다각형의 교집합 계산
    intersection = poly1.intersection(poly2)

    # 교집합의 면적 반환
    return intersection.area

# CIE xy에서 CIE u'v'로 변환하는 함수


def xy_to_uv(x, y):
    # CIE 1976 u'v' 좌표계로 변환
    denominator = -2*x + 12*y + 3
    u_prime = 4*x / denominator
    v_prime = 9*y / denominator
    return u_prime, v_prime

# 색재현율 계산 함수 수정 - u'v' 계산 추가


def calculate_gamut_coverage(rgb_coords, target_space):
    # RGB 좌표로 삼각형 구성 (xy 좌표계)
    rgb_triangle_xy = np.array([
        [rgb_coords['red'][0], rgb_coords['red'][1]],
        [rgb_coords['green'][0], rgb_coords['green'][1]],
        [rgb_coords['blue'][0], rgb_coords['blue'][1]]
    ])

    # 타겟 색공간 삼각형 구성 (xy 좌표계)
    target_triangle_xy = np.array([
        [target_space['red'][0], target_space['red'][1]],
        [target_space['green'][0], target_space['green'][1]],
        [target_space['blue'][0], target_space['blue'][1]]
    ])

    # xy 좌표계에서 각 삼각형 면적 계산
    rgb_area_xy = triangle_area(rgb_triangle_xy)
    target_area_xy = triangle_area(target_triangle_xy)

    # xy 좌표계에서 겹치는 영역 면적 계산
    overlap_xy = overlap_area(rgb_triangle_xy, target_triangle_xy)

    # xy 좌표계에서 색재현율 계산
    coverage_ratio_xy = overlap_xy / target_area_xy * 100  # 타겟 색공간 대비 겹치는 영역 비율
    area_ratio_xy = rgb_area_xy / target_area_xy * 100     # 타겟 색공간 대비 면적 비율

    # RGB 좌표를 u'v' 좌표계로 변환
    rgb_triangle_uv = np.array([
        xy_to_uv(rgb_coords['red'][0], rgb_coords['red'][1]),
        xy_to_uv(rgb_coords['green'][0], rgb_coords['green'][1]),
        xy_to_uv(rgb_coords['blue'][0], rgb_coords['blue'][1])
    ])

    # 타겟 색공간 좌표를 u'v' 좌표계로 변환
    target_triangle_uv = np.array([
        xy_to_uv(target_space['red'][0], target_space['red'][1]),
        xy_to_uv(target_space['green'][0], target_space['green'][1]),
        xy_to_uv(target_space['blue'][0], target_space['blue'][1])
    ])

    # u'v' 좌표계에서 각 삼각형 면적 계산
    rgb_area_uv = triangle_area(rgb_triangle_uv)
    target_area_uv = triangle_area(target_triangle_uv)

    # u'v' 좌표계에서 겹치는 영역 면적 계산
    overlap_uv = overlap_area(rgb_triangle_uv, target_triangle_uv)

    # u'v' 좌표계에서 색재현율 계산
    coverage_ratio_uv = overlap_uv / target_area_uv * 100  # 타겟 색공간 대비 겹치는 영역 비율
    area_ratio_uv = rgb_area_uv / target_area_uv * 100     # 타겟 색공간 대비 면적 비율

    return (coverage_ratio_xy, area_ratio_xy, coverage_ratio_uv, area_ratio_uv)


# 앱 생성
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# 레이아웃 정의
app.layout = html.Div([
    html.H1("OLED 발광 스펙트럼 분석 및 색재현율 계산 대시보드",
            style={'textAlign': 'center', 'margin': '20px 0', 'color': '#2c3e50', 'fontWeight': 'bold'}),

    # 전체 컨테이너
    html.Div([
        # 좌측 영역 - 스펙트럼 데이터 입력 및 스펙트럼 그래프
        html.Div([
            # 스펙트럼 데이터 입력 섹션 (가로 배열)
            html.Div([
                html.H3("RGB 발광 스펙트럼 입력",
                        style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'color': '#2c3e50', 'marginBottom': '20px'}),

                # RGB 입력 컨테이너 (가로 배열)
                html.Div([
                    # R 스펙트럼 입력
                    html.Div([
                        html.H4("R 스펙트럼", style={
                                'color': '#e74c3c', 'textAlign': 'center', 'marginTop': '0'}),
                        dcc.Textarea(
                            id='input-r-data',
                            placeholder='파장,강도 형식으로 입력',
                            style={'width': '100%', 'height': '150px',
                                   'borderColor': '#e74c3c', 'borderWidth': '2px'},
                        ),
                        html.Button('적용',
                                    id='apply-r-data',
                                    n_clicks=0,
                                    style={'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                                           'padding': '8px 15px', 'margin': '10px 0', 'borderRadius': '4px', 'width': '100%'}),
                        html.Div(id='r-spectrum-info', style={
                                 'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '4px', 'height': '80px'})
                    ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                    # G 스펙트럼 입력
                    html.Div([
                        html.H4("G 스펙트럼", style={
                                'color': '#27ae60', 'textAlign': 'center', 'marginTop': '0'}),
                        dcc.Textarea(
                            id='input-g-data',
                            placeholder='파장,강도 형식으로 입력',
                            style={'width': '100%', 'height': '150px',
                                   'borderColor': '#27ae60', 'borderWidth': '2px'},
                        ),
                        html.Button('적용',
                                    id='apply-g-data',
                                    n_clicks=0,
                                    style={'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none',
                                           'padding': '8px 15px', 'margin': '10px 0', 'borderRadius': '4px', 'width': '100%'}),
                        html.Div(id='g-spectrum-info', style={
                                 'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '4px', 'height': '80px'})
                    ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'}),

                    # B 스펙트럼 입력
                    html.Div([
                        html.H4("B 스펙트럼", style={
                                'color': '#3498db', 'textAlign': 'center', 'marginTop': '0'}),
                        dcc.Textarea(
                            id='input-b-data',
                            placeholder='파장,강도 형식으로 입력',
                            style={'width': '100%', 'height': '150px',
                                   'borderColor': '#3498db', 'borderWidth': '2px'},
                        ),
                        html.Button('적용',
                                    id='apply-b-data',
                                    n_clicks=0,
                                    style={'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                                           'padding': '8px 15px', 'margin': '10px 0', 'borderRadius': '4px', 'width': '100%'}),
                        html.Div(id='b-spectrum-info', style={
                                 'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '4px', 'height': '80px'})
                    ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'}),
                ], style={'display': 'flex', 'justifyContent': 'space-between'}),
            ], style={'padding': '20px', 'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'borderRadius': '8px', 'margin': '0 0 20px 0'}),

            # 스펙트럼 그래프 섹션
            html.Div([
                html.H3("RGB 발광 스펙트럼",
                        style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'color': '#2c3e50'}),

                # 정규화 체크박스 추가
                html.Div([
                    dcc.Checklist(
                        id='normalize-spectrum',
                        options=[
                            {'label': ' 스펙트럼 정규화 (최대값=1)',
                             'value': 'normalize'}
                        ],
                        value=['normalize'],  # 기본값은 체크 상태
                        labelStyle={'fontSize': '14px', 'color': '#2c3e50'}
                    )
                ], style={'marginBottom': '10px'}),

                dcc.Graph(id='spectrum-graph', style={'height': '400px'})
            ], style={'padding': '20px', 'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'borderRadius': '8px'})

        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # 우측 영역 - 색공간 관련 수치와 그래프
        html.Div([
            # 색재현율 결과 섹션 (체크박스 제거)
            html.Div([
                html.H3("색재현율 분석 결과",
                        style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'color': '#2c3e50'}),

                html.Div(id='gamut-results', style={
                         'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '4px', 'minHeight': '150px', 'marginTop': '15px'})
            ], style={'padding': '20px', 'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'borderRadius': '8px', 'margin': '0 0 20px 0'}),

            # CIE 색도 다이어그램 섹션 - 탭 추가
            html.Div([
                html.H3("CIE 색도 다이어그램",
                        style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'color': '#2c3e50'}),

                # 색공간 선택 체크박스 추가 (이동됨)
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

                # 탭 컴포넌트
                dcc.Tabs([
                    dcc.Tab(label='CIE 1931 xy', children=[
                        # 450px에서 30% 증가
                        dcc.Graph(id='cie-diagram', style={'height': '585px'})
                    ], style={'padding': '15px 0'},
                        selected_style={'borderTop': '2px solid #3498db', 'padding': '15px 0'}),

                    dcc.Tab(label='CIE 1976 u\'v\'', children=[
                        dcc.Graph(id='cie-uv-diagram',
                                  style={'height': '585px'})  # 450px에서 30% 증가
                    ], style={'padding': '15px 0'},
                        selected_style={'borderTop': '2px solid #3498db', 'padding': '15px 0'})
                ])
            ], style={'padding': '20px', 'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'borderRadius': '8px'})

        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'})
])

# 텍스트 데이터 파싱 함수


def parse_text_data(text_data):
    try:
        # 텍스트를 줄 단위로 분리
        lines = text_data.strip().split('\n')
        data = []

        for line in lines:
            # 쉼표로 구분된 값 처리
            if ',' in line:
                values = line.strip().split(',')
            # 공백이나 탭으로 분리된 값들을 처리
            else:
                values = line.strip().split()

            if len(values) >= 2:  # 최소 두 개의 값(파장, 강도)이 있어야 함
                try:
                    wavelength = float(values[0])
                    intensity = float(values[1])
                    data.append([wavelength, intensity])
                except ValueError:
                    continue  # 숫자로 변환할 수 없는 행은 건너뜀

        if not data:
            return None

        # 데이터프레임으로 변환
        df = pd.DataFrame(data, columns=['wavelength', 'intensity'])
        return df
    except Exception as e:
        print(f"데이터 파싱 오류: {e}")
        return None

# R 스펙트럼 데이터 콜백


@app.callback(
    Output('r-spectrum-info', 'children'),
    Input('apply-r-data', 'n_clicks'),
    State('input-r-data', 'value')
)
def update_r_info(n_clicks, text_data):
    if n_clicks == 0 or not text_data:
        return html.Div("")

    df = parse_text_data(text_data)
    if df is not None:
        # CIE 좌표 계산
        _, _, _, x, y = spectrum_to_cie(
            df['wavelength'].values, df['intensity'].values)

        # CIE u'v' 좌표 계산
        u_prime, v_prime = xy_to_uv(x, y)

        return html.Div([
            html.P(f"데이터 포인트: {len(df)}개"),
            html.P(f"CIE x: {x:.4f}, CIE y: {y:.4f}"),
            html.P(f"CIE u': {u_prime:.4f}, CIE v': {v_prime:.4f}")
        ])
    else:
        return html.Div("데이터 형식이 올바르지 않습니다.")

# G 스펙트럼 데이터 콜백


@app.callback(
    Output('g-spectrum-info', 'children'),
    Input('apply-g-data', 'n_clicks'),
    State('input-g-data', 'value')
)
def update_g_info(n_clicks, text_data):
    if n_clicks == 0 or not text_data:
        return html.Div("")

    df = parse_text_data(text_data)
    if df is not None:
        # CIE 좌표 계산
        _, _, _, x, y = spectrum_to_cie(
            df['wavelength'].values, df['intensity'].values)

        # CIE u'v' 좌표 계산
        u_prime, v_prime = xy_to_uv(x, y)

        return html.Div([
            html.P(f"데이터 포인트: {len(df)}개"),
            html.P(f"CIE x: {x:.4f}, CIE y: {y:.4f}"),
            html.P(f"CIE u': {u_prime:.4f}, CIE v': {v_prime:.4f}")
        ])
    else:
        return html.Div("데이터 형식이 올바르지 않습니다.")

# B 스펙트럼 데이터 콜백


@app.callback(
    Output('b-spectrum-info', 'children'),
    Input('apply-b-data', 'n_clicks'),
    State('input-b-data', 'value')
)
def update_b_info(n_clicks, text_data):
    if n_clicks == 0 or not text_data:
        return html.Div("")

    df = parse_text_data(text_data)
    if df is not None:
        # CIE 좌표 계산
        _, _, _, x, y = spectrum_to_cie(
            df['wavelength'].values, df['intensity'].values)

        # CIE u'v' 좌표 계산
        u_prime, v_prime = xy_to_uv(x, y)

        return html.Div([
            html.P(f"데이터 포인트: {len(df)}개"),
            html.P(f"CIE x: {x:.4f}, CIE y: {y:.4f}"),
            html.P(f"CIE u': {u_prime:.4f}, CIE v': {v_prime:.4f}")
        ])
    else:
        return html.Div("데이터 형식이 올바르지 않습니다.")

# 스펙트럼 그래프 업데이트 콜백


@app.callback(
    Output('spectrum-graph', 'figure'),
    [Input('apply-r-data', 'n_clicks'),
     Input('apply-g-data', 'n_clicks'),
     Input('apply-b-data', 'n_clicks'),
     Input('normalize-spectrum', 'value')],
    [State('input-r-data', 'value'),
     State('input-g-data', 'value'),
     State('input-b-data', 'value')]
)
def update_spectrum_graph(r_clicks, g_clicks, b_clicks, normalize_option, r_data, g_data, b_data):
    fig = go.Figure()

    # 파장 범위 설정
    x_range = [380, 780]

    # 정규화 여부 확인
    normalize = 'normalize' in normalize_option if normalize_option else False

    # R 스펙트럼 추가
    if r_clicks > 0 and r_data:
        df_r = parse_text_data(r_data)
        if df_r is not None:
            # 정규화 적용
            y_values = df_r['intensity'].values
            if normalize and len(y_values) > 0 and max(y_values) > 0:
                y_values = y_values / max(y_values)

            fig.add_trace(go.Scatter(
                x=df_r['wavelength'],
                y=y_values,
                mode='lines',
                name='R 스펙트럼',
                line=dict(color='red', width=2)
            ))

    # G 스펙트럼 추가
    if g_clicks > 0 and g_data:
        df_g = parse_text_data(g_data)
        if df_g is not None:
            # 정규화 적용
            y_values = df_g['intensity'].values
            if normalize and len(y_values) > 0 and max(y_values) > 0:
                y_values = y_values / max(y_values)

            fig.add_trace(go.Scatter(
                x=df_g['wavelength'],
                y=y_values,
                mode='lines',
                name='G 스펙트럼',
                line=dict(color='green', width=2)
            ))

    # B 스펙트럼 추가
    if b_clicks > 0 and b_data:
        df_b = parse_text_data(b_data)
        if df_b is not None:
            # 정규화 적용
            y_values = df_b['intensity'].values
            if normalize and len(y_values) > 0 and max(y_values) > 0:
                y_values = y_values / max(y_values)

            fig.add_trace(go.Scatter(
                x=df_b['wavelength'],
                y=y_values,
                mode='lines',
                name='B 스펙트럼',
                line=dict(color='blue', width=2)
            ))

    # Y축 제목 설정 (정규화 여부에 따라 다르게)
    y_axis_title = '상대 강도 (정규화)' if normalize else '상대 강도'

    # 그래프 레이아웃 설정
    fig.update_layout(
        title='RGB 발광 스펙트럼',
        xaxis_title='파장 (nm)',
        yaxis_title=y_axis_title,
        xaxis=dict(
            range=x_range,
            showgrid=False,
            zeroline=True,
            zerolinecolor='#bdbdbd',
            zerolinewidth=1,
            showline=True,
            linecolor='#bdbdbd',
            linewidth=1
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=True,
            zerolinecolor='#bdbdbd',
            zerolinewidth=1,
            showline=True,
            linecolor='#bdbdbd',
            linewidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig

# CIE 1931 xy 다이어그램 업데이트 콜백


@app.callback(
    Output('cie-diagram', 'figure'),
    [Input('apply-r-data', 'n_clicks'),
     Input('apply-g-data', 'n_clicks'),
     Input('apply-b-data', 'n_clicks'),
     Input('color-space-checklist', 'value')],
    [State('input-r-data', 'value'),
     State('input-g-data', 'value'),
     State('input-b-data', 'value')]
)
def update_cie_diagram(r_clicks, g_clicks, b_clicks, selected_color_spaces, r_data, g_data, b_data):
    # CIE 1931 색도 다이어그램 배경 이미지를 생성하거나 로드
    # 여기서는 간단히 스펙트럼 궤적만 그림

    fig = go.Figure()

    # 스펙트럼 궤적 그리기 (스펙트럼 색상의 곡선)
    # 더 부드러운 곡선을 위해 간격을 1nm로 줄임
    wavelengths = np.arange(380, 781, 1)  # 1nm 간격으로 변경
    x_coords = []
    y_coords = []

    # 단색광의 xy 좌표 계산
    for wl in wavelengths:
        intensity = np.zeros_like(wavelengths)
        intensity[wavelengths == wl] = 1.0
        _, _, _, x, y = spectrum_to_cie(wavelengths, intensity)
        x_coords.append(x)
        y_coords.append(y)

    # 스펙트럼 궤적 그리기 (더 부드러운 선으로)
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='lines',
        line=dict(color='black', width=1.5, shape='spline'),  # 스플라인 곡선으로 변경
        showlegend=False
    ))

    # 스펙트럼 궤적 닫기 (검정색 실선으로 변경)
    fig.add_trace(go.Scatter(
        x=[x_coords[-1], x_coords[0]],
        y=[y_coords[-1], y_coords[0]],
        mode='lines',
        line=dict(color='black', width=1.5),  # 검정색 실선으로 변경
        showlegend=False
    ))

    # 색공간 삼각형 그리기
    color_spaces = get_color_spaces()
    for cs_name in selected_color_spaces:
        if cs_name in color_spaces:
            cs = color_spaces[cs_name]

            # 색공간 삼각형 좌표
            x_points = [cs['red'][0], cs['green']
                        [0], cs['blue'][0], cs['red'][0]]
            y_points = [cs['red'][1], cs['green']
                        [1], cs['blue'][1], cs['red'][1]]

            # 색공간 삼각형 그리기
            fig.add_trace(go.Scatter(
                x=x_points,
                y=y_points,
                mode='lines+markers',
                name=cs_name,
                line=dict(width=2)
            ))

    # RGB 좌표 계산 및 표시
    rgb_coords = {}

    if r_clicks > 0 and r_data:
        df_r = parse_text_data(r_data)
        if df_r is not None:
            _, _, _, x, y = spectrum_to_cie(
                df_r['wavelength'].values, df_r['intensity'].values)
            rgb_coords['red'] = (x, y)
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                name='R 좌표',
                marker=dict(color='red', size=10)
            ))

    if g_clicks > 0 and g_data:
        df_g = parse_text_data(g_data)
        if df_g is not None:
            _, _, _, x, y = spectrum_to_cie(
                df_g['wavelength'].values, df_g['intensity'].values)
            rgb_coords['green'] = (x, y)
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                name='G 좌표',
                marker=dict(color='green', size=10)
            ))

    if b_clicks > 0 and b_data:
        df_b = parse_text_data(b_data)
        if df_b is not None:
            _, _, _, x, y = spectrum_to_cie(
                df_b['wavelength'].values, df_b['intensity'].values)
            rgb_coords['blue'] = (x, y)
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                name='B 좌표',
                marker=dict(color='blue', size=10)
            ))

    # 모든 RGB 좌표가 있으면 삼각형 그리기
    if 'red' in rgb_coords and 'green' in rgb_coords and 'blue' in rgb_coords:
        x_points = [rgb_coords['red'][0], rgb_coords['green']
                    [0], rgb_coords['blue'][0], rgb_coords['red'][0]]
        y_points = [rgb_coords['red'][1], rgb_coords['green']
                    [1], rgb_coords['blue'][1], rgb_coords['red'][1]]

        fig.add_trace(go.Scatter(
            x=x_points,
            y=y_points,
            mode='lines',
            name='측정 RGB 색역',
            line=dict(color='purple', width=2, dash='dash')
        ))

    # 그래프 레이아웃 설정
    fig.update_layout(
        title='CIE 1931 xy 색도 다이어그램',
        xaxis_title='x',
        yaxis_title='y',
        xaxis=dict(
            range=[0, 0.8],
            constrain='domain',
            showgrid=False,
            zeroline=True,
            zerolinecolor='#bdbdbd',
            zerolinewidth=1,
            showline=True,
            linecolor='#bdbdbd',
            linewidth=1
        ),
        yaxis=dict(
            range=[0, 0.9],
            scaleanchor='x',
            scaleratio=1,
            showgrid=False,
            zeroline=True,
            zerolinecolor='#bdbdbd',
            zerolinewidth=1,
            showline=True,
            linecolor='#bdbdbd',
            linewidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",  # 가로 방향 범례
            yanchor="top",
            y=-0.15,  # 그래프 아래에 배치
            xanchor="center",
            x=0.5,  # 중앙 정렬
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#CCCCCC',
            borderwidth=1
        )
    )

    return fig

# CIE 1976 u'v' 다이어그램 업데이트 콜백


@app.callback(
    Output('cie-uv-diagram', 'figure'),
    [Input('apply-r-data', 'n_clicks'),
     Input('apply-g-data', 'n_clicks'),
     Input('apply-b-data', 'n_clicks'),
     Input('color-space-checklist', 'value')],
    [State('input-r-data', 'value'),
     State('input-g-data', 'value'),
     State('input-b-data', 'value')]
)
def update_cie_uv_diagram(r_clicks, g_clicks, b_clicks, selected_color_spaces, r_data, g_data, b_data):
    # CIE 1976 u'v' 색도 다이어그램 생성
    fig = go.Figure()

    # 스펙트럼 궤적 그리기 (스펙트럼 색상의 곡선)
    wavelengths = np.arange(380, 781, 1)  # 1nm 간격
    u_coords = []
    v_coords = []

    # 단색광의 u'v' 좌표 계산
    for wl in wavelengths:
        intensity = np.zeros_like(wavelengths)
        intensity[wavelengths == wl] = 1.0
        _, _, _, x, y = spectrum_to_cie(wavelengths, intensity)
        u_prime, v_prime = xy_to_uv(x, y)
        u_coords.append(u_prime)
        v_coords.append(v_prime)

    # 스펙트럼 궤적 그리기
    fig.add_trace(go.Scatter(
        x=u_coords,
        y=v_coords,
        mode='lines',
        line=dict(color='black', width=1.5, shape='spline'),
        showlegend=False
    ))

    # 스펙트럼 궤적 닫기
    fig.add_trace(go.Scatter(
        x=[u_coords[-1], u_coords[0]],
        y=[v_coords[-1], v_coords[0]],
        mode='lines',
        line=dict(color='black', width=1.5),
        showlegend=False
    ))

    # 색공간 삼각형 그리기
    color_spaces = get_color_spaces()
    for cs_name in selected_color_spaces:
        if cs_name in color_spaces:
            cs = color_spaces[cs_name]

            # 색공간 삼각형 좌표를 u'v'로 변환
            u_points = []
            v_points = []
            for point in ['red', 'green', 'blue', 'red']:
                u_prime, v_prime = xy_to_uv(cs[point][0], cs[point][1])
                u_points.append(u_prime)
                v_points.append(v_prime)

            # 색공간 삼각형 그리기
            fig.add_trace(go.Scatter(
                x=u_points,
                y=v_points,
                mode='lines+markers',
                name=cs_name,
                line=dict(width=2)
            ))

    # RGB 좌표 계산 및 표시
    rgb_coords_uv = {}

    if r_clicks > 0 and r_data:
        df_r = parse_text_data(r_data)
        if df_r is not None:
            _, _, _, x, y = spectrum_to_cie(
                df_r['wavelength'].values, df_r['intensity'].values)
            u_prime, v_prime = xy_to_uv(x, y)
            rgb_coords_uv['red'] = (u_prime, v_prime)
            fig.add_trace(go.Scatter(
                x=[u_prime],
                y=[v_prime],
                mode='markers',
                name='R 좌표',
                marker=dict(color='red', size=10)
            ))

    if g_clicks > 0 and g_data:
        df_g = parse_text_data(g_data)
        if df_g is not None:
            _, _, _, x, y = spectrum_to_cie(
                df_g['wavelength'].values, df_g['intensity'].values)
            u_prime, v_prime = xy_to_uv(x, y)
            rgb_coords_uv['green'] = (u_prime, v_prime)
            fig.add_trace(go.Scatter(
                x=[u_prime],
                y=[v_prime],
                mode='markers',
                name='G 좌표',
                marker=dict(color='green', size=10)
            ))

    if b_clicks > 0 and b_data:
        df_b = parse_text_data(b_data)
        if df_b is not None:
            _, _, _, x, y = spectrum_to_cie(
                df_b['wavelength'].values, df_b['intensity'].values)
            u_prime, v_prime = xy_to_uv(x, y)
            rgb_coords_uv['blue'] = (u_prime, v_prime)
            fig.add_trace(go.Scatter(
                x=[u_prime],
                y=[v_prime],
                mode='markers',
                name='B 좌표',
                marker=dict(color='blue', size=10)
            ))

    # 모든 RGB 좌표가 있으면 삼각형 그리기
    if 'red' in rgb_coords_uv and 'green' in rgb_coords_uv and 'blue' in rgb_coords_uv:
        u_points = [rgb_coords_uv['red'][0], rgb_coords_uv['green'][0],
                    rgb_coords_uv['blue'][0], rgb_coords_uv['red'][0]]
        v_points = [rgb_coords_uv['red'][1], rgb_coords_uv['green'][1],
                    rgb_coords_uv['blue'][1], rgb_coords_uv['red'][1]]

        fig.add_trace(go.Scatter(
            x=u_points,
            y=v_points,
            mode='lines',
            name='측정 RGB 색역',
            line=dict(color='purple', width=2, dash='dash')
        ))

    # 그래프 레이아웃 설정
    fig.update_layout(
        title='CIE 1976 u\'v\' 색도 다이어그램',
        xaxis_title='u\'',
        yaxis_title='v\'',
        xaxis=dict(
            range=[0, 0.7],
            constrain='domain',
            showgrid=False,
            zeroline=True,
            zerolinecolor='#bdbdbd',
            zerolinewidth=1,
            showline=True,
            linecolor='#bdbdbd',
            linewidth=1
        ),
        yaxis=dict(
            range=[0, 0.6],
            scaleanchor='x',
            scaleratio=1,
            showgrid=False,
            zeroline=True,
            zerolinecolor='#bdbdbd',
            zerolinewidth=1,
            showline=True,
            linecolor='#bdbdbd',
            linewidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",  # 가로 방향 범례
            yanchor="top",
            y=-0.15,  # 그래프 아래에 배치
            xanchor="center",
            x=0.5,  # 중앙 정렬
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#CCCCCC',
            borderwidth=1
        )
    )

    return fig

# 색재현율 결과 업데이트 콜백 - 항상 모든 색공간 계산


@app.callback(
    Output('gamut-results', 'children'),
    [Input('apply-r-data', 'n_clicks'),
     Input('apply-g-data', 'n_clicks'),
     Input('apply-b-data', 'n_clicks')],  # 색공간 체크박스 의존성 제거
    [State('input-r-data', 'value'),
     State('input-g-data', 'value'),
     State('input-b-data', 'value')]
)
def update_gamut_results(r_clicks, g_clicks, b_clicks, r_data, g_data, b_data):
    # RGB 좌표 계산
    rgb_coords = {}

    if r_clicks > 0 and r_data:
        df_r = parse_text_data(r_data)
        if df_r is not None:
            _, _, _, x, y = spectrum_to_cie(
                df_r['wavelength'].values, df_r['intensity'].values)
            rgb_coords['red'] = (x, y)

    if g_clicks > 0 and g_data:
        df_g = parse_text_data(g_data)
        if df_g is not None:
            _, _, _, x, y = spectrum_to_cie(
                df_g['wavelength'].values, df_g['intensity'].values)
            rgb_coords['green'] = (x, y)

    if b_clicks > 0 and b_data:
        df_b = parse_text_data(b_data)
        if df_b is not None:
            _, _, _, x, y = spectrum_to_cie(
                df_b['wavelength'].values, df_b['intensity'].values)
            rgb_coords['blue'] = (x, y)

    # 모든 RGB 좌표가 있는지 확인
    if 'red' in rgb_coords and 'green' in rgb_coords and 'blue' in rgb_coords:
        # 색공간 정보 가져오기
        color_spaces = get_color_spaces()

        # 항상 모든 색공간 사용 (체크박스와 무관하게)
        selected_color_spaces = ['DCI-P3', 'adobeRGB', 'BT2020']

        # 표 헤더
        table_header = [
            html.Thead(html.Tr([
                html.Th("색공간", style={'width': '16%', 'textAlign': 'left',
                        'padding': '10px', 'borderBottom': '2px solid #ddd'}),
                html.Th("CIE xy 중첩 비율", style={
                        'width': '21%', 'textAlign': 'center', 'padding': '10px', 'borderBottom': '2px solid #ddd'}),
                html.Th("CIE xy 면적 비율", style={
                        'width': '21%', 'textAlign': 'center', 'padding': '10px', 'borderBottom': '2px solid #ddd'}),
                html.Th("CIE u'v' 중첩 비율", style={
                        'width': '21%', 'textAlign': 'center', 'padding': '10px', 'borderBottom': '2px solid #ddd'}),
                html.Th("CIE u'v' 면적 비율", style={
                        'width': '21%', 'textAlign': 'center', 'padding': '10px', 'borderBottom': '2px solid #ddd'})
            ]))
        ]

        table_rows = []

        # 선택된 각 색공간에 대한 색재현율 계산
        for cs_name in selected_color_spaces:
            if cs_name in color_spaces:
                cs = color_spaces[cs_name]

                try:
                    coverage_ratio_xy, area_ratio_xy, coverage_ratio_uv, area_ratio_uv = calculate_gamut_coverage(
                        rgb_coords, cs)

                    # 색상 결정 (높은 값은 녹색, 낮은 값은 빨간색)
                    coverage_color_xy = '#27ae60' if coverage_ratio_xy >= 90 else '#e74c3c' if coverage_ratio_xy < 70 else '#f39c12'
                    area_color_xy = '#27ae60' if area_ratio_xy >= 90 else '#e74c3c' if area_ratio_xy < 70 else '#f39c12'
                    coverage_color_uv = '#27ae60' if coverage_ratio_uv >= 90 else '#e74c3c' if coverage_ratio_uv < 70 else '#f39c12'
                    area_color_uv = '#27ae60' if area_ratio_uv >= 90 else '#e74c3c' if area_ratio_uv < 70 else '#f39c12'

                    table_rows.append(html.Tr([
                        html.Td(cs_name, style={
                                'padding': '10px', 'borderBottom': '1px solid #ddd', 'fontWeight': 'bold'}),
                        html.Td(f"{coverage_ratio_xy:.2f}%",
                                style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid #ddd',
                                       'color': coverage_color_xy, 'fontWeight': 'bold'}),
                        html.Td(f"{area_ratio_xy:.2f}%",
                                style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid #ddd',
                                       'color': area_color_xy, 'fontWeight': 'bold'}),
                        html.Td(f"{coverage_ratio_uv:.2f}%",
                                style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid #ddd',
                                       'color': coverage_color_uv, 'fontWeight': 'bold'}),
                        html.Td(f"{area_ratio_uv:.2f}%",
                                style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid #ddd',
                                       'color': area_color_uv, 'fontWeight': 'bold'})
                    ]))
                except Exception as e:
                    table_rows.append(html.Tr([
                        html.Td(cs_name, style={
                                'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                        html.Td(colspan=4, style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid #ddd', 'color': '#e74c3c'},
                                children=f"계산 오류: {str(e)}")
                    ]))

        table_body = [html.Tbody(table_rows)]

        # 표 설명
        table_caption = html.Div([
            html.P("중첩 비율: 타겟 색공간 대비 겹치는 영역의 비율",
                   style={'fontSize': '12px', 'color': '#7f8c8d', 'margin': '10px 0 0 0'}),
            html.P("면적 비율: 타겟 색공간 대비 측정 RGB 색역의 면적 비율",
                   style={'fontSize': '12px', 'color': '#7f8c8d', 'margin': '5px 0 0 0'})
        ])

        # 전체 표 구성
        table = html.Table(
            table_header + table_body,
            style={'width': '100%', 'borderCollapse': 'collapse',
                   'backgroundColor': 'white', 'fontSize': '13px'}
        )

        return html.Div([table, table_caption])
    else:
        return html.Div("")


# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=True)
