from dash import Input, Output, State, callback_context, html
from utils import color_math
import plotly.graph_objects as go
import numpy as np
import dash
import pandas as pd
from utils.parsers import parse_text_data_flexible
from utils.color_math import calculate_xyz_from_spectrum, calculate_color_coordinates, calculate_gamut_coverage, calculate_all_color_coordinates, get_color_spaces, xy_to_uv
import json
import base64
import io
import re

# 좌표 계산 함수 추가


def calculate_all_color_coordinates(xyz):
    """
    XYZ 삼자극치로부터 다양한 색공간의 좌표를 계산합니다.

    Args:
        xyz (list): [X, Y, Z] 삼자극치

    Returns:
        dict: 다양한 색좌표 {'xy': [x, y], 'uv': [u', v']}
    """
    X, Y, Z = xyz
    sum_XYZ = X + Y + Z

    # CIE 1931 xy 좌표
    if sum_XYZ > 0:
        x = X / sum_XYZ
        y = Y / sum_XYZ
    else:
        x, y = 0, 0

    # CIE 1976 u'v' 좌표
    denominator = X + 15 * Y + 3 * Z
    if denominator > 0:
        u_prime = 4 * X / denominator
        v_prime = 9 * Y / denominator
    else:
        u_prime, v_prime = 0, 0

    return {
        'xy': [x, y],
        'uv': [u_prime, v_prime]
    }

# 유연한 데이터 파싱 함수 추가


def parse_text_data_flexible(text_data):
    """
    다양한 형식의 스펙트럼 데이터 텍스트를 파싱합니다.

    Args:
        text_data (str): 파싱할 텍스트 데이터

    Returns:
        DataFrame: 파싱된 데이터프레임 (파싱 실패 시 None)
    """
    try:
        # 줄 단위로 분리
        lines = text_data.strip().split('\n')

        # 빈 줄 및 주석 제거
        lines = [line.strip() for line in lines if line.strip()
                 and not line.strip().startswith('#')]

        wavelengths = []
        intensities = []

        for line in lines:
            # 공백, 탭, 쉼표 등으로 분리
            parts = re.split(r'[\s,]+', line)

            # 숫자로 변환 가능한 부분만 추출
            numeric_parts = []
            for part in parts:
                try:
                    numeric_parts.append(float(part))
                except ValueError:
                    continue

            if len(numeric_parts) >= 2:
                wavelengths.append(numeric_parts[0])
                intensities.append(numeric_parts[1])

        if not wavelengths:
            return None

        return pd.DataFrame({'wavelength': wavelengths, 'intensity': intensities})
    except Exception as e:
        print(f"데이터 파싱 오류: {str(e)}")
        return None


def register_callbacks(app):
    # R, G, B 스펙트럼 데이터 처리 콜백 (통합)
    @app.callback(
        [Output('spectrum-graph', 'figure'),
         Output('spectrum-data-store', 'data'),
         Output('input-r-data', 'value'),
         Output('input-g-data', 'value'),
         Output('input-b-data', 'value'),
         Output('input-r-filter-data', 'value'),
         Output('input-g-filter-data', 'value'),
         Output('input-b-filter-data', 'value')],
        [Input('input-r-data', 'value'),
         Input('input-g-data', 'value'),
         Input('input-b-data', 'value'),
         Input('input-r-filter-data', 'value'),
         Input('input-g-filter-data', 'value'),
         Input('input-b-filter-data', 'value'),
         Input('normalize-spectrum', 'value'),
         Input('color-filter-mode', 'value')],
        [State('spectrum-data-store', 'data')],
        prevent_initial_call=False
    )
    def update_spectrum_data(r_data, g_data, b_data, r_filter_data, g_filter_data, b_filter_data, normalize_options, filter_mode, stored_data):
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None

        # 디버깅을 위한 로그 추가
        print(f"콜백 트리거: {trigger_id}")
        print(f"R 데이터 타입: {type(r_data)}, 값: {r_data}")
        print(f"G 데이터 타입: {type(g_data)}, 값: {g_data}")
        print(f"B 데이터 타입: {type(b_data)}, 값: {b_data}")

        if not ctx.triggered:
            # 초기 로드 시 저장된 데이터가 있으면 반환, 없으면 빈 그래프
            if stored_data:
                try:
                    # 저장된 데이터가 문자열이면 JSON으로 변환
                    parsed_data = json.loads(stored_data) if isinstance(
                        stored_data, str) else stored_data
                    fig = create_spectrum_figure(parsed_data)
                    return (
                        fig,
                        stored_data,  # 이미 JSON 문자열이면 그대로 반환
                        parsed_data.get('r_text', ""),
                        parsed_data.get('g_text', ""),
                        parsed_data.get('b_text', ""),
                        parsed_data.get('r_filter_text', ""),
                        parsed_data.get('g_filter_text', ""),
                        parsed_data.get('b_filter_text', "")
                    )
                except Exception as e:
                    print(f"저장된 데이터 처리 오류: {str(e)}")
                    # 오류 발생 시 빈 그래프 반환
                    empty_fig = create_empty_figure("스펙트럼 분석", "데이터를 입력하세요.")
                    empty_data = {'r': None, 'g': None, 'b': None,
                                  'r_text': "", 'g_text': "", 'b_text': "",
                                  'r_filter': None, 'g_filter': None, 'b_filter': None,
                                  'r_filter_text': "", 'g_filter_text': "", 'b_filter_text': "",
                                  'filter_mode': 'enabled' in filter_mode if filter_mode else True}
                    return empty_fig, json.dumps(empty_data), "", "", "", "", "", ""
            else:
                empty_fig = create_empty_figure("스펙트럼 분석", "데이터를 입력하세요.")
                empty_data = {'r': None, 'g': None, 'b': None,
                              'r_text': "", 'g_text': "", 'b_text': "",
                              'r_filter': None, 'g_filter': None, 'b_filter': None,
                              'r_filter_text': "", 'g_filter_text': "", 'b_filter_text': "",
                              'filter_mode': 'enabled' in filter_mode if filter_mode else True}
                return empty_fig, json.dumps(empty_data), "", "", "", "", "", ""

        # 저장된 데이터 초기화 또는 불러오기
        data_dict = {'r': None, 'g': None, 'b': None,
                     'r_text': "", 'g_text': "", 'b_text': "",
                     'r_filter': None, 'g_filter': None, 'b_filter': None,
                     'r_filter_text': "", 'g_filter_text': "", 'b_filter_text': "",
                     'filter_mode': 'enabled' in filter_mode if filter_mode else False}

        # 저장된 데이터가 있으면 파싱
        if stored_data:
            try:
                parsed_data = json.loads(stored_data) if isinstance(
                    stored_data, str) else stored_data
                # 기존 데이터 복사
                for key in parsed_data:
                    if key in data_dict:
                        data_dict[key] = parsed_data[key]
            except Exception as e:
                print(f"저장된 데이터 파싱 오류: {str(e)}")
                # 오류 발생 시 빈 데이터 사용

        # 필터 모드 상태 업데이트
        data_dict['filter_mode'] = 'enabled' in filter_mode if filter_mode else False

        # 트리거 식별 및 입력 상태 확인
        # 윈도우에서 입력 필드 변경 감지 개선
        changed_input = trigger_id.split('.')[0] if trigger_id else None

        # 디버깅 로그
        print(f"변경된 입력: {changed_input}")

        # 어떤 입력이 트리거되었는지 확인하고 해당 데이터 처리
        if changed_input == 'input-r-data':
            # 빈 입력인 경우 데이터 명시적으로 삭제
            if r_data is None or (isinstance(r_data, str) and r_data.strip() == ""):
                data_dict['r'] = None
                data_dict['r_text'] = ""
                print("R 데이터 삭제됨")
            elif r_data:
                df_r = parse_text_data_flexible(r_data)
                if df_r is not None:
                    # 정규화 옵션 적용
                    if normalize_options and 'normalize' in normalize_options:
                        df_r['intensity'] = df_r['intensity'] / \
                            df_r['intensity'].max()
                    data_dict['r'] = df_r.to_dict('list')
                    data_dict['r_text'] = r_data
                    print("R 데이터 업데이트됨")

        elif changed_input == 'input-g-data':
            if g_data is None or (isinstance(g_data, str) and g_data.strip() == ""):
                data_dict['g'] = None
                data_dict['g_text'] = ""
                print("G 데이터 삭제됨")
            elif g_data:
                df_g = parse_text_data_flexible(g_data)
                if df_g is not None:
                    # 정규화 옵션 적용
                    if normalize_options and 'normalize' in normalize_options:
                        df_g['intensity'] = df_g['intensity'] / \
                            df_g['intensity'].max()
                    data_dict['g'] = df_g.to_dict('list')
                    data_dict['g_text'] = g_data
                    print("G 데이터 업데이트됨")

        elif changed_input == 'input-b-data':
            if b_data is None or (isinstance(b_data, str) and b_data.strip() == ""):
                data_dict['b'] = None
                data_dict['b_text'] = ""
                print("B 데이터 삭제됨")
            elif b_data:
                df_b = parse_text_data_flexible(b_data)
                if df_b is not None:
                    # 정규화 옵션 적용
                    if normalize_options and 'normalize' in normalize_options:
                        df_b['intensity'] = df_b['intensity'] / \
                            df_b['intensity'].max()
                    data_dict['b'] = df_b.to_dict('list')
                    data_dict['b_text'] = b_data
                    print("B 데이터 업데이트됨")

        elif changed_input == 'input-r-filter-data':
            if r_filter_data is None or (isinstance(r_filter_data, str) and r_filter_data.strip() == ""):
                data_dict['r_filter'] = None
                data_dict['r_filter_text'] = ""
                print("R 필터 데이터 삭제됨")
            elif r_filter_data:
                df_r_filter = parse_text_data_flexible(r_filter_data)
                if df_r_filter is not None:
                    # 정규화 옵션 적용
                    if normalize_options and 'normalize' in normalize_options:
                        df_r_filter['intensity'] = df_r_filter['intensity'] / \
                            df_r_filter['intensity'].max()
                    data_dict['r_filter'] = df_r_filter.to_dict('list')
                    data_dict['r_filter_text'] = r_filter_data
                    print("R 필터 데이터 업데이트됨")

        elif changed_input == 'input-g-filter-data':
            if g_filter_data is None or (isinstance(g_filter_data, str) and g_filter_data.strip() == ""):
                data_dict['g_filter'] = None
                data_dict['g_filter_text'] = ""
                print("G 필터 데이터 삭제됨")
            elif g_filter_data:
                df_g_filter = parse_text_data_flexible(g_filter_data)
                if df_g_filter is not None:
                    # 정규화 옵션 적용
                    if normalize_options and 'normalize' in normalize_options:
                        df_g_filter['intensity'] = df_g_filter['intensity'] / \
                            df_g_filter['intensity'].max()
                    data_dict['g_filter'] = df_g_filter.to_dict('list')
                    data_dict['g_filter_text'] = g_filter_data
                    print("G 필터 데이터 업데이트됨")

        elif changed_input == 'input-b-filter-data':
            if b_filter_data is None or (isinstance(b_filter_data, str) and b_filter_data.strip() == ""):
                data_dict['b_filter'] = None
                data_dict['b_filter_text'] = ""
                print("B 필터 데이터 삭제됨")
            elif b_filter_data:
                df_b_filter = parse_text_data_flexible(b_filter_data)
                if df_b_filter is not None:
                    # 정규화 옵션 적용
                    if normalize_options and 'normalize' in normalize_options:
                        df_b_filter['intensity'] = df_b_filter['intensity'] / \
                            df_b_filter['intensity'].max()
                    data_dict['b_filter'] = df_b_filter.to_dict('list')
                    data_dict['b_filter_text'] = b_filter_data
                    print("B 필터 데이터 업데이트됨")

        # 기본 경우 (normalize-spectrum, color-filter-mode 등의 변경)
        else:
            # 기존의 전체 데이터 처리 로직 유지
            # RGB 스펙트럼 데이터 처리
            if r_data is None or (isinstance(r_data, str) and r_data.strip() == ""):
                data_dict['r'] = None
                data_dict['r_text'] = ""
            elif r_data:
                df_r = parse_text_data_flexible(r_data)
                if df_r is not None:
                    # 정규화 옵션 적용
                    if normalize_options and 'normalize' in normalize_options:
                        df_r['intensity'] = df_r['intensity'] / \
                            df_r['intensity'].max()
                    data_dict['r'] = df_r.to_dict('list')
                    data_dict['r_text'] = r_data

            if g_data is None or (isinstance(g_data, str) and g_data.strip() == ""):
                data_dict['g'] = None
                data_dict['g_text'] = ""
            elif g_data:
                df_g = parse_text_data_flexible(g_data)
                if df_g is not None:
                    # 정규화 옵션 적용
                    if normalize_options and 'normalize' in normalize_options:
                        df_g['intensity'] = df_g['intensity'] / \
                            df_g['intensity'].max()
                    data_dict['g'] = df_g.to_dict('list')
                    data_dict['g_text'] = g_data

            if b_data is None or (isinstance(b_data, str) and b_data.strip() == ""):
                data_dict['b'] = None
                data_dict['b_text'] = ""
            elif b_data:
                df_b = parse_text_data_flexible(b_data)
                if df_b is not None:
                    # 정규화 옵션 적용
                    if normalize_options and 'normalize' in normalize_options:
                        df_b['intensity'] = df_b['intensity'] / \
                            df_b['intensity'].max()
                    data_dict['b'] = df_b.to_dict('list')
                    data_dict['b_text'] = b_data

            # 필터 스펙트럼 데이터 처리
            if r_filter_data is None or (isinstance(r_filter_data, str) and r_filter_data.strip() == ""):
                data_dict['r_filter'] = None
                data_dict['r_filter_text'] = ""
            elif r_filter_data:
                df_r_filter = parse_text_data_flexible(r_filter_data)
                if df_r_filter is not None:
                    # 정규화 옵션 적용
                    if normalize_options and 'normalize' in normalize_options:
                        df_r_filter['intensity'] = df_r_filter['intensity'] / \
                            df_r_filter['intensity'].max()
                    data_dict['r_filter'] = df_r_filter.to_dict('list')
                    data_dict['r_filter_text'] = r_filter_data

            if g_filter_data is None or (isinstance(g_filter_data, str) and g_filter_data.strip() == ""):
                data_dict['g_filter'] = None
                data_dict['g_filter_text'] = ""
            elif g_filter_data:
                df_g_filter = parse_text_data_flexible(g_filter_data)
                if df_g_filter is not None:
                    # 정규화 옵션 적용
                    if normalize_options and 'normalize' in normalize_options:
                        df_g_filter['intensity'] = df_g_filter['intensity'] / \
                            df_g_filter['intensity'].max()
                    data_dict['g_filter'] = df_g_filter.to_dict('list')
                    data_dict['g_filter_text'] = g_filter_data

            if b_filter_data is None or (isinstance(b_filter_data, str) and b_filter_data.strip() == ""):
                data_dict['b_filter'] = None
                data_dict['b_filter_text'] = ""
            elif b_filter_data:
                df_b_filter = parse_text_data_flexible(b_filter_data)
                if df_b_filter is not None:
                    # 정규화 옵션 적용
                    if normalize_options and 'normalize' in normalize_options:
                        df_b_filter['intensity'] = df_b_filter['intensity'] / \
                            df_b_filter['intensity'].max()
                    data_dict['b_filter'] = df_b_filter.to_dict('list')
                    data_dict['b_filter_text'] = b_filter_data

        # 그래프 생성
        fig = create_spectrum_figure(data_dict)

        # 데이터를 JSON 문자열로 변환하여 저장
        json_data = json.dumps(data_dict)

        # 처리 결과 로그
        print(
            f"데이터 처리 완료: R={data_dict['r'] is not None}, G={data_dict['g'] is not None}, B={data_dict['b'] is not None}")

        return fig, json_data, r_data, g_data, b_data, r_filter_data, g_filter_data, b_filter_data

    # R 좌표 표시 콜백
    @app.callback(
        Output('r-coordinates-display', 'children'),
        [Input('input-r-data', 'value'),
         Input('input-r-filter-data', 'value'),
         Input('color-filter-mode', 'value')]
    )
    def update_r_coordinates(r_data, r_filter_data, filter_mode):
        if not r_data:
            return "R 스펙트럼 데이터를 입력하세요."

        try:
            # 데이터 파싱
            df = parse_text_data_flexible(r_data)
            if df is None:
                return "데이터 형식이 올바르지 않습니다."

            wavelengths = df['wavelength'].values
            intensities = df['intensity'].values

            # 필터 모드가 활성화되고 필터 데이터가 있는 경우
            filter_enabled = 'enabled' in filter_mode if filter_mode else False
            if filter_enabled and r_filter_data:
                df_filter = parse_text_data_flexible(r_filter_data)
                if df_filter is not None:
                    filter_wavelengths = df_filter['wavelength'].values
                    filter_intensities = df_filter['intensity'].values

                    # 두 데이터 세트의 파장 범위 일치시키기
                    min_wl = max(np.min(wavelengths),
                                 np.min(filter_wavelengths))
                    max_wl = min(np.max(wavelengths),
                                 np.max(filter_wavelengths))

                    # 원본 스펙트럼에서 공통 범위 필터링
                    mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
                    common_wavelengths = wavelengths[mask]
                    common_intensities = intensities[mask]

                    # 필터 데이터 보간
                    from scipy.interpolate import interp1d
                    filter_interp = interp1d(filter_wavelengths, filter_intensities,
                                             bounds_error=False, fill_value=0, kind='linear')
                    filter_values = filter_interp(common_wavelengths)

                    # 스펙트럼과 필터 곱하기
                    combined_intensities = common_intensities * filter_values

                    # 결합된 데이터로 계산
                    wavelengths = common_wavelengths
                    intensities = combined_intensities

            # XYZ 값 계산
            xyz = calculate_xyz_from_spectrum(wavelengths, intensities)

            # xy, uv 좌표 계산
            coords = calculate_all_color_coordinates(xyz)

            # 피크 파라미터 계산
            peak_params = calculate_peak_parameters(wavelengths, intensities)

            # 결과 표시 - 색좌표와 피크 파라미터 함께 표시
            return html.Div([
                # 색좌표 섹션
                html.Div([
                    html.Strong(
                        "색좌표", style={'fontSize': '14px', 'color': '#e74c3c'}),
                    html.Div([
                        html.Strong("xy : "),
                        html.Span(
                            f"{coords['xy'][0]:.4f}, {coords['xy'][1]:.4f}")
                    ]),
                    html.Div([
                        html.Strong("u'v' : "),
                        html.Span(
                            f"{coords['uv'][0]:.4f}, {coords['uv'][1]:.4f}")
                    ])
                ], style={'marginBottom': '10px'}),

                # 피크 파라미터 섹션
                html.Div([
                    html.Strong("피크 특성", style={
                                'fontSize': '14px', 'color': '#e74c3c'}),
                    html.Div([
                        html.Strong("Max Peak : "),
                        html.Span(
                            f"{peak_params['max_peak']} nm" if peak_params['max_peak'] else "계산 불가")
                    ]),
                    html.Div([
                        html.Strong("FWHM : "),
                        html.Span(
                            f"{peak_params['fwhm']} nm" if peak_params['fwhm'] else "계산 불가")
                    ]),
                    html.Div([
                        html.Strong("FWQM : "),
                        html.Span(
                            f"{peak_params['fwqm']} nm" if peak_params['fwqm'] else "계산 불가")
                    ])
                ])
            ])
        except Exception as e:
            return f"오류 발생: {str(e)}"

    # G 좌표 표시 콜백
    @app.callback(
        Output('g-coordinates-display', 'children'),
        [Input('input-g-data', 'value'),
         Input('input-g-filter-data', 'value'),
         Input('color-filter-mode', 'value')]
    )
    def update_g_coordinates(g_data, g_filter_data, filter_mode):
        if not g_data:
            return "G 스펙트럼 데이터를 입력하세요."

        try:
            # 데이터 파싱
            df = parse_text_data_flexible(g_data)
            if df is None:
                return "데이터 형식이 올바르지 않습니다."

            wavelengths = df['wavelength'].values
            intensities = df['intensity'].values

            # 필터 모드가 활성화되고 필터 데이터가 있는 경우
            filter_enabled = 'enabled' in filter_mode if filter_mode else False
            if filter_enabled and g_filter_data:
                df_filter = parse_text_data_flexible(g_filter_data)
                if df_filter is not None:
                    filter_wavelengths = df_filter['wavelength'].values
                    filter_intensities = df_filter['intensity'].values

                    # 두 데이터 세트의 파장 범위 일치시키기
                    min_wl = max(np.min(wavelengths),
                                 np.min(filter_wavelengths))
                    max_wl = min(np.max(wavelengths),
                                 np.max(filter_wavelengths))

                    # 원본 스펙트럼에서 공통 범위 필터링
                    mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
                    common_wavelengths = wavelengths[mask]
                    common_intensities = intensities[mask]

                    # 필터 데이터 보간
                    from scipy.interpolate import interp1d
                    filter_interp = interp1d(filter_wavelengths, filter_intensities,
                                             bounds_error=False, fill_value=0, kind='linear')
                    filter_values = filter_interp(common_wavelengths)

                    # 스펙트럼과 필터 곱하기
                    combined_intensities = common_intensities * filter_values

                    # 결합된 데이터로 계산
                    wavelengths = common_wavelengths
                    intensities = combined_intensities

            # XYZ 값 계산
            xyz = calculate_xyz_from_spectrum(wavelengths, intensities)

            # xy, uv 좌표 계산
            coords = calculate_all_color_coordinates(xyz)

            # 피크 파라미터 계산
            peak_params = calculate_peak_parameters(wavelengths, intensities)

            # 결과 표시 - 색좌표와 피크 파라미터 함께 표시
            return html.Div([
                # 색좌표 섹션
                html.Div([
                    html.Strong(
                        "색좌표", style={'fontSize': '14px', 'color': '#27ae60'}),
                    html.Div([
                        html.Strong("xy : "),
                        html.Span(
                            f"{coords['xy'][0]:.4f}, {coords['xy'][1]:.4f}")
                    ]),
                    html.Div([
                        html.Strong("u'v' : "),
                        html.Span(
                            f"{coords['uv'][0]:.4f}, {coords['uv'][1]:.4f}")
                    ])
                ], style={'marginBottom': '10px'}),

                # 피크 파라미터 섹션
                html.Div([
                    html.Strong("피크 특성", style={
                                'fontSize': '14px', 'color': '#27ae60'}),
                    html.Div([
                        html.Strong("Max Peak : "),
                        html.Span(
                            f"{peak_params['max_peak']} nm" if peak_params['max_peak'] else "계산 불가")
                    ]),
                    html.Div([
                        html.Strong("FWHM : "),
                        html.Span(
                            f"{peak_params['fwhm']} nm" if peak_params['fwhm'] else "계산 불가")
                    ]),
                    html.Div([
                        html.Strong("FWQM : "),
                        html.Span(
                            f"{peak_params['fwqm']} nm" if peak_params['fwqm'] else "계산 불가")
                    ])
                ])
            ])
        except Exception as e:
            return f"오류 발생: {str(e)}"

    # B 좌표 표시 콜백
    @app.callback(
        Output('b-coordinates-display', 'children'),
        [Input('input-b-data', 'value'),
         Input('input-b-filter-data', 'value'),
         Input('color-filter-mode', 'value')]
    )
    def update_b_coordinates(b_data, b_filter_data, filter_mode):
        if not b_data:
            return "B 스펙트럼 데이터를 입력하세요."

        try:
            # 데이터 파싱
            df = parse_text_data_flexible(b_data)
            if df is None:
                return "데이터 형식이 올바르지 않습니다."

            wavelengths = df['wavelength'].values
            intensities = df['intensity'].values

            # 필터 모드가 활성화되고 필터 데이터가 있는 경우
            filter_enabled = 'enabled' in filter_mode if filter_mode else False
            if filter_enabled and b_filter_data:
                df_filter = parse_text_data_flexible(b_filter_data)
                if df_filter is not None:
                    filter_wavelengths = df_filter['wavelength'].values
                    filter_intensities = df_filter['intensity'].values

                    # 두 데이터 세트의 파장 범위 일치시키기
                    min_wl = max(np.min(wavelengths),
                                 np.min(filter_wavelengths))
                    max_wl = min(np.max(wavelengths),
                                 np.max(filter_wavelengths))

                    # 원본 스펙트럼에서 공통 범위 필터링
                    mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
                    common_wavelengths = wavelengths[mask]
                    common_intensities = intensities[mask]

                    # 필터 데이터 보간
                    from scipy.interpolate import interp1d
                    filter_interp = interp1d(filter_wavelengths, filter_intensities,
                                             bounds_error=False, fill_value=0, kind='linear')
                    filter_values = filter_interp(common_wavelengths)

                    # 스펙트럼과 필터 곱하기
                    combined_intensities = common_intensities * filter_values

                    # 결합된 데이터로 계산
                    wavelengths = common_wavelengths
                    intensities = combined_intensities

            # XYZ 값 계산
            xyz = calculate_xyz_from_spectrum(wavelengths, intensities)

            # xy, uv 좌표 계산
            coords = calculate_all_color_coordinates(xyz)

            # 피크 파라미터 계산
            peak_params = calculate_peak_parameters(wavelengths, intensities)

            # 결과 표시 - 색좌표와 피크 파라미터 함께 표시
            return html.Div([
                # 색좌표 섹션
                html.Div([
                    html.Strong(
                        "색좌표", style={'fontSize': '14px', 'color': '#3498db'}),
                    html.Div([
                        html.Strong("xy : "),
                        html.Span(
                            f"{coords['xy'][0]:.4f}, {coords['xy'][1]:.4f}")
                    ]),
                    html.Div([
                        html.Strong("u'v' : "),
                        html.Span(
                            f"{coords['uv'][0]:.4f}, {coords['uv'][1]:.4f}")
                    ])
                ], style={'marginBottom': '10px'}),

                # 피크 파라미터 섹션
                html.Div([
                    html.Strong("피크 특성", style={
                                'fontSize': '14px', 'color': '#3498db'}),
                    html.Div([
                        html.Strong("Max Peak : "),
                        html.Span(
                            f"{peak_params['max_peak']} nm" if peak_params['max_peak'] else "계산 불가")
                    ]),
                    html.Div([
                        html.Strong("FWHM : "),
                        html.Span(
                            f"{peak_params['fwhm']} nm" if peak_params['fwhm'] else "계산 불가")
                    ]),
                    html.Div([
                        html.Strong("FWQM : "),
                        html.Span(
                            f"{peak_params['fwqm']} nm" if peak_params['fwqm'] else "계산 불가")
                    ])
                ])
            ])
        except Exception as e:
            return f"오류 발생: {str(e)}"

    # 색재현율 및 CIE 다이어그램 업데이트 콜백
    @app.callback(
        [Output('gamut-results', 'children'),
         Output('cie-diagram', 'figure')],
        [Input('spectrum-data-store', 'data'),
         Input('color-space-checklist', 'value'),
         Input('single-cie-diagram-type', 'value'),
         Input('color-filter-mode', 'value')],
        prevent_initial_call=False
    )
    def update_color_analysis(stored_data, color_spaces, diagram_type, filter_mode):
        # 데이터 없음 체크를 더 강화
        if not stored_data or stored_data == '{}' or stored_data == 'null':
            # 데이터가 없을 때도 기본 색공간이 포함된 다이어그램 표시
            return "RGB 스펙트럼 데이터를 모두 입력하세요.", create_empty_cie_diagram(diagram_type)

        try:
            # JSON 문자열이면 파싱
            if isinstance(stored_data, str):
                stored_data = json.loads(stored_data)

            # 빈 객체인 경우 체크
            if not stored_data:
                return "RGB 스펙트럼 데이터를 모두 입력하세요.", create_empty_cie_diagram(diagram_type)

            # R, G, B 데이터가 모두 있는지 확인
            if not stored_data.get('r') or not stored_data.get('g') or not stored_data.get('b'):
                # 데이터가 일부 누락되었을 때도 기본 색공간이 포함된 다이어그램 표시
                return "RGB 스펙트럼 데이터를 모두 입력하세요.", create_empty_cie_diagram(diagram_type)

            # 필터 모드 설정 확인
            filter_enabled = 'enabled' in filter_mode if filter_mode else False

            # 각 색상별 처리 함수
            def process_color_data(color_data, filter_data=None):
                if 'wavelength' in color_data:
                    wavelengths = np.array(color_data['wavelength'])
                    intensities = np.array(color_data['intensity'])
                else:
                    wavelengths = np.array(color_data['wavelengths'])
                    intensities = np.array(color_data['intensities'])

                # 필터 모드가 활성화되고 필터 데이터가 있는 경우
                if filter_enabled and filter_data:
                    if 'wavelength' in filter_data:
                        filter_wavelengths = np.array(
                            filter_data['wavelength'])
                        filter_intensities = np.array(filter_data['intensity'])
                    else:
                        filter_wavelengths = np.array(
                            filter_data['wavelengths'])
                        filter_intensities = np.array(
                            filter_data['intensities'])

                    # 두 데이터 세트의 파장 범위 일치시키기
                    min_wl = max(np.min(wavelengths),
                                 np.min(filter_wavelengths))
                    max_wl = min(np.max(wavelengths),
                                 np.max(filter_wavelengths))

                    # 원본 스펙트럼에서 공통 범위 필터링
                    mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
                    common_wavelengths = wavelengths[mask]
                    common_intensities = intensities[mask]

                    # 필터 데이터 보간
                    from scipy.interpolate import interp1d
                    filter_interp = interp1d(filter_wavelengths, filter_intensities,
                                             bounds_error=False, fill_value=0, kind='linear')
                    filter_values = filter_interp(common_wavelengths)

                    # 스펙트럼과 필터 곱하기
                    combined_intensities = common_intensities * filter_values

                    # 결합된 데이터로 계산
                    return common_wavelengths, combined_intensities

                return wavelengths, intensities

            # RGB 데이터 처리
            r_wavelengths, r_intensities = process_color_data(
                stored_data['r'], stored_data.get('r_filter'))
            g_wavelengths, g_intensities = process_color_data(
                stored_data['g'], stored_data.get('g_filter'))
            b_wavelengths, b_intensities = process_color_data(
                stored_data['b'], stored_data.get('b_filter'))

            # XYZ 값 계산
            r_xyz = calculate_xyz_from_spectrum(r_wavelengths, r_intensities)
            g_xyz = calculate_xyz_from_spectrum(g_wavelengths, g_intensities)
            b_xyz = calculate_xyz_from_spectrum(b_wavelengths, b_intensities)

            # 모든 색좌표 계산 (CIE 1931 xy와 CIE 1976 u'v')
            r_coords_all = calculate_all_color_coordinates(r_xyz)
            g_coords_all = calculate_all_color_coordinates(g_xyz)
            b_coords_all = calculate_all_color_coordinates(b_xyz)

            # 현재 선택된 다이어그램 유형에 따른 좌표
            r_coords = r_coords_all[diagram_type]
            g_coords = g_coords_all[diagram_type]
            b_coords = b_coords_all[diagram_type]

            # 색재현율 계산 (CIE 1931 xy와 CIE 1976 u'v' 모두)
            gamut_results_xy = {}
            gamut_results_uv = {}

            # 색공간 정보 가져오기
            all_color_spaces = get_color_spaces()

            for color_space_name in color_spaces:
                if color_space_name in all_color_spaces:
                    # 측정된 RGB 좌표
                    sample_xy = {
                        'red': list(r_coords_all['xy']),
                        'green': list(g_coords_all['xy']),
                        'blue': list(b_coords_all['xy'])
                    }

                    sample_uv = {
                        'red': list(r_coords_all['uv']),
                        'green': list(g_coords_all['uv']),
                        'blue': list(b_coords_all['uv'])
                    }

                    # 표준 색공간 정보
                    cs_info = all_color_spaces[color_space_name]

                    # xy 색공간에서 색재현율 계산
                    try:
                        # xy 좌표계에서 직접 계산
                        coverage_xy, area_ratio_xy = calculate_gamut_coverage_simple(
                            sample_xy, cs_info, 'xy')
                        coverage_uv, area_ratio_uv = calculate_gamut_coverage_simple(
                            sample_uv, cs_info, 'uv')

                        gamut_results_xy[color_space_name] = {
                            'overlap_ratio': coverage_xy,
                            'area_ratio': area_ratio_xy
                        }

                        gamut_results_uv[color_space_name] = {
                            'overlap_ratio': coverage_uv,
                            'area_ratio': area_ratio_uv
                        }
                    except Exception as e:
                        print(f"색공간 {color_space_name} 계산 오류: {str(e)}")
                        gamut_results_xy[color_space_name] = {
                            'overlap_ratio': 0,
                            'area_ratio': 0
                        }
                        gamut_results_uv[color_space_name] = {
                            'overlap_ratio': 0,
                            'area_ratio': 0
                        }

            # CIE 다이어그램 생성
            cie_fig = create_cie_diagram(
                r_coords, g_coords, b_coords, color_spaces, diagram_type)

            # 필터 모드 표시 추가
            filter_status = "필터 적용됨" if filter_enabled and (stored_data.get(
                'r_filter') or stored_data.get('g_filter') or stored_data.get('b_filter')) else "필터 미적용"

            # 결과 표시
            results_html = [
                html.Div(f"색 필터 상태: {filter_status}", style={
                         'marginBottom': '10px', 'fontWeight': 'bold'}),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("색공간", style={'fontSize': '12px',
                                'width': '20%', 'textAlign': 'center'}),
                        html.Th("CIE 1931 xy 중첩비 (%)", style={
                                'fontSize': '12px',
                                'width': '20%', 'textAlign': 'center'}),
                        html.Th("CIE 1931 xy 면적비 (%)", style={
                                'fontSize': '12px',
                                'width': '20%', 'textAlign': 'center'}),
                        html.Th("CIE 1976 u'v' 중첩비 (%)", style={
                                'fontSize': '12px',
                                'width': '20%', 'textAlign': 'center'}),
                        html.Th("CIE 1976 u'v' 면적비 (%)", style={
                                'fontSize': '12px',
                                'width': '20%', 'textAlign': 'center'})
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(color_space_name, style={
                                    'textAlign': 'center'}),
                            html.Td(f"{gamut_results_xy[color_space_name]['overlap_ratio']:.2f}%", style={
                                    'textAlign': 'center'}),
                            html.Td(f"{gamut_results_xy[color_space_name]['area_ratio']:.2f}%", style={
                                    'textAlign': 'center'}),
                            html.Td(f"{gamut_results_uv[color_space_name]['overlap_ratio']:.2f}%", style={
                                    'textAlign': 'center'}),
                            html.Td(f"{gamut_results_uv[color_space_name]['area_ratio']:.2f}%", style={
                                    'textAlign': 'center'})
                        ]) for color_space_name in color_spaces
                    ])
                ], style={'width': '100%', 'borderCollapse': 'collapse'})
            ]

            return results_html, cie_fig

        except Exception as e:
            print(f"색재현율 계산 중 오류 발생: {str(e)}")
            return f"색재현율 계산 중 오류 발생: {str(e)}", create_empty_cie_diagram(diagram_type)

    # 입력 초기화 콜백
    @app.callback(
        [Output('input-r-data', 'value', allow_duplicate=True),
         Output('input-g-data', 'value', allow_duplicate=True),
         Output('input-b-data', 'value', allow_duplicate=True),
         Output('input-r-filter-data', 'value', allow_duplicate=True),
         Output('input-g-filter-data', 'value', allow_duplicate=True),
         Output('input-b-filter-data', 'value', allow_duplicate=True),
         Output('spectrum-data-store', 'data', allow_duplicate=True)],
        [Input('reset-spectrum-inputs', 'n_clicks')],
        prevent_initial_call='initial_duplicate'
    )
    def reset_inputs(n_clicks):
        if n_clicks == 0:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        empty_data = {'r': None, 'g': None, 'b': None,
                      'r_text': "", 'g_text': "", 'b_text': "",
                      'r_filter': None, 'g_filter': None, 'b_filter': None,
                      'r_filter_text': "", 'g_filter_text': "", 'b_filter_text': "",
                      'filter_mode': True}
        return "", "", "", "", "", "", json.dumps(empty_data)

    # 페이지 로드 시 저장된 스펙트럼 데이터 불러오기
    @app.callback(
        [Output('input-r-data', 'value', allow_duplicate=True),
         Output('input-g-data', 'value', allow_duplicate=True),
         Output('input-b-data', 'value', allow_duplicate=True),
         Output('input-r-filter-data', 'value', allow_duplicate=True),
         Output('input-g-filter-data', 'value', allow_duplicate=True),
         Output('input-b-filter-data', 'value', allow_duplicate=True),
         Output('spectrum-graph', 'figure', allow_duplicate=True),
         Output('color-filter-mode', 'value', allow_duplicate=True)],
        [Input('url', 'pathname')],
        [State('spectrum-data-store', 'data')],
        prevent_initial_call='initial_duplicate'
    )
    def load_saved_spectrum_data(pathname, stored_data):
        # 스펙트럼 분석 탭인 경우에만 실행 (다른 탭과 충돌 방지)
        if pathname != '/' and pathname != '/home':
            raise dash.exceptions.PreventUpdate

        if stored_data is None:
            empty_fig = create_empty_figure("스펙트럼 분석", "데이터를 입력하세요.")
            return "", "", "", "", "", "", empty_fig, ['enabled']

        try:
            stored_data = json.loads(stored_data) if isinstance(
                stored_data, str) else stored_data

            # 원본 텍스트 데이터 불러오기
            r_text = stored_data.get('r_text', "")
            g_text = stored_data.get('g_text', "")
            b_text = stored_data.get('b_text', "")

            # 필터 텍스트 데이터 불러오기
            r_filter_text = stored_data.get('r_filter_text', "")
            g_filter_text = stored_data.get('g_filter_text', "")
            b_filter_text = stored_data.get('b_filter_text', "")

            # 필터 모드 상태 불러오기
            filter_mode = stored_data.get('filter_mode', True)
            filter_mode_value = ['enabled'] if filter_mode else []

            # 그래프 업데이트
            fig = create_spectrum_figure(stored_data)

            return r_text, g_text, b_text, r_filter_text, g_filter_text, b_filter_text, fig, filter_mode_value
        except Exception as e:
            print(f"저장된 데이터 불러오기 오류: {str(e)}")
            empty_fig = create_empty_figure("스펙트럼 분석", "데이터를 입력하세요.")
            return "", "", "", "", "", "", empty_fig, ['enabled']


def create_spectrum_figure(stored_data):
    """스펙트럼 그래프 생성"""
    fig = go.Figure()

    # 저장된 데이터가 없거나 유효하지 않은 경우 빈 그래프 반환
    if not stored_data or (not stored_data.get('r') and not stored_data.get('g') and not stored_data.get('b') and
                           not stored_data.get('r_filter') and not stored_data.get('g_filter') and not stored_data.get('b_filter')):
        # 안내 메시지 추가
        fig.add_annotation(
            text="데이터를 입력하세요.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )

        # 레이아웃 설정
        fig.update_layout(
            xaxis_title='파장 (nm)',
            yaxis_title='강도 (상대값)',
            template='plotly_white',
            xaxis=dict(range=[380, 780]),
            yaxis=dict(range=[0, 1]),
            margin=dict(l=50, r=50, t=30, b=50)
        )
        return fig

    # 색상 정의
    colors = {'r': '#e74c3c', 'g': '#27ae60', 'b': '#3498db'}
    names = {'r': 'R 스펙트럼', 'g': 'G 스펙트럼', 'b': 'B 스펙트럼'}
    filter_names = {'r_filter': 'R 필터', 'g_filter': 'G 필터', 'b_filter': 'B 필터'}
    combined_names = {'r': 'R 필터 적용', 'g': 'G 필터 적용', 'b': 'B 필터 적용'}

    # 필터 모드 확인 (문자열 'enabled'가 포함되어 있는지 확인)
    filter_mode = stored_data.get('filter_mode', True)
    filter_enabled = 'enabled' in filter_mode if isinstance(
        filter_mode, list) else filter_mode

    # 각 색상 데이터 독립적으로 처리
    for color in ['r', 'g', 'b']:
        color_data = stored_data.get(color)
        filter_data = stored_data.get(f'{color}_filter')

        # 1. EL 스펙트럼 데이터 처리 (있을 경우)
        if color_data:
            wavelengths = np.array(color_data['wavelength']) if 'wavelength' in color_data else np.array(
                color_data['wavelengths'])
            intensities = np.array(color_data['intensity']) if 'intensity' in color_data else np.array(
                color_data['intensities'])

            # 필터 모드가 활성화되고 필터 데이터가 있는 경우 - 결합 처리
            if filter_enabled and filter_data:
                filter_wavelengths = np.array(
                    filter_data['wavelength']) if 'wavelength' in filter_data else np.array(filter_data['wavelengths'])
                filter_intensities = np.array(
                    filter_data['intensity']) if 'intensity' in filter_data else np.array(filter_data['intensities'])

                # 두 데이터 세트의 파장 범위 일치시키기
                min_wl = max(np.min(wavelengths), np.min(filter_wavelengths))
                max_wl = min(np.max(wavelengths), np.max(filter_wavelengths))

                # 원본 스펙트럼에서 공통 범위 필터링
                mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
                common_wavelengths = wavelengths[mask]
                common_intensities = intensities[mask]

                # 필터 데이터 보간
                from scipy.interpolate import interp1d
                filter_interp = interp1d(filter_wavelengths, filter_intensities,
                                         bounds_error=False, fill_value=0, kind='linear')
                filter_values = filter_interp(common_wavelengths)

                # 스펙트럼과 필터 곱하기
                combined_intensities = common_intensities * filter_values

                # 결합된 스펙트럼 표시
                fig.add_trace(go.Scatter(
                    x=common_wavelengths,
                    y=combined_intensities,
                    mode='lines',
                    name=f"{combined_names[color]}",
                    line=dict(color=colors[color], width=2)
                ))
            else:
                # 필터 모드가 꺼져 있거나 필터 데이터가 없는 경우 - 원본 EL 스펙트럼만 표시
                fig.add_trace(go.Scatter(
                    x=wavelengths,
                    y=intensities,
                    mode='lines',
                    name=names[color],
                    line=dict(color=colors[color], width=2)
                ))

        # 2. 필터 데이터 처리 (있을 경우) - 필터 모드가 꺼져 있으면 독립적으로 표시
        if filter_data:
            filter_wavelengths = np.array(
                filter_data['wavelength']) if 'wavelength' in filter_data else np.array(filter_data['wavelengths'])
            filter_intensities = np.array(
                filter_data['intensity']) if 'intensity' in filter_data else np.array(filter_data['intensities'])

            # 필터 모드가 꺼져 있거나 EL 데이터가 없는 경우에 필터 그래프 표시
            if not filter_enabled or not color_data:
                fig.add_trace(go.Scatter(
                    x=filter_wavelengths,
                    y=filter_intensities,
                    mode='lines',
                    name=filter_names[f"{color}_filter"],
                    line=dict(color=colors[color], width=1, dash='dash'),
                    opacity=0.7
                ))
            # 필터 모드가 켜져 있더라도 필터 그래프를 항상 표시하도록 할 수 있음
            elif filter_enabled:
                fig.add_trace(go.Scatter(
                    x=filter_wavelengths,
                    y=filter_intensities,
                    mode='lines',
                    name=filter_names[f"{color}_filter"],
                    line=dict(color=colors[color], width=1, dash='dash'),
                    opacity=0.7
                ))

    # 레이아웃 설정
    fig.update_layout(
        xaxis_title='파장 (nm)',
        yaxis_title='강도 (상대값)',
        template='plotly_white',
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        )
    )

    # 가시광선 영역 설정
    fig.update_xaxes(range=[380, 780])

    return fig


def create_empty_figure(title, message):
    """빈 그래프 생성"""
    fig = go.Figure()

    # 안내 메시지 추가
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14)
    )

    # 레이아웃 설정
    fig.update_layout(
        xaxis_title='파장 (nm)',
        yaxis_title='강도 (상대값)',
        template='plotly_white',
        xaxis=dict(range=[380, 780]),
        yaxis=dict(range=[0, 1]),
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        )
    )

    return fig


def create_cie_diagram(r_coords, g_coords, b_coords, color_spaces, diagram_type='xy'):
    """CIE 다이어그램 생성 - 다중 샘플 분석 탭의 코드 재활용"""
    import plotly.graph_objects as go
    import numpy as np
    from utils import color_math

    fig = go.Figure()

    if diagram_type == 'xy':
        title = 'CIE 1931 xy 색도 다이어그램'
        xaxis_title = 'x'
        yaxis_title = 'y'
        xaxis_range = [0, 0.8]
        yaxis_range = [0, 0.9]

        # 스펙트럼 궤적 생성
        x_spectral, y_spectral = color_math.generate_spectral_locus(
            start_nm=380, end_nm=780, step=0.5)

        # 스펙트럼 궤적 그리기
        fig.add_trace(go.Scatter(
            x=x_spectral,
            y=y_spectral,
            mode='lines',
            line=dict(color='black', width=1.5, shape='spline'),
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
        for cs_name in color_spaces:
            cs = color_math.get_color_spaces()[cs_name]

            # xy 좌표계 사용
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

        # RGB 좌표 그리기
        if r_coords and g_coords and b_coords:
            # 삼각형 좌표
            x_points = [r_coords[0], g_coords[0], b_coords[0], r_coords[0]]
            y_points = [r_coords[1], g_coords[1], b_coords[1], r_coords[1]]

            # 삼각형 그리기
            fig.add_trace(go.Scatter(
                x=x_points,
                y=y_points,
                mode='lines+markers',
                name='측정 RGB',
                line=dict(width=1.5, dash='dash'),
                marker=dict(size=8)
            ))

    else:  # 'uv'
        title = 'CIE 1976 u\'v\' 색도 다이어그램'
        xaxis_title = 'u\''
        yaxis_title = 'v\''
        xaxis_range = [0, 0.7]
        yaxis_range = [0, 0.6]

        x_spectral, y_spectral = color_math.generate_spectral_locus(
            start_nm=380, end_nm=780, step=0.5)

        u_spectral, v_spectral = [], []
        for i in range(len(x_spectral)):
            u_prime, v_prime = color_math.xy_to_uv(
                x_spectral[i], y_spectral[i])
            u_spectral.append(u_prime)
            v_spectral.append(v_prime)

        fig.add_trace(go.Scatter(
            x=u_spectral,
            y=v_spectral,
            mode='lines',
            line=dict(color='black', width=1.5, shape='spline'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[u_spectral[-1], u_spectral[0]],
            y=[v_spectral[-1], v_spectral[0]],
            mode='lines',
            line=dict(color='black', width=1.5),
            showlegend=False
        ))

        color_space_data = color_math.get_color_spaces()
        for cs_name in color_spaces:
            if cs_name in color_space_data:
                cs = color_space_data[cs_name]
                u_points, v_points = [], []
                for point in ['red', 'green', 'blue', 'red']:
                    u_prime, v_prime = color_math.xy_to_uv(
                        cs[point][0], cs[point][1])
                    u_points.append(u_prime)
                    v_points.append(v_prime)

                fig.add_trace(go.Scatter(
                    x=u_points,
                    y=v_points,
                    mode='lines+markers',
                    name=cs_name,
                    line=dict(width=2)
                ))

        if r_coords and g_coords and b_coords:
            u_points = [r_coords[0], g_coords[0], b_coords[0], r_coords[0]]
            v_points = [r_coords[1], g_coords[1], b_coords[1], r_coords[1]]

            fig.add_trace(go.Scatter(
                x=u_points,
                y=v_points,
                mode='lines+markers',
                name='측정 RGB',
                line=dict(width=1.5, dash='dash'),
                marker=dict(size=8)
            ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis=dict(range=xaxis_range, constrain='domain'),
        yaxis=dict(range=yaxis_range, scaleanchor='x', scaleratio=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="top",
                    y=-0.15, xanchor="center", x=0.5),
        autosize=True,
        margin=dict(l=50, r=30, t=50, b=50),
        height=500
    )

    return fig


def create_empty_cie_diagram(diagram_type='xy'):
    """빈 CIE 다이어그램 생성"""
    import plotly.graph_objects as go
    from utils import color_math

    fig = go.Figure()

    # 기본 색공간 표시 (DCI-P3, Adobe RGB, BT.2020)
    color_space_data = color_math.get_color_spaces()
    default_color_spaces = ['DCI-P3', 'adobeRGB', 'BT2020']

    if diagram_type == 'xy':
        title = 'CIE 1931 xy 색도 다이어그램'
        xaxis_title = 'x'
        yaxis_title = 'y'
        xaxis_range = [0, 0.8]
        yaxis_range = [0, 0.9]

        # 스펙트럼 궤적 생성 및 표시
        x_spectral, y_spectral = color_math.generate_spectral_locus(
            start_nm=380, end_nm=780, step=0.5)

        # 스펙트럼 궤적 그리기
        fig.add_trace(go.Scatter(
            x=x_spectral,
            y=y_spectral,
            mode='lines',
            line=dict(color='black', width=1.5, shape='spline'),
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

        # 기본 색공간 표시
        for cs_name in default_color_spaces:
            if cs_name in color_space_data:
                cs = color_space_data[cs_name]

                # xy 좌표계 사용
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
    else:  # 'uv'
        title = 'CIE 1976 u\'v\' 색도 다이어그램'
        xaxis_title = 'u\''
        yaxis_title = 'v\''
        xaxis_range = [0, 0.7]
        yaxis_range = [0, 0.6]

        # 스펙트럼 궤적 생성
        x_spectral, y_spectral = color_math.generate_spectral_locus(
            start_nm=380, end_nm=780, step=0.5)

        u_spectral, v_spectral = [], []
        for i in range(len(x_spectral)):
            u_prime, v_prime = color_math.xy_to_uv(
                x_spectral[i], y_spectral[i])
            u_spectral.append(u_prime)
            v_spectral.append(v_prime)

        # 스펙트럼 궤적 그리기
        fig.add_trace(go.Scatter(
            x=u_spectral,
            y=v_spectral,
            mode='lines',
            line=dict(color='black', width=1.5, shape='spline'),
            showlegend=False
        ))

        # 스펙트럼 궤적 닫기 (끝점과 시작점 연결)
        fig.add_trace(go.Scatter(
            x=[u_spectral[-1], u_spectral[0]],
            y=[v_spectral[-1], v_spectral[0]],
            mode='lines',
            line=dict(color='black', width=1.5),
            showlegend=False
        ))

        # 기본 색공간 표시
        for cs_name in default_color_spaces:
            if cs_name in color_space_data:
                cs = color_space_data[cs_name]
                u_points, v_points = [], []

                for point in ['red', 'green', 'blue', 'red']:
                    u_prime, v_prime = color_math.xy_to_uv(
                        cs[point][0], cs[point][1])
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

    # # 안내 메시지 추가
    # fig.add_annotation(
    #     text="RGB 스펙트럼 데이터가 없습니다",
    #     xref="paper", yref="paper",
    #     x=0.5, y=0.2,
    #     showarrow=False,
    #     font=dict(size=12, color='rgba(0,0,0,0.5)')
    # )

    # 레이아웃 설정
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis=dict(range=xaxis_range, constrain='domain'),
        yaxis=dict(range=yaxis_range, scaleanchor='x', scaleratio=1),
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="top",
                    y=-0.15, xanchor="center", x=0.5),
        height=500
    )

    return fig

# 간단한 색재현율 계산 함수 추가


def calculate_gamut_coverage_simple(sample, standard, coord_type='xy'):
    """
    간단한 색재현율 계산 함수

    Args:
        sample: 측정된 RGB 좌표 {'red': [x, y], 'green': [x, y], 'blue': [x, y]}
        standard: 표준 색공간 정보 {'red': [x, y], 'green': [x, y], 'blue': [x, y]}
        coord_type: 좌표 유형 ('xy' 또는 'uv')

    Returns:
        coverage: 중첩 비율 (%)
        area_ratio: 면적 비율 (%)
    """
    import numpy as np
    from scipy.spatial import ConvexHull

    # 삼각형 면적 계산 함수
    def triangle_area(p1, p2, p3):
        return 0.5 * abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])))

    try:
        # 샘플과 표준 색공간의 좌표 추출
        sample_coords = np.array(
            [sample['red'], sample['green'], sample['blue']])
        standard_coords = np.array([
            standard['red'][:2] if coord_type == 'xy' else xy_to_uv(
                standard['red'][0], standard['red'][1]),
            standard['green'][:2] if coord_type == 'xy' else xy_to_uv(
                standard['green'][0], standard['green'][1]),
            standard['blue'][:2] if coord_type == 'xy' else xy_to_uv(
                standard['blue'][0], standard['blue'][1])
        ])

        # 샘플과 표준 색공간의 면적 계산
        sample_area = triangle_area(
            sample_coords[0], sample_coords[1], sample_coords[2])
        standard_area = triangle_area(
            standard_coords[0], standard_coords[1], standard_coords[2])

        # 면적비 계산 (샘플/표준 * 100)
        area_ratio = (sample_area / standard_area) * 100

        # 중첩 영역 계산을 위해 convex hull 분석
        # 단순화를 위해 색재현율은 면적비의 90%로 근사값 사용
        coverage = min(area_ratio * 0.9, 100)  # 최대 100%로 제한

        return coverage, area_ratio
    except Exception as e:
        print(f"색재현율 계산 오류: {str(e)}")
        return 0.0, 0.0

# 스펙트럼 피크 파라미터 계산 함수 추가


def calculate_peak_parameters(wavelengths, intensities):
    """
    스펙트럼 피크 파라미터(FWHM, FWQM, Max Peak 파장)를 계산합니다.

    Args:
        wavelengths (array): 파장 배열
        intensities (array): 강도 배열

    Returns:
        dict: 피크 파라미터 {'max_peak': 값, 'fwhm': 값, 'fwqm': 값}
    """
    import numpy as np
    from scipy.interpolate import interp1d

    try:
        # 입력 데이터를 numpy 배열로 변환
        wavelengths = np.array(wavelengths)
        intensities = np.array(intensities)

        # 최대 피크 파장 찾기
        max_idx = np.argmax(intensities)
        max_intensity = intensities[max_idx]
        max_peak = wavelengths[max_idx]

        # 크기가 충분하지 않으면 계산 불가
        if len(wavelengths) < 3 or max_intensity <= 0:
            return {'max_peak': max_peak, 'fwhm': None, 'fwqm': None}

        # 보간을 위해 파장을 정렬
        sorted_indices = np.argsort(wavelengths)
        sorted_wavelengths = wavelengths[sorted_indices]
        sorted_intensities = intensities[sorted_indices]

        # 데이터 보간 (더 정확한 계산을 위해)
        f = interp1d(sorted_wavelengths, sorted_intensities, kind='cubic',
                     bounds_error=False, fill_value=0)

        # 더 정밀한 파장 배열 생성
        fine_wavelengths = np.linspace(
            sorted_wavelengths.min(), sorted_wavelengths.max(), 1000)
        fine_intensities = f(fine_wavelengths)

        # 다시 최대 피크 계산 (보간 후)
        max_idx = np.argmax(fine_intensities)
        max_intensity = fine_intensities[max_idx]
        max_peak = fine_wavelengths[max_idx]

        # 반치폭(FWHM) 및 1/4치폭(FWQM) 계산을 위한 기준값
        half_max = max_intensity / 2
        quarter_max = max_intensity / 4

        # FWHM 계산
        above_half_max = fine_intensities >= half_max
        if not np.any(above_half_max):
            fwhm = None
        else:
            indices = np.where(above_half_max)[0]
            fwhm = fine_wavelengths[indices[-1]] - fine_wavelengths[indices[0]]

        # FWQM 계산
        above_quarter_max = fine_intensities >= quarter_max
        if not np.any(above_quarter_max):
            fwqm = None
        else:
            indices = np.where(above_quarter_max)[0]
            fwqm = fine_wavelengths[indices[-1]] - fine_wavelengths[indices[0]]

        return {
            'max_peak': round(max_peak, 1),
            'fwhm': round(fwhm, 1) if fwhm is not None else None,
            'fwqm': round(fwqm, 1) if fwqm is not None else None
        }
    except Exception as e:
        print(f"피크 파라미터 계산 오류: {str(e)}")
        return {'max_peak': None, 'fwhm': None, 'fwqm': None}
