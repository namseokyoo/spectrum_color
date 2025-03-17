from dash import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from utils import parsers
from scipy import interpolate
import dash


def register_callbacks(app):
    # R PL 그래프 업데이트 콜백
    @app.callback(
        Output('r-pl-graph', 'figure'),
        [Input('r-pl-data', 'value')]
    )
    def update_r_pl_graph(data):
        return update_spectrum_graph(data, "PL", "#e74c3c")

    # R EL 그래프 업데이트 콜백
    @app.callback(
        Output('r-el-graph', 'figure'),
        [Input('r-el-data', 'value')]
    )
    def update_r_el_graph(data):
        return update_spectrum_graph(data, "EL", "#e74c3c")

    # G PL 그래프 업데이트 콜백
    @app.callback(
        Output('g-pl-graph', 'figure'),
        [Input('g-pl-data', 'value')]
    )
    def update_g_pl_graph(data):
        return update_spectrum_graph(data, "PL", "#2ecc71")

    # G EL 그래프 업데이트 콜백
    @app.callback(
        Output('g-el-graph', 'figure'),
        [Input('g-el-data', 'value')]
    )
    def update_g_el_graph(data):
        return update_spectrum_graph(data, "EL", "#2ecc71")

    # B PL 그래프 업데이트 콜백
    @app.callback(
        Output('b-pl-graph', 'figure'),
        [Input('b-pl-data', 'value')]
    )
    def update_b_pl_graph(data):
        return update_spectrum_graph(data, "PL", "#3498db")

    # B EL 그래프 업데이트 콜백
    @app.callback(
        Output('b-el-graph', 'figure'),
        [Input('b-el-data', 'value')]
    )
    def update_b_el_graph(data):
        return update_spectrum_graph(data, "EL", "#3498db")

    # R 광학 상수 계산 콜백
    @app.callback(
        [Output('r-oc-graph', 'figure'),
         Output('r-oc-data', 'value')],
        [Input('calculate-r-oc', 'n_clicks')],
        [State('r-pl-data', 'value'),
         State('r-el-data', 'value'),
         State('r-noise-reduction', 'value')]
    )
    def calculate_r_oc(n_clicks, pl_data, el_data, noise_reduction):
        if n_clicks == 0 or not pl_data or not el_data:
            return create_empty_figure("R 광학 상수 스펙트럼", "계산을 위해 PL과 EL 데이터를 입력하고 계산 버튼을 클릭하세요"), ""

        # 노이즈 제거 활성화 여부 확인
        use_noise_reduction = noise_reduction and 'enabled' in noise_reduction

        return calculate_oc(pl_data, el_data, "R 광학 상수 스펙트럼", "#e74c3c", use_noise_reduction)

    # G 광학 상수 계산 콜백
    @app.callback(
        [Output('g-oc-graph', 'figure'),
         Output('g-oc-data', 'value')],
        [Input('calculate-g-oc', 'n_clicks')],
        [State('g-pl-data', 'value'),
         State('g-el-data', 'value'),
         State('g-noise-reduction', 'value')]
    )
    def calculate_g_oc(n_clicks, pl_data, el_data, noise_reduction):
        if n_clicks == 0 or not pl_data or not el_data:
            return create_empty_figure("G 광학 상수 스펙트럼", "계산을 위해 PL과 EL 데이터를 입력하고 계산 버튼을 클릭하세요"), ""

        # 노이즈 제거 활성화 여부 확인
        use_noise_reduction = noise_reduction and 'enabled' in noise_reduction

        return calculate_oc(pl_data, el_data, "G 광학 상수 스펙트럼", "#2ecc71", use_noise_reduction)

    # B 광학 상수 계산 콜백
    @app.callback(
        [Output('b-oc-graph', 'figure'),
         Output('b-oc-data', 'value')],
        [Input('calculate-b-oc', 'n_clicks')],
        [State('b-pl-data', 'value'),
         State('b-el-data', 'value'),
         State('b-noise-reduction', 'value')]
    )
    def calculate_b_oc(n_clicks, pl_data, el_data, noise_reduction):
        if n_clicks == 0 or not pl_data or not el_data:
            return create_empty_figure("B 광학 상수 스펙트럼", "계산을 위해 PL과 EL 데이터를 입력하고 계산 버튼을 클릭하세요"), ""

        # 노이즈 제거 활성화 여부 확인
        use_noise_reduction = noise_reduction and 'enabled' in noise_reduction

        return calculate_oc(pl_data, el_data, "B 광학 상수 스펙트럼", "#3498db", use_noise_reduction)

    # 입력 데이터 저장 콜백 - R
    @app.callback(
        Output('optical-data-store', 'data'),
        [Input('r-pl-data', 'value'),
         Input('r-el-data', 'value'),
         Input('g-pl-data', 'value'),
         Input('g-el-data', 'value'),
         Input('b-pl-data', 'value'),
         Input('b-el-data', 'value')],
        [State('optical-data-store', 'data')]
    )
    def save_optical_data(r_pl, r_el, g_pl, g_el, b_pl, b_el, stored_data):
        # 콜백 컨텍스트에서 트리거된 입력 확인
        ctx = dash.callback_context
        if not ctx.triggered:
            return stored_data or {}

        # 저장할 데이터 초기화
        if stored_data is None:
            stored_data = {}

        # 트리거된 입력에 따라 데이터 업데이트
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if input_id == 'r-pl-data' and r_pl:
            stored_data['r_pl'] = r_pl
        elif input_id == 'r-el-data' and r_el:
            stored_data['r_el'] = r_el
        elif input_id == 'g-pl-data' and g_pl:
            stored_data['g_pl'] = g_pl
        elif input_id == 'g-el-data' and g_el:
            stored_data['g_el'] = g_el
        elif input_id == 'b-pl-data' and b_pl:
            stored_data['b_pl'] = b_pl
        elif input_id == 'b-el-data' and b_el:
            stored_data['b_el'] = b_el

        return stored_data

    # 저장된 데이터 로드 콜백 - R PL
    @app.callback(
        Output('r-pl-data', 'value'),
        [Input('optical-data-store', 'modified_timestamp')],
        [State('optical-data-store', 'data'),
         State('r-pl-data', 'value')]
    )
    def load_r_pl_data(ts, stored_data, current_value):
        if ts is None or stored_data is None or current_value:
            return current_value or ""
        return stored_data.get('r_pl', "")

    # 저장된 데이터 로드 콜백 - R EL
    @app.callback(
        Output('r-el-data', 'value'),
        [Input('optical-data-store', 'modified_timestamp')],
        [State('optical-data-store', 'data'),
         State('r-el-data', 'value')]
    )
    def load_r_el_data(ts, stored_data, current_value):
        if ts is None or stored_data is None or current_value:
            return current_value or ""
        return stored_data.get('r_el', "")

    # G PL 데이터 로드
    @app.callback(
        Output('g-pl-data', 'value'),
        [Input('optical-data-store', 'modified_timestamp')],
        [State('optical-data-store', 'data'),
         State('g-pl-data', 'value')]
    )
    def load_g_pl_data(ts, stored_data, current_value):
        if ts is None or stored_data is None or current_value:
            return current_value or ""
        return stored_data.get('g_pl', "")

    # G EL 데이터 로드
    @app.callback(
        Output('g-el-data', 'value'),
        [Input('optical-data-store', 'modified_timestamp')],
        [State('optical-data-store', 'data'),
         State('g-el-data', 'value')]
    )
    def load_g_el_data(ts, stored_data, current_value):
        if ts is None or stored_data is None or current_value:
            return current_value or ""
        return stored_data.get('g_el', "")

    # B PL 데이터 로드
    @app.callback(
        Output('b-pl-data', 'value'),
        [Input('optical-data-store', 'modified_timestamp')],
        [State('optical-data-store', 'data'),
         State('b-pl-data', 'value')]
    )
    def load_b_pl_data(ts, stored_data, current_value):
        if ts is None or stored_data is None or current_value:
            return current_value or ""
        return stored_data.get('b_pl', "")

    # B EL 데이터 로드
    @app.callback(
        Output('b-el-data', 'value'),
        [Input('optical-data-store', 'modified_timestamp')],
        [State('optical-data-store', 'data'),
         State('b-el-data', 'value')]
    )
    def load_b_el_data(ts, stored_data, current_value):
        if ts is None or stored_data is None or current_value:
            return current_value or ""
        return stored_data.get('b_el', "")

    # 새 PL 그래프 업데이트 콜백 - R
    @app.callback(
        Output('r-new-pl-graph', 'figure'),
        [Input('r-new-pl-data', 'value')]
    )
    def update_r_new_pl_graph(data):
        return update_spectrum_graph(data, "새 PL", "#e74c3c")

    # 새 PL 그래프 업데이트 콜백 - G
    @app.callback(
        Output('g-new-pl-graph', 'figure'),
        [Input('g-new-pl-data', 'value')]
    )
    def update_g_new_pl_graph(data):
        return update_spectrum_graph(data, "새 PL", "#2ecc71")

    # 새 PL 그래프 업데이트 콜백 - B
    @app.callback(
        Output('b-new-pl-graph', 'figure'),
        [Input('b-new-pl-data', 'value')]
    )
    def update_b_new_pl_graph(data):
        return update_spectrum_graph(data, "새 PL", "#3498db")

    # R 예측 EL 계산 콜백
    @app.callback(
        [Output('r-predicted-el-graph', 'figure'),
         Output('r-predicted-el-data', 'value')],
        [Input('calculate-r-predicted-el', 'n_clicks')],
        [State('r-new-pl-data', 'value'),
         State('r-oc-data', 'value')]
    )
    def calculate_r_predicted_el(n_clicks, new_pl_data, oc_data):
        if n_clicks == 0 or not new_pl_data or not oc_data:
            return create_empty_figure("", "계산을 위해 새 PL 데이터를 입력하고 OC를 먼저 계산한 후 EL 예측 버튼을 클릭하세요"), ""

        return predict_el_from_oc_and_pl(new_pl_data, oc_data, "R 예측 EL 스펙트럼", "#e74c3c")

    # G 예측 EL 계산 콜백
    @app.callback(
        [Output('g-predicted-el-graph', 'figure'),
         Output('g-predicted-el-data', 'value')],
        [Input('calculate-g-predicted-el', 'n_clicks')],
        [State('g-new-pl-data', 'value'),
         State('g-oc-data', 'value')]
    )
    def calculate_g_predicted_el(n_clicks, new_pl_data, oc_data):
        if n_clicks == 0 or not new_pl_data or not oc_data:
            return create_empty_figure("", "계산을 위해 새 PL 데이터를 입력하고 OC를 먼저 계산한 후 EL 예측 버튼을 클릭하세요"), ""

        return predict_el_from_oc_and_pl(new_pl_data, oc_data, "G 예측 EL 스펙트럼", "#2ecc71")

    # B 예측 EL 계산 콜백
    @app.callback(
        [Output('b-predicted-el-graph', 'figure'),
         Output('b-predicted-el-data', 'value')],
        [Input('calculate-b-predicted-el', 'n_clicks')],
        [State('b-new-pl-data', 'value'),
         State('b-oc-data', 'value')]
    )
    def calculate_b_predicted_el(n_clicks, new_pl_data, oc_data):
        if n_clicks == 0 or not new_pl_data or not oc_data:
            return create_empty_figure("", "계산을 위해 새 PL 데이터를 입력하고 OC를 먼저 계산한 후 EL 예측 버튼을 클릭하세요"), ""

        return predict_el_from_oc_and_pl(new_pl_data, oc_data, "B 예측 EL 스펙트럼", "#3498db")


def update_spectrum_graph(data, title, color):
    """스펙트럼 데이터를 파싱하고 그래프 생성"""
    try:
        if not data:
            return create_empty_figure("", "데이터를 입력하세요")

        df = parse_text_data_flexible(data, multi_column=True)
        if df is None:
            return create_empty_figure("", "데이터 파싱 오류. 올바른 형식으로 입력했는지 확인하세요.")

        fig = go.Figure()

        # 파장 열 확인
        if 'wavelength' not in df.columns:
            return create_empty_figure("", "파장 데이터가 없습니다. 첫 번째 열이 파장 데이터인지 확인하세요.")

        # 다중 열 처리
        for i, col in enumerate(df.columns):
            if col != 'wavelength':
                # 색상 그라데이션 생성 (여러 조건을 구분하기 위해)
                if len(df.columns) > 2:  # 파장 + 2개 이상의 데이터 열
                    r, g, b = hex_to_rgb(color)
                    factor = 0.7 + (0.3 * i / (len(df.columns) - 2)
                                    ) if len(df.columns) > 2 else 1.0
                    trace_color = rgb_to_hex(
                        int(r * factor), int(g * factor), int(b * factor))
                else:
                    trace_color = color

                fig.add_trace(go.Scatter(
                    x=df['wavelength'],
                    y=df[col],
                    mode='lines',
                    name=col,
                    line=dict(color=trace_color, width=2)
                ))

        # 레이아웃 설정
        fig.update_layout(
            xaxis=dict(
                title=dict(
                    text='파장 (nm)',
                    font=dict(size=10)  # x축 제목 폰트 크기
                )
            ),
            yaxis=dict(
                title=dict(
                    text='강도 (상대값)',
                    font=dict(size=10)  # y축 제목 폰트 크기
                )
            ),
            template='plotly_white',
            margin=dict(l=20, r=20, t=10, b=10),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-1.15,
                xanchor="center",
                x=0.5,
                font=dict(size=10)  # 범례 폰트 크기 20% 감소 (기본 12.5에서 10으로)
            )
        )

        return fig

    except Exception as e:
        return create_empty_figure("", f"그래프 생성 중 오류 발생: {str(e)}")


def create_mini_empty_figure(title, message="데이터를 입력하세요"):
    """작은 빈 그래프 생성"""
    fig = go.Figure()

    # 안내 메시지 추가
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=5)
    )

    # 레이아웃 설정
    fig.update_layout(
        title=title,
        xaxis_title='파장 (nm)',
        yaxis_title='강도',
        template='plotly_white',
        xaxis=dict(range=[380, 780]),
        yaxis=dict(range=[0, 1]),
        margin=dict(l=30, r=10, t=30, b=30),
        height=150
    )

    return fig


def calculate_oc(pl_data, el_data, title, color, use_noise_reduction=True):
    """PL과 EL 데이터로부터 OC 계산, 다중 열 지원, 노이즈 제거 옵션 추가"""
    try:
        # 데이터 파싱
        df_pl = parse_text_data_flexible(pl_data)
        df_el = parse_text_data_flexible(el_data, multi_column=True)

        if df_pl is None or df_el is None:
            return create_empty_figure("", "데이터 파싱 오류. 올바른 형식으로 입력했는지 확인하세요."), ""

        # 파장 범위 결정 (두 데이터셋의 교집합)
        min_wl = max(df_pl['wavelength'].min(), df_el['wavelength'].min())
        max_wl = min(df_pl['wavelength'].max(), df_el['wavelength'].max())

        # 더 세밀한 파장 간격으로 보간 (1nm 단위)
        wavelengths = np.linspace(min_wl, max_wl, int(max_wl - min_wl) + 1)

        # PL 데이터 보간
        f_pl = interpolate.interp1d(
            df_pl['wavelength'], df_pl['intensity'], kind='cubic', bounds_error=False, fill_value=0)
        pl_interp = f_pl(wavelengths)

        # 0으로 나누기 방지 - 최소 임계값 설정 제거, 대신 0인 경우만 처리
        zero_indices = np.where(pl_interp == 0)[0]
        if len(zero_indices) > 0:
            # 0인 값만 매우 작은 값으로 대체 (원본 데이터 보존)
            pl_interp_safe = pl_interp.copy()
            pl_interp_safe[zero_indices] = np.finfo(float).eps  # 머신 엡실론 사용
        else:
            pl_interp_safe = pl_interp

        # 그래프 생성
        fig = go.Figure()

        # 데이터 텍스트 초기화 - 탭으로 구분 (Excel 친화적)
        data_text = "파장(nm)"

        # EL 열 이름 가져오기 (파장 열 제외)
        el_columns = [col for col in df_el.columns if col != 'wavelength']

        # 각 열의 정규화 최대값을 저장할 딕셔너리
        max_oc_values = {}

        # 각 EL 열에 대해 OC 계산 및 최대값 찾기
        for col_name in el_columns:
            # EL 데이터 보간
            f_el = interpolate.interp1d(
                df_el['wavelength'], df_el[col_name], kind='cubic', bounds_error=False, fill_value=0)
            el_interp = f_el(wavelengths)

            # OC = EL / PL 계산 (0으로 나누기 방지)
            oc_values = np.divide(el_interp, pl_interp_safe, out=np.zeros_like(
                el_interp), where=pl_interp_safe != 0)

            # 노이즈 제거 옵션이 활성화된 경우
            if use_noise_reduction:
                # EL 스펙트럼의 최대 피크 파장 찾기
                el_peak_idx = np.argmax(el_interp)
                el_peak_wavelength = wavelengths[el_peak_idx]

                # 피크 주변 ±50nm 범위 설정
                peak_range_min = el_peak_wavelength - 50
                peak_range_max = el_peak_wavelength + 50

                # 해당 범위 내의 인덱스 찾기
                peak_range_indices = np.where(
                    (wavelengths >= peak_range_min) &
                    (wavelengths <= peak_range_max)
                )[0]

                # 해당 범위 내에서 최대값 찾기
                if len(peak_range_indices) > 0:
                    peak_range_oc = oc_values[peak_range_indices]
                    max_oc = np.max(peak_range_oc)
                else:
                    max_oc = np.max(oc_values)  # 범위가 없으면 전체에서 최대값 찾기
            else:
                # 노이즈 제거 비활성화 시 전체 범위에서 최대값 찾기
                max_oc = np.max(oc_values)

            # 각 열의 최대값 저장
            max_oc_values[col_name] = max_oc

        # 노이즈 제거 텍스트 (그래프 제목에 표시)
        noise_reduction_text = " (노이즈 제거)" if use_noise_reduction else ""

        # 각 EL 열에 대해 정규화된 OC 계산 및 그래프 추가
        for i, col_name in enumerate(el_columns):
            # 색상 그라데이션 생성 (여러 조건을 구분하기 위해)
            if len(el_columns) > 1:
                # 색상 그라데이션 계산
                r, g, b = hex_to_rgb(color)
                factor = 0.7 + (0.3 * i / (len(el_columns) - 1)
                                ) if len(el_columns) > 1 else 1.0
                trace_color = rgb_to_hex(
                    int(r * factor), int(g * factor), int(b * factor))
            else:
                trace_color = color

            # EL 데이터 보간
            f_el = interpolate.interp1d(
                df_el['wavelength'], df_el[col_name], kind='cubic', bounds_error=False, fill_value=0)
            el_interp = f_el(wavelengths)

            # OC = EL / PL 계산 (0으로 나누기 방지)
            oc_values = np.divide(el_interp, pl_interp_safe, out=np.zeros_like(
                el_interp), where=pl_interp_safe != 0)

            # 해당 열의 최대값으로 정규화
            max_oc = max_oc_values[col_name]
            if max_oc > 0:
                oc_values_normalized = oc_values / max_oc

                # 노이즈 제거 활성화 시 1을 초과하는 값을 이동평균으로 대체
                if use_noise_reduction:
                    # 1을 초과하는 값 찾기
                    over_one_indices = np.where(oc_values_normalized > 1.0)[0]

                    if len(over_one_indices) > 0:
                        # 이동평균 윈도우 크기
                        window_size = 10

                        for idx in over_one_indices:
                            # 윈도우 범위 설정 (경계 고려)
                            start_idx = max(0, idx - window_size // 2)
                            end_idx = min(len(oc_values_normalized),
                                          idx + window_size // 2 + 1)

                            # 윈도우 내 1 이하인 값들만 선택
                            window_values = oc_values_normalized[start_idx:end_idx]
                            valid_values = window_values[window_values <= 1.0]

                            if len(valid_values) > 0:
                                # 유효한 값들의 평균으로 대체
                                oc_values_normalized[idx] = np.mean(
                                    valid_values)
                            else:
                                # 유효한 값이 없으면 1.0으로 설정
                                oc_values_normalized[idx] = 1.0
            else:
                oc_values_normalized = np.zeros_like(oc_values)

            # 그래프에 추가
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=oc_values_normalized,
                mode='lines',
                name=col_name,
                line=dict(color=trace_color, width=2)
            ))

            # 데이터 텍스트에 열 이름 추가 (첫 행)
            if i == 0:
                data_text = f"{col_name}"
            else:
                data_text = data_text + f"\t{col_name}"

        # 데이터 텍스트에 파장별 OC 값 추가
        for j, wl in enumerate(wavelengths):
            row_text = f"{wl:.2f}"

            for col_name in el_columns:
                # EL 데이터 보간
                f_el = interpolate.interp1d(
                    df_el['wavelength'], df_el[col_name], kind='cubic', bounds_error=False, fill_value=0)
                el_val = f_el(wl)

                # PL 데이터 보간
                pl_val = f_pl(wl)

                # OC 계산 (0으로 나누기 방지)
                oc_val = el_val / pl_val if pl_val != 0 else 0

                # 해당 열의 저장된 최대값으로 정규화
                max_oc = max_oc_values[col_name]
                if max_oc > 0:
                    oc_val_normalized = oc_val / max_oc

                    # 노이즈 제거 활성화 시 1을 초과하는 값 처리
                    if use_noise_reduction and oc_val_normalized > 1.0:
                        # 이 경우 이미 그래프에서 처리된 값을 사용
                        idx = np.argmin(np.abs(wavelengths - wl))
                        oc_val_normalized = min(1.0, oc_val_normalized)
                else:
                    oc_val_normalized = 0

                row_text += f"\t{oc_val_normalized:.6f}"

            data_text += f"\n{row_text}"

        # 레이아웃 설정
        fig.update_layout(
            xaxis_title=dict(
                text='파장 (nm)',
                font=dict(size=10)  # x축 제목 폰트 크기
            ),
            yaxis_title=dict(
                text='광학 상수 (정규화)',
                font=dict(size=10)  # y축 제목 폰트 크기
            ),
            template='plotly_white',
            margin=dict(l=50, r=50, t=30, b=50),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=10)  # 범례 폰트 크기 20% 감소 (기본 12.5에서 10으로)
            )
        )

        return fig, data_text

    except Exception as e:
        return create_empty_figure("", f"계산 중 오류 발생: {str(e)}"), ""


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
        xaxis_title=dict(
            text='파장 (nm)',
            font=dict(size=10)  # x축 제목 폰트 크기
        ),
        yaxis_title=dict(
            text='광학 상수 (상대값)',
            font=dict(size=10)  # y축 제목 폰트 크기
        ),
        template='plotly_white',
        xaxis=dict(range=[380, 780]),
        yaxis=dict(range=[0, 1]),
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=10)  # 범례 폰트 크기 20% 감소
        )
    )

    return fig


def parse_text_data_flexible(data, multi_column=False):
    """헤더가 있거나 없는 데이터를 모두 처리할 수 있는 파서, 다중 열 지원"""
    try:
        # 줄 단위로 분리
        lines = [line.strip()
                 for line in data.strip().split('\n') if line.strip()]

        if not lines:
            return None

        # 첫 번째 줄 분석
        first_line = lines[0].split()

        # 헤더 여부 확인
        has_header = not (len(first_line) >= 2 and is_numeric(first_line[0]))

        # 헤더 처리
        headers = []
        if has_header:
            headers = first_line
            lines = lines[1:]  # 헤더 제거

        # 데이터 파싱
        wavelengths = []
        intensities_dict = {}

        # 다중 열 처리를 위한 초기화
        if multi_column:
            # 헤더가 있는 경우 헤더 사용, 없는 경우 열 번호 사용
            if has_header and len(headers) > 1:
                column_names = headers[1:]  # 첫 번째 열은 파장
            else:
                # 첫 번째 데이터 행으로 열 수 파악
                column_count = len(lines[0].split()) - 1  # 파장 열 제외
                column_names = [f"Column_{i+1}" for i in range(column_count)]

            # 각 열에 대한 빈 리스트 초기화
            for col_name in column_names:
                intensities_dict[col_name] = []
        else:
            # 단일 열 처리
            intensities_dict["intensity"] = []

        # 데이터 행 처리
        for line in lines:
            values = line.split()
            if len(values) >= 2 and is_numeric(values[0]):
                wavelength = float(values[0])
                wavelengths.append(wavelength)

                if multi_column:
                    # 다중 열 처리
                    for i, col_name in enumerate(intensities_dict.keys()):
                        if i + 1 < len(values) and is_numeric(values[i + 1]):
                            intensities_dict[col_name].append(
                                float(values[i + 1]))
                        else:
                            intensities_dict[col_name].append(
                                0.0)  # 누락된 값은 0으로 처리
                else:
                    # 단일 열 처리
                    intensities_dict["intensity"].append(float(values[1]))

        if not wavelengths:
            return None

        # 데이터프레임 생성
        df = pd.DataFrame({'wavelength': wavelengths})
        for col_name, intensities in intensities_dict.items():
            if len(intensities) == len(wavelengths):  # 길이 확인
                df[col_name] = intensities

        return df

    except Exception as e:
        print(f"데이터 파싱 오류: {str(e)}")
        return None


def is_numeric(val):
    """문자열이 숫자로 변환 가능한지 확인"""
    try:
        float(val)
        return True
    except ValueError:
        return False


# 색상 변환 유틸리티 함수
def hex_to_rgb(hex_color):
    """HEX 색상 코드를 RGB로 변환"""
    hex_color = hex_color.lstrip('#')
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def rgb_to_hex(r, g, b):
    """RGB 값을 HEX 색상 코드로 변환"""
    return f'#{r:02x}{g:02x}{b:02x}'


def predict_el_from_oc_and_pl(pl_data, oc_data, title, color):
    """OC와 PL 데이터로부터 EL 예측"""
    try:
        # 데이터 파싱
        df_pl = parse_text_data_flexible(pl_data)
        df_oc = parse_text_data_flexible(oc_data, multi_column=True)

        if df_pl is None or df_oc is None:
            return create_empty_figure("", "데이터 파싱 오류. 올바른 형식으로 입력했는지 확인하세요."), ""

        # 파장 범위 결정 (두 데이터셋의 교집합)
        min_wl = max(df_pl['wavelength'].min(), df_oc['wavelength'].min())
        max_wl = min(df_pl['wavelength'].max(), df_oc['wavelength'].max())

        # 더 세밀한 파장 간격으로 보간 (1nm 단위)
        wavelengths = np.linspace(min_wl, max_wl, int(max_wl - min_wl) + 1)

        # PL 데이터 보간
        f_pl = interpolate.interp1d(
            df_pl['wavelength'], df_pl['intensity'], kind='cubic', bounds_error=False, fill_value=0)
        pl_interp = f_pl(wavelengths)

        # 그래프 생성
        fig = go.Figure()

        # 데이터 텍스트 초기화 - 탭으로 구분 (Excel 친화적)
        data_text = "파장(nm)"

        # OC 열 이름 가져오기 (파장 열 제외)
        oc_columns = [col for col in df_oc.columns if col != 'wavelength']

        # 각 OC 열에 대해 EL 계산
        for i, col_name in enumerate(oc_columns):
            # OC 데이터 보간
            f_oc = interpolate.interp1d(
                df_oc['wavelength'], df_oc[col_name], kind='cubic', bounds_error=False, fill_value=0)
            oc_interp = f_oc(wavelengths)

            # EL = OC * PL 계산
            el_values = oc_interp * pl_interp

            # 최대값으로 정규화
            max_el = np.max(el_values)
            if max_el > 0:
                el_values_normalized = el_values / max_el
            else:
                el_values_normalized = el_values

            # 색상 그라데이션 생성 (여러 조건을 구분하기 위해)
            if len(oc_columns) > 1:
                r, g, b = hex_to_rgb(color)
                factor = 0.7 + (0.3 * i / (len(oc_columns) - 1)
                                ) if len(oc_columns) > 1 else 1.0
                trace_color = rgb_to_hex(
                    int(r * factor), int(g * factor), int(b * factor))
            else:
                trace_color = color

            # 그래프에 추가
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=el_values_normalized,
                mode='lines',
                name=col_name,
                line=dict(color=trace_color, width=2)
            ))

            # 데이터 텍스트에 열 이름 추가 (첫 행)
            if i == 0:
                data_text += f"\t{col_name}"
            else:
                data_text += f"\t{col_name}"

        # 데이터 텍스트에 파장별 EL 값 추가
        for j, wl in enumerate(wavelengths):
            row_text = f"{wl:.2f}"

            for col_name in oc_columns:
                # OC 데이터 보간
                f_oc = interpolate.interp1d(
                    df_oc['wavelength'], df_oc[col_name], kind='cubic', bounds_error=False, fill_value=0)
                oc_val = f_oc(wl)

                # PL 데이터 보간
                pl_val = f_pl(wl)

                # EL 계산
                el_val = oc_val * pl_val

                # 최대값으로 정규화
                max_el = np.max(el_values)
                if max_el > 0:
                    el_val_normalized = el_val / max_el
                else:
                    el_val_normalized = el_val

                row_text += f"\t{el_val_normalized:.6f}"

            data_text += f"\n{row_text}"

        # 레이아웃 설정
        fig.update_layout(
            xaxis_title=dict(
                text='파장 (nm)',
                font=dict(size=10)  # x축 제목 폰트 크기
            ),
            yaxis_title=dict(
                text='EL 강도 (정규화)',
                font=dict(size=10)  # y축 제목 폰트 크기
            ),
            template='plotly_white',
            margin=dict(l=50, r=50, t=30, b=50),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            )
        )

        return fig, data_text

    except Exception as e:
        return create_empty_figure("", f"계산 중 오류 발생: {str(e)}"), ""
