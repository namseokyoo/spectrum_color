import pandas as pd
import re
import numpy as np


def parse_text_data(text_data):
    """스펙트럼 텍스트 데이터 파싱"""
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
                    pass  # 숫자로 변환할 수 없는 행은 건너뜀

        if not data:
            return None

        # 데이터프레임으로 변환
        df = pd.DataFrame(data, columns=['wavelength', 'intensity'])
        return df
    except Exception as e:
        print(f"데이터 파싱 오류: {e}")
        return None


def parse_text_data_flexible(data, multi_column=False):
    """
    다양한 형식의 텍스트 데이터를 파싱하여 DataFrame으로 변환
    multi_column: True이면 다중 열 데이터 처리
    """
    if not data:
        return None
    
    try:
        # 줄 단위로 분리
        lines = data.strip().split('\n')
        
        # 빈 줄 제거
        lines = [line.strip() for line in lines if line.strip()]
        
        # 구분자 감지 (탭, 쉼표, 공백)
        first_line = lines[0]
        if '\t' in first_line:
            delimiter = '\t'
        elif ',' in first_line:
            delimiter = ','
        else:
            delimiter = None  # 공백으로 구분
        
        # 데이터 파싱
        data_rows = []
        for line in lines:
            if delimiter:
                values = [val.strip() for val in line.split(delimiter) if val.strip()]
            else:
                values = [val.strip() for val in re.split(r'\s+', line) if val.strip()]
            
            # 숫자로 변환 가능한 값만 추출
            numeric_values = []
            for val in values:
                try:
                    numeric_values.append(float(val))
                except ValueError:
                    # 숫자가 아닌 경우 헤더일 수 있으므로 무시
                    pass
            
            if numeric_values:
                data_rows.append(numeric_values)
        
        # 모든 행의 길이가 같은지 확인
        if not data_rows:
            return None
        
        row_lengths = [len(row) for row in data_rows]
        if len(set(row_lengths)) > 1:
            # 행 길이가 다른 경우, 가장 긴 행을 기준으로 설정
            max_length = max(row_lengths)
            data_rows = [row + [np.nan] * (max_length - len(row)) for row in data_rows]
        
        # DataFrame 생성
        if multi_column:
            # 다중 열 데이터 처리
            df = pd.DataFrame(data_rows)
            if len(df.columns) > 1:
                # 첫 번째 열을 파장으로 설정
                df.columns = ['wavelength'] + [f'intensity_{i}' for i in range(len(df.columns) - 1)]
            else:
                df.columns = ['wavelength']
        else:
            # 단일 열 데이터 처리
            if len(data_rows[0]) == 1:
                # 파장 값이 없는 경우, 인덱스를 파장으로 사용
                wavelengths = list(range(len(data_rows)))
                intensities = [row[0] for row in data_rows]
                df = pd.DataFrame({'wavelength': wavelengths, 'intensity': intensities})
            else:
                # 파장과 강도 값이 있는 경우
                df = pd.DataFrame(data_rows, columns=['wavelength', 'intensity'])
        
        return df
    
    except Exception as e:
        print(f"데이터 파싱 오류: {str(e)}")
        return None


def parse_multi_sample_data(text_data):
    """다중 샘플 데이터 파싱"""
    try:
        # 텍스트를 줄 단위로 분리
        lines = text_data.strip().split('\n')
        samples = []

        # 헤더 행 감지 플래그
        header_detected = False

        for i, line in enumerate(lines):
            # 빈 줄 건너뛰기
            if not line.strip():
                continue

            # 쉼표로 구분된 값 처리
            if ',' in line:
                values = [v.strip() for v in line.split(',')]
                # 첫 번째 값이 샘플 이름
                if len(values) >= 7:
                    sample_name = values[0]
                    coords = values[1:7]
                else:
                    continue
            # 공백이나 탭으로 분리된 값들을 처리
            else:
                values = line.strip().split()

                # 헤더 행 감지 (숫자가 아닌 값이 있는지 확인)
                is_header = False
                for val in values:
                    # 숫자로 변환 시도
                    try:
                        float(val)
                    except ValueError:
                        # 숫자로 변환할 수 없으면 헤더로 간주
                        is_header = True
                        header_detected = True
                        break

                # 헤더 행이면 건너뛰기
                if is_header:
                    continue

                # 샘플 이름이 없는 경우 자동 생성
                if len(values) >= 6:
                    sample_name = f"Sample{i+1}"
                    coords = values[0:6]
                else:
                    continue

            try:
                rx, ry = float(coords[0]), float(coords[1])
                gx, gy = float(coords[2]), float(coords[3])
                bx, by = float(coords[4]), float(coords[5])

                # 샘플 데이터 구성
                sample = {
                    'name': sample_name,
                    'red': (rx, ry),
                    'green': (gx, gy),
                    'blue': (bx, by)
                }
                samples.append(sample)
            except (ValueError, IndexError) as e:
                print(f"샘플 데이터 파싱 오류: {e}")
                continue

        return samples
    except Exception as e:
        print(f"다중 샘플 데이터 파싱 오류: {e}")
        return None
