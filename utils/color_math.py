import numpy as np
import colorsys
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from scipy import interpolate
import pandas as pd
import matplotlib.path as mpath


def get_color_spaces():
    """표준 색공간 정보 반환"""
    color_spaces = {
        'DCI-P3': {
            'red': (0.680, 0.320),
            'green': (0.265, 0.690),
            'blue': (0.150, 0.060),
            'white': (0.3127, 0.3290)
        },
        'adobeRGB': {
            'red': (0.640, 0.330),
            'green': (0.210, 0.710),
            'blue': (0.150, 0.060),
            'white': (0.3127, 0.3290)
        },
        'BT2020': {
            'red': (0.708, 0.292),
            'green': (0.170, 0.797),
            'blue': (0.131, 0.046),
            'white': (0.3127, 0.3290)
        }
    }
    return color_spaces


def xy_to_uv(x, y):
    """CIE 1931 xy 좌표를 CIE 1976 u'v' 좌표로 변환"""
    denom = -2 * x + 12 * y + 3
    u_prime = 4 * x / denom
    v_prime = 9 * y / denom
    return u_prime, v_prime


def triangle_area(vertices):
    """삼각형 면적 계산"""
    hull = ConvexHull(vertices)
    return hull.volume


def overlap_area(vertices1, vertices2):
    """두 다각형의 겹치는 영역 계산"""
    poly1 = Polygon(vertices1)
    poly2 = Polygon(vertices2)

    if poly1.intersects(poly2):
        return poly1.intersection(poly2).area
    return 0.0


def calculate_gamut_coverage(sample, target_space):
    """
    RGB 색좌표에 대한 색역 커버리지 계산 (shapely 사용)

    Parameters:
    sample (dict): {'red': [x, y], 'green': [x, y], 'blue': [x, y]} 형식의 샘플 데이터
    target_space (dict): {'red': [x, y], 'green': [x, y], 'blue': [x, y]} 형식의 색공간

    Returns:
    tuple: (xy_coverage, xy_area, uv_coverage, uv_area)
    """
    try:
        # 샘플 RGB 좌표 가져오기
        rgb_coords_xy = [
            sample['red'],
            sample['green'],
            sample['blue']
        ]

        # 표준 색공간 좌표 가져오기
        target_coords_xy = [
            target_space['red'],
            target_space['green'],
            target_space['blue']
        ]

        # xy 좌표계 계산
        rgb_area_xy = calculate_triangle_area(rgb_coords_xy)
        target_area_xy = calculate_triangle_area(target_coords_xy)
        intersection_area_xy = overlap_area(rgb_coords_xy, target_coords_xy)

        xy_coverage = 100 * intersection_area_xy / \
            target_area_xy if target_area_xy > 0 else 0
        xy_area = 100 * rgb_area_xy / target_area_xy if target_area_xy > 0 else 0

        # uv 좌표계로 변환
        rgb_coords_uv = [
            xy_to_uv(sample['red'][0], sample['red'][1]),
            xy_to_uv(sample['green'][0], sample['green'][1]),
            xy_to_uv(sample['blue'][0], sample['blue'][1])
        ]

        target_coords_uv = [
            xy_to_uv(target_space['red'][0], target_space['red'][1]),
            xy_to_uv(target_space['green'][0], target_space['green'][1]),
            xy_to_uv(target_space['blue'][0], target_space['blue'][1])
        ]

        # uv 좌표계 계산
        rgb_area_uv = calculate_triangle_area(rgb_coords_uv)
        target_area_uv = calculate_triangle_area(target_coords_uv)
        intersection_area_uv = overlap_area(rgb_coords_uv, target_coords_uv)

        uv_coverage = 100 * intersection_area_uv / \
            target_area_uv if target_area_uv > 0 else 0
        uv_area = 100 * rgb_area_uv / target_area_uv if target_area_uv > 0 else 0

        return xy_coverage, xy_area, uv_coverage, uv_area

    except Exception as e:
        print(f"색재현율 계산 오류: {e}")
        return 0, 0, 0, 0


def calculate_triangle_area(vertices):
    """
    삼각형 면적 계산

    Parameters:
    vertices (list): [(x1, y1), (x2, y2), (x3, y3)] 형태의 꼭지점 좌표

    Returns:
    float: 삼각형 면적
    """
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]

    # 삼각형 면적 공식 사용
    area = 0.5 * abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))
    return area


def spectrum_to_cie(wavelengths, intensities):
    """
    스펙트럼 데이터를 CIE XYZ 및 xy 좌표로 변환합니다.

    Parameters:
    -----------
    wavelengths : array-like
        파장 배열 (nm)
    intensities : array-like
        각 파장에 대한 강도 배열

    Returns:
    --------
    X, Y, Z : float
        CIE XYZ 값
    x, y : float
        CIE xy 좌표
    """
    # CIE 1931 표준 관찰자 색매칭 함수 로드
    cmfs_wavelengths, x_bar, y_bar, z_bar = load_cie_cmf()

    # 입력 파장 범위가 색매칭 함수 범위를 벗어나면 0으로 처리
    mask = (wavelengths >= min(cmfs_wavelengths)) & (
        wavelengths <= max(cmfs_wavelengths))
    valid_wavelengths = wavelengths[mask]
    valid_intensities = intensities[mask]

    if len(valid_wavelengths) == 0:
        return 0, 0, 0, 0.3127, 0.3290  # D65 백색점

    # 색매칭 함수를 입력 파장에 맞게 보간
    x_bar_interp = np.interp(valid_wavelengths, cmfs_wavelengths, x_bar)
    y_bar_interp = np.interp(valid_wavelengths, cmfs_wavelengths, y_bar)
    z_bar_interp = np.interp(valid_wavelengths, cmfs_wavelengths, z_bar)

    # XYZ 계산
    X = np.sum(valid_intensities * x_bar_interp)
    Y = np.sum(valid_intensities * y_bar_interp)
    Z = np.sum(valid_intensities * z_bar_interp)

    # 정규화
    norm = X + Y + Z
    if norm == 0:
        return 0, 0, 0, 0.3127, 0.3290  # D65 백색점

    # xy 좌표 계산
    x = X / norm
    y = Y / norm

    return X, Y, Z, x, y


def wavelength_to_xy(wavelength):
    """파장을 CIE 1931 xy 좌표로 변환 (더 정확한 구현)"""
    # 실제 스펙트럼 궤적 데이터 (CIE 1931 표준 관찰자 기반)
    # 이 데이터는 colour 라이브러리에서 가져온 값을 기반으로 함
    spectrum_locus = {
        380: (0.1741, 0.0050),
        385: (0.1740, 0.0050),
        390: (0.1738, 0.0049),
        395: (0.1736, 0.0049),
        400: (0.1733, 0.0048),
        405: (0.1730, 0.0048),
        410: (0.1726, 0.0048),
        415: (0.1721, 0.0048),
        420: (0.1714, 0.0051),
        425: (0.1703, 0.0058),
        430: (0.1689, 0.0069),
        435: (0.1669, 0.0086),
        440: (0.1644, 0.0109),
        445: (0.1611, 0.0138),
        450: (0.1566, 0.0177),
        455: (0.1510, 0.0227),
        460: (0.1440, 0.0297),
        465: (0.1355, 0.0399),
        470: (0.1241, 0.0578),
        475: (0.1096, 0.0868),
        480: (0.0913, 0.1327),
        485: (0.0687, 0.2007),
        490: (0.0454, 0.2950),
        495: (0.0235, 0.4127),
        500: (0.0082, 0.5384),
        505: (0.0039, 0.6548),
        510: (0.0139, 0.7502),
        515: (0.0389, 0.8120),
        520: (0.0743, 0.8338),
        525: (0.1142, 0.8262),
        530: (0.1547, 0.8059),
        535: (0.1929, 0.7816),
        540: (0.2296, 0.7543),
        545: (0.2658, 0.7243),
        550: (0.3016, 0.6923),
        555: (0.3373, 0.6589),
        560: (0.3731, 0.6245),
        565: (0.4087, 0.5896),
        570: (0.4441, 0.5547),
        575: (0.4788, 0.5202),
        580: (0.5125, 0.4866),
        585: (0.5448, 0.4544),
        590: (0.5752, 0.4242),
        595: (0.6029, 0.3965),
        600: (0.6270, 0.3725),
        605: (0.6482, 0.3514),
        610: (0.6658, 0.3340),
        615: (0.6801, 0.3197),
        620: (0.6915, 0.3083),
        625: (0.7006, 0.2993),
        630: (0.7079, 0.2920),
        635: (0.7140, 0.2859),
        640: (0.7190, 0.2809),
        645: (0.7230, 0.2770),
        650: (0.7260, 0.2740),
        655: (0.7283, 0.2717),
        660: (0.7300, 0.2700),
        665: (0.7311, 0.2689),
        670: (0.7320, 0.2680),
        675: (0.7327, 0.2673),
        680: (0.7334, 0.2666),
        685: (0.7340, 0.2660),
        690: (0.7344, 0.2656),
        695: (0.7346, 0.2654),
        700: (0.7347, 0.2653),
        705: (0.7347, 0.2653),
        710: (0.7347, 0.2653),
        715: (0.7347, 0.2653),
        720: (0.7347, 0.2653),
        725: (0.7347, 0.2653),
        730: (0.7347, 0.2653),
        735: (0.7347, 0.2653),
        740: (0.7347, 0.2653),
        745: (0.7347, 0.2653),
        750: (0.7347, 0.2653),
        755: (0.7347, 0.2653),
        760: (0.7347, 0.2653),
        765: (0.7347, 0.2653),
        770: (0.7347, 0.2653),
        775: (0.7347, 0.2653),
        780: (0.7347, 0.2653)
    }

    # 파장이 범위를 벗어나면 경계값 반환
    if wavelength < 380:
        return spectrum_locus[380]
    elif wavelength > 780:
        return spectrum_locus[780]

    # 정확한 파장이 있으면 해당 값 반환
    if wavelength in spectrum_locus:
        return spectrum_locus[wavelength]

    # 없으면 선형 보간
    lower_wl = max([wl for wl in spectrum_locus.keys() if wl < wavelength])
    upper_wl = min([wl for wl in spectrum_locus.keys() if wl > wavelength])

    x1, y1 = spectrum_locus[lower_wl]
    x2, y2 = spectrum_locus[upper_wl]

    # 선형 보간
    ratio = (wavelength - lower_wl) / (upper_wl - lower_wl)
    x = x1 + ratio * (x2 - x1)
    y = y1 + ratio * (y2 - y1)

    return x, y


def generate_spectral_locus(start_nm=380, end_nm=780, step=1):
    """스펙트럼 궤적 생성 (colour 라이브러리 사용)"""
    import colour

    # colour 라이브러리에서 스펙트럼 궤적 가져오기
    cmfs = colour.colorimetry.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']

    # 파장 범위 필터링
    mask = (cmfs.wavelengths >= start_nm) & (cmfs.wavelengths <= end_nm)
    wavelengths = cmfs.wavelengths[mask]

    # 각 파장에 대한 xy 좌표 계산
    xy = colour.XYZ_to_xy(cmfs.values[mask])

    return xy[:, 0], xy[:, 1]


def load_cie_cmf():
    """CIE 1931 표준 관찰자 색매칭 함수 로드 (colour 라이브러리 사용)"""
    import colour

    # colour 라이브러리에서 CIE 1931 2도 관찰자 색매칭 함수 가져오기
    cmfs = colour.colorimetry.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    wavelengths = cmfs.wavelengths
    x_bar = cmfs.values[:, 0]
    y_bar = cmfs.values[:, 1]
    z_bar = cmfs.values[:, 2]

    return wavelengths, x_bar, y_bar, z_bar


# CIE 1931 표준 관찰자 데이터 (2도)
# 파장 범위: 380-780nm, 5nm 간격
WAVELENGTHS = np.arange(380, 785, 5)

# CIE 1931 표준 관찰자 데이터 (2도)
# 출처: http://www.cvrl.org/
CIE_X = np.array([
    0.0014, 0.0022, 0.0042, 0.0076, 0.0143, 0.0232, 0.0435, 0.0776, 0.1344, 0.2148,
    0.2839, 0.3285, 0.3483, 0.3481, 0.3362, 0.3187, 0.2908, 0.2511, 0.1954, 0.1421,
    0.0956, 0.0580, 0.0320, 0.0147, 0.0049, 0.0024, 0.0093, 0.0291, 0.0633, 0.1096,
    0.1655, 0.2257, 0.2904, 0.3597, 0.4334, 0.5121, 0.5945, 0.6784, 0.7621, 0.8425,
    0.9163, 0.9786, 1.0263, 1.0567, 1.0622, 1.0456, 1.0026, 0.9384, 0.8544, 0.7514,
    0.6424, 0.5419, 0.4479, 0.3608, 0.2835, 0.2187, 0.1649, 0.1212, 0.0874, 0.0636,
    0.0468, 0.0329, 0.0227, 0.0158, 0.0114, 0.0081, 0.0058, 0.0041, 0.0029, 0.0020,
    0.0014, 0.0010, 0.0007, 0.0005, 0.0003, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001,
    0.0000
])

CIE_Y = np.array([
    0.0000, 0.0001, 0.0001, 0.0002, 0.0004, 0.0006, 0.0012, 0.0022, 0.0040, 0.0073,
    0.0116, 0.0168, 0.0230, 0.0298, 0.0380, 0.0480, 0.0600, 0.0739, 0.0910, 0.1126,
    0.1390, 0.1693, 0.2080, 0.2586, 0.3230, 0.4073, 0.5030, 0.6082, 0.7100, 0.7932,
    0.8620, 0.9149, 0.9540, 0.9803, 0.9950, 1.0000, 0.9950, 0.9786, 0.9520, 0.9154,
    0.8700, 0.8163, 0.7570, 0.6949, 0.6310, 0.5668, 0.5030, 0.4412, 0.3810, 0.3210,
    0.2650, 0.2170, 0.1750, 0.1382, 0.1070, 0.0816, 0.0610, 0.0446, 0.0320, 0.0232,
    0.0170, 0.0119, 0.0082, 0.0057, 0.0041, 0.0029, 0.0021, 0.0015, 0.0010, 0.0007,
    0.0005, 0.0004, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000,
    0.0000
])

CIE_Z = np.array([
    0.0065, 0.0105, 0.0201, 0.0362, 0.0679, 0.1102, 0.2074, 0.3713, 0.6456, 1.0391,
    1.3856, 1.6230, 1.7471, 1.7826, 1.7721, 1.7441, 1.6692, 1.5281, 1.2876, 1.0419,
    0.8130, 0.6162, 0.4652, 0.3533, 0.2720, 0.2123, 0.1582, 0.1117, 0.0782, 0.0573,
    0.0422, 0.0298, 0.0203, 0.0134, 0.0087, 0.0057, 0.0039, 0.0027, 0.0021, 0.0018,
    0.0017, 0.0014, 0.0011, 0.0010, 0.0008, 0.0006, 0.0003, 0.0002, 0.0002, 0.0001,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000
])

# 색공간 정의
COLOR_SPACES = {
    'DCI-P3': {
        'xy': [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
        'uv': [[0.496, 0.526], [0.099, 0.578], [0.175, 0.158]]
    },
    'adobeRGB': {
        'xy': [[0.640, 0.330], [0.210, 0.710], [0.150, 0.060]],
        'uv': [[0.451, 0.523], [0.076, 0.576], [0.175, 0.158]]
    },
    'BT2020': {
        'xy': [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
        'uv': [[0.526, 0.488], [0.059, 0.621], [0.159, 0.126]]
    }
}


def calculate_xyz_from_spectrum(wavelengths, intensities):
    """
    주어진 스펙트럼에서 XYZ 값을 계산

    Parameters:
    wavelengths (array): 파장 배열 (nm)
    intensities (array): 강도 배열

    Returns:
    tuple: (X, Y, Z) 값
    """
    # 입력 데이터가 표준 파장 범위와 다를 경우 보간
    if len(wavelengths) != len(WAVELENGTHS) or not np.array_equal(wavelengths, WAVELENGTHS):
        # 파장 범위 확인 및 조정
        min_wl = max(np.min(wavelengths), np.min(WAVELENGTHS))
        max_wl = min(np.max(wavelengths), np.max(WAVELENGTHS))

        # 보간 함수 생성
        f = interpolate.interp1d(
            wavelengths, intensities, bounds_error=False, fill_value=0)

        # 표준 파장에 대한 강도 계산
        std_intensities = np.zeros_like(WAVELENGTHS, dtype=float)
        mask = (WAVELENGTHS >= min_wl) & (WAVELENGTHS <= max_wl)
        std_intensities[mask] = f(WAVELENGTHS[mask])
    else:
        std_intensities = intensities

    # XYZ 계산
    X = np.sum(std_intensities * CIE_X) * 5  # 5nm 간격 보정
    Y = np.sum(std_intensities * CIE_Y) * 5
    Z = np.sum(std_intensities * CIE_Z) * 5

    return (X, Y, Z)


def calculate_color_coordinates(xyz, diagram_type='xy'):
    """
    XYZ 값에서 색좌표 계산

    Parameters:
    xyz (tuple): (X, Y, Z) 값
    diagram_type (str): 'xy' 또는 'uv' (CIE 1931 또는 CIE 1976)

    Returns:
    tuple: (x, y) 또는 (u', v') 좌표
    """
    X, Y, Z = xyz
    sum_XYZ = X + Y + Z

    if sum_XYZ == 0:
        return (0, 0)

    # CIE 1931 xy 좌표 계산
    x = X / sum_XYZ
    y = Y / sum_XYZ

    # CIE 1976 u'v' 좌표 계산
    u_prime = 4 * X / (X + 15 * Y + 3 * Z)
    v_prime = 9 * Y / (X + 15 * Y + 3 * Z)

    if diagram_type == 'xy':
        return (x, y)
    else:  # u'v'
        return (u_prime, v_prime)


def calculate_all_color_coordinates(xyz):
    """
    XYZ 값에서 CIE 1931 xy와 CIE 1976 u'v' 좌표를 모두 계산

    Parameters:
    xyz (tuple): (X, Y, Z) 값

    Returns:
    dict: {'xy': (x, y), 'uv': (u', v')} 형태의 딕셔너리
    """
    X, Y, Z = xyz
    sum_XYZ = X + Y + Z

    if sum_XYZ == 0:
        return {'xy': (0, 0), 'uv': (0, 0)}

    # CIE 1931 xy 좌표 계산
    x = X / sum_XYZ
    y = Y / sum_XYZ

    # CIE 1976 u'v' 좌표 계산
    u_prime = 4 * X / (X + 15 * Y + 3 * Z)
    v_prime = 9 * Y / (X + 15 * Y + 3 * Z)

    return {'xy': (x, y), 'uv': (u_prime, v_prime)}


def calculate_color_temperature(xy_coords):
    """
    xy 좌표에서 상관 색온도 계산 (McCamy's 근사식 사용)

    Parameters:
    xy_coords (tuple): (x, y) 좌표

    Returns:
    float: 상관 색온도 (K)
    """
    x, y = xy_coords

    # McCamy's 근사식
    n = (x - 0.3320) / (0.1858 - y)
    CCT = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33

    return CCT


def calculate_duv(xy_coords):
    """
    xy 좌표에서 흑체 궤적으로부터의 편차 (Duv) 계산

    Parameters:
    xy_coords (tuple): (x, y) 좌표

    Returns:
    float: Duv 값
    """
    # 이 함수는 복잡한 계산이 필요하므로 간단한 근사치만 반환
    # 실제 구현에서는 더 정확한 계산 필요
    x, y = xy_coords

    # 흑체 궤적 근사
    if 0.25 <= x <= 0.45 and 0.25 <= y <= 0.45:
        # 매우 간단한 근사
        blackbody_y = -2.5 * x**2 + 2.5 * x + 0.1
        duv = (y - blackbody_y) * 0.5
        return duv
    else:
        return 0.0  # 범위 밖의 값은 0으로 처리


def parse_multi_sample_data(data):
    samples = []
    lines = data.strip().split('\n')

    for i, line in enumerate(lines):
        # 쉼표로 분리하고 각 항목의 앞뒤 공백 제거
        parts = [part.strip() for part in line.split(',')]

        if len(parts) >= 7:  # 샘플 이름 + 6개 좌표값
            try:
                sample = {
                    'name': parts[0],
                    'red': [float(parts[1]), float(parts[2])],
                    'green': [float(parts[3]), float(parts[4])],
                    'blue': [float(parts[5]), float(parts[6])]
                }
                samples.append(sample)
            except (ValueError, IndexError) as e:
                print(f"행 {i+1} 파싱 오류: {e}")
                continue

    return samples
