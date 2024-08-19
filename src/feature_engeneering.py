import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


def read_data(path, cat_cols, num_cols, new_num_cols, args):
    df = pd.read_csv(path, low_memory=False)

    # 'age_approx' の変換と欠損値の処理
    df['age_approx'] = df['age_approx'].replace('NA', np.nan).astype(float)
    df['age_approx'] = df['age_approx'].fillna(df['age_approx'].median())

    # 新しい列の計算
    df['lesion_size_ratio'] = df['tbp_lv_minorAxisMM'] / df['clin_size_long_diam_mm']
    df['lesion_shape_index'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM'] ** 2)
    df['hue_contrast'] = (df['tbp_lv_H'] - df['tbp_lv_Hext']).abs()
    df['luminance_contrast'] = (df['tbp_lv_L'] - df['tbp_lv_Lext']).abs()
    df['lesion_color_difference'] = np.sqrt(df['tbp_lv_deltaA'] ** 2 + df['tbp_lv_deltaB'] ** 2 + df['tbp_lv_deltaL'] ** 2)
    df['border_complexity'] = df['tbp_lv_norm_border'] + df['tbp_lv_symm_2axis']
    df['color_uniformity'] = df['tbp_lv_color_std_mean'] / (df['tbp_lv_radial_color_std_max'] + args.err)

    # 追加の列計算
    df['position_distance_3d'] = np.sqrt(df['tbp_lv_x'] ** 2 + df['tbp_lv_y'] ** 2 + df['tbp_lv_z'] ** 2)
    df['perimeter_to_area_ratio'] = df['tbp_lv_perimeterMM'] / df['tbp_lv_areaMM2']
    df['area_to_perimeter_ratio'] = df['tbp_lv_areaMM2'] / df['tbp_lv_perimeterMM']
    df['lesion_visibility_score'] = df['tbp_lv_deltaLBnorm'] + df['tbp_lv_norm_color']
    df['combined_anatomical_site'] = df['anatom_site_general'].astype(str) + '_' + df['tbp_lv_location'].astype(str)
    df['symmetry_border_consistency'] = df['tbp_lv_symm_2axis'] * df['tbp_lv_norm_border']
    df['consistency_symmetry_border'] = df['symmetry_border_consistency'] / (df['tbp_lv_symm_2axis'] + df['tbp_lv_norm_border'])

    # 他の計算
    df['color_consistency'] = df['tbp_lv_stdL'] / df['tbp_lv_Lext']
    df['consistency_color'] = df['tbp_lv_stdL'] * df['tbp_lv_Lext'] / (df['tbp_lv_stdL'] + df['tbp_lv_Lext'])
    df['size_age_interaction'] = df['clin_size_long_diam_mm'] * df['age_approx']
    df['hue_color_std_interaction'] = df['tbp_lv_H'] * df['tbp_lv_color_std_mean']
    df['lesion_severity_index'] = (df['tbp_lv_norm_border'] + df['tbp_lv_norm_color'] + df['tbp_lv_eccentricity']) / 3
    df['shape_complexity_index'] = df['border_complexity'] + df['lesion_shape_index']
    df['color_contrast_index'] = df['tbp_lv_deltaA'] + df['tbp_lv_deltaB'] + df['tbp_lv_deltaL'] + df['tbp_lv_deltaLBnorm']

    # さらなる列の追加
    df['log_lesion_area'] = np.log(df['tbp_lv_areaMM2'] + 1)
    df['normalized_lesion_size'] = df['clin_size_long_diam_mm'] / df['age_approx']
    df['mean_hue_difference'] = (df['tbp_lv_H'] + df['tbp_lv_Hext']) / 2
    df['std_dev_contrast'] = np.sqrt((df['tbp_lv_deltaA'] ** 2 + df['tbp_lv_deltaB'] ** 2 + df['tbp_lv_deltaL'] ** 2) / 3)
    df['color_shape_composite_index'] = (df['tbp_lv_color_std_mean'] + df['tbp_lv_area_perim_ratio'] + df['tbp_lv_symm_2axis']) / 3
    df['lesion_orientation_3d'] = np.arctan2(df['tbp_lv_y'], df['tbp_lv_x'])
    df['overall_color_difference'] = (df['tbp_lv_deltaA'] + df['tbp_lv_deltaB'] + df['tbp_lv_deltaL']) / 3

    # 列の更なる操作
    df['symmetry_perimeter_interaction'] = df['tbp_lv_symm_2axis'] * df['tbp_lv_perimeterMM']
    df['comprehensive_lesion_index'] = (df['tbp_lv_area_perim_ratio'] + df['tbp_lv_eccentricity'] + df['tbp_lv_norm_color'] + df['tbp_lv_symm_2axis']) / 4
    df['color_variance_ratio'] = df['tbp_lv_color_std_mean'] / df['tbp_lv_stdLExt']
    df['border_color_interaction'] = df['tbp_lv_norm_border'] * df['tbp_lv_norm_color']
    df['border_color_interaction_2'] = df['border_color_interaction'] / (df['tbp_lv_norm_border'] + df['tbp_lv_norm_color'])
    df['size_color_contrast_ratio'] = df['clin_size_long_diam_mm'] / df['tbp_lv_deltaLBnorm']
    df['age_normalized_nevi_confidence'] = df['tbp_lv_nevi_confidence'] / df['age_approx']
    df['age_normalized_nevi_confidence_2'] = np.sqrt(df['clin_size_long_diam_mm']**2 + df['age_approx']**2)
    df['color_asymmetry_index'] = df['tbp_lv_radial_color_std_max'] * df['tbp_lv_symm_2axis']

    # 最後の列操作
    df['volume_approximation_3d'] = df['tbp_lv_areaMM2'] * np.sqrt(df['tbp_lv_x']**2 + df['tbp_lv_y']**2 + df['tbp_lv_z']**2)
    df['color_range'] = (df['tbp_lv_L'] - df['tbp_lv_Lext']).abs() + (df['tbp_lv_A'] - df['tbp_lv_Aext']).abs() + (df['tbp_lv_B'] - df['tbp_lv_Bext']).abs()
    df['shape_color_consistency'] = df['tbp_lv_eccentricity'] * df['tbp_lv_color_std_mean']
    df['border_length_ratio'] = df['tbp_lv_perimeterMM'] / (2 * np.pi * np.sqrt(df['tbp_lv_areaMM2'] / np.pi))
    df['age_size_symmetry_index'] = df['age_approx'] * df['clin_size_long_diam_mm'] * df['tbp_lv_symm_2axis']
    df['index_age_size_symmetry'] = df['age_approx'] * df['tbp_lv_areaMM2'] * df['tbp_lv_symm_2axis']

    # 患者ごとの正規化
    normalized_columns = {}
    for col in num_cols + new_num_cols:
        mean = df.groupby('patient_id')[col].transform('mean')
        std = df.groupby('patient_id')[col].transform('std')
        normalized_columns[f'{col}_patient_norm'] = (df[col] - mean) / (std + args.err)

    # 一度にすべての新しい列を追加
    df = pd.concat([df, pd.DataFrame(normalized_columns)], axis=1)

    # 患者ごとのカウント
    df = df.assign(count_per_patient=df.groupby('patient_id')['isic_id'].transform('count'))

    # カテゴリカル変数の型変換
    df[cat_cols] = df[cat_cols].astype('category')

    return df



def feature_engeneering_for_cnn(df, err=1e-5):
    df_for_cnn = df.copy()
     # 'age_approx' の変換と欠損値の処理
    df_for_cnn['age_approx'] = df_for_cnn['age_approx'].replace('NA', np.nan).astype(float)
    df_for_cnn['age_approx'] = df_for_cnn['age_approx'].fillna(df_for_cnn['age_approx'].median())

    # meta_cols のみを使用して常に計算する特徴量
    df_for_cnn['lesion_size_ratio'] = df_for_cnn['tbp_lv_minorAxisMM'] / df_for_cnn['clin_size_long_diam_mm']
    df_for_cnn['hue_contrast'] = (df_for_cnn['tbp_lv_H'] - df_for_cnn['tbp_lv_Hext']).abs()
    df_for_cnn['luminance_contrast'] = (df_for_cnn['tbp_lv_L'] - df_for_cnn['tbp_lv_Lext']).abs()
    df_for_cnn['lesion_color_difference'] = np.sqrt(df_for_cnn['tbp_lv_deltaA'] ** 2 + df_for_cnn['tbp_lv_deltaB'] ** 2 + df_for_cnn['tbp_lv_deltaL'] ** 2)
    df_for_cnn['border_complexity'] = df_for_cnn['tbp_lv_norm_border'] + df_for_cnn['tbp_lv_symm_2axis']
    df_for_cnn['color_uniformity'] = df_for_cnn['tbp_lv_color_std_mean'] / (df_for_cnn['tbp_lv_radial_color_std_max'] + err)
    df_for_cnn['position_distance_3d'] = np.sqrt(df_for_cnn['tbp_lv_x'] ** 2 + df_for_cnn['tbp_lv_y'] ** 2 + df_for_cnn['tbp_lv_z'] ** 2)
    df_for_cnn['perimeter_to_area_ratio'] = df_for_cnn['tbp_lv_perimeterMM'] / df_for_cnn['tbp_lv_areaMM2']
    df_for_cnn['area_to_perimeter_ratio'] = df_for_cnn['tbp_lv_areaMM2'] / df_for_cnn['tbp_lv_perimeterMM']
    df_for_cnn['lesion_visibility_score'] = df_for_cnn['tbp_lv_deltaLBnorm'] + df_for_cnn['tbp_lv_norm_color']
    df_for_cnn['symmetry_border_consistency'] = df_for_cnn['tbp_lv_symm_2axis'] * df_for_cnn['tbp_lv_norm_border']
    df_for_cnn['color_consistency'] = df_for_cnn['tbp_lv_stdL'] / df_for_cnn['tbp_lv_Lext']
    df_for_cnn['hue_color_std_interaction'] = df_for_cnn['tbp_lv_H'] * df_for_cnn['tbp_lv_color_std_mean']
    df_for_cnn['lesion_severity_index'] = (df_for_cnn['tbp_lv_norm_border'] + df_for_cnn['tbp_lv_norm_color'] + df_for_cnn['tbp_lv_eccentricity']) / 3
    df_for_cnn['color_contrast_index'] = df_for_cnn['tbp_lv_deltaA'] + df_for_cnn['tbp_lv_deltaB'] + df_for_cnn['tbp_lv_deltaL'] + df_for_cnn['tbp_lv_deltaLBnorm']
    df_for_cnn['log_lesion_area'] = np.log(df_for_cnn['tbp_lv_areaMM2'] + 1)
    df_for_cnn['mean_hue_difference'] = (df_for_cnn['tbp_lv_H'] + df_for_cnn['tbp_lv_Hext']) / 2
    df_for_cnn['std_dev_contrast'] = np.sqrt((df_for_cnn['tbp_lv_deltaA'] ** 2 + df_for_cnn['tbp_lv_deltaB'] ** 2 + df_for_cnn['tbp_lv_deltaL'] ** 2) / 3)
    df_for_cnn['lesion_orientation_3d'] = np.arctan2(df_for_cnn['tbp_lv_y'], df_for_cnn['tbp_lv_x'])
    df_for_cnn['overall_color_difference'] = (df_for_cnn['tbp_lv_deltaA'] + df_for_cnn['tbp_lv_deltaB'] + df_for_cnn['tbp_lv_deltaL']) / 3
    df_for_cnn['symmetry_perimeter_interaction'] = df_for_cnn['tbp_lv_symm_2axis'] * df_for_cnn['tbp_lv_perimeterMM']
    df_for_cnn['color_variance_ratio'] = df_for_cnn['tbp_lv_color_std_mean'] / df_for_cnn['tbp_lv_stdLExt']
    df_for_cnn['border_color_interaction'] = df_for_cnn['tbp_lv_norm_border'] * df_for_cnn['tbp_lv_norm_color']
    df_for_cnn['size_color_contrast_ratio'] = df_for_cnn['clin_size_long_diam_mm'] / df_for_cnn['tbp_lv_deltaLBnorm']
    df_for_cnn['color_asymmetry_index'] = df_for_cnn['tbp_lv_radial_color_std_max'] * df_for_cnn['tbp_lv_symm_2axis']
    df_for_cnn['volume_approximation_3d'] = df_for_cnn['tbp_lv_areaMM2'] * np.sqrt(df_for_cnn['tbp_lv_x']**2 + df_for_cnn['tbp_lv_y']**2 + df_for_cnn['tbp_lv_z']**2)
    df_for_cnn['color_range'] = (df_for_cnn['tbp_lv_L'] - df_for_cnn['tbp_lv_Lext']).abs() + (df_for_cnn['tbp_lv_A'] - df_for_cnn['tbp_lv_Aext']).abs() + (df_for_cnn['tbp_lv_B'] - df_for_cnn['tbp_lv_Bext']).abs()
    df_for_cnn['shape_color_consistency'] = df_for_cnn['tbp_lv_eccentricity'] * df_for_cnn['tbp_lv_color_std_mean']
    df_for_cnn['border_length_ratio'] = df_for_cnn['tbp_lv_perimeterMM'] / (2 * np.pi * np.sqrt(df_for_cnn['tbp_lv_areaMM2'] / np.pi))

    return df_for_cnn


def preprocess(df_train, df_test, feature_cols, cat_cols):
    
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32, handle_unknown='ignore')
    encoder.fit(df_train[cat_cols])
    
    new_cat_cols = [f'onehot_{i}' for i in range(len(encoder.get_feature_names_out()))]

    df_train[new_cat_cols] = encoder.transform(df_train[cat_cols])
    df_train[new_cat_cols] = df_train[new_cat_cols].astype('category')

    df_test[new_cat_cols] = encoder.transform(df_test[cat_cols])
    df_test[new_cat_cols] = df_test[new_cat_cols].astype('category')

    for col in cat_cols:
        feature_cols.remove(col)

    feature_cols.extend(new_cat_cols)
    
    return df_train, df_test, feature_cols, new_cat_cols