import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import joblib
import os

class CustomOneHotEncoder:
    def __init__(self, categorical_columns):
        self.categorical_columns = categorical_columns
        self.encoders = {}
        self.encoded_feature_names = {}

    def fit(self, df):
        for col in self.categorical_columns:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoder.fit(df[[col]])
            self.encoders[col] = encoder
            self.encoded_feature_names[col] = encoder.get_feature_names_out([col])

    def transform(self, df):
        encoded_data = []
        for col in self.categorical_columns:
            encoder = self.encoders[col]
            encoded_col = encoder.transform(df[[col]])
            encoded_data.append(pd.DataFrame(encoded_col, columns=self.encoded_feature_names[col], index=df.index))
        
        return pd.concat(encoded_data, axis=1)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def save(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        return joblib.load(filename)


def read_data(path, cat_cols, num_cols, new_num_cols, err=1e-6):
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
    df['color_uniformity'] = df['tbp_lv_color_std_mean'] / (df['tbp_lv_radial_color_std_max'] + err)

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
        normalized_columns[f'{col}_patient_norm'] = (df[col] - mean) / (std + err)

    # 一度にすべての新しい列を追加
    df = pd.concat([df, pd.DataFrame(normalized_columns)], axis=1)

    # 患者ごとのカウント
    df = df.assign(count_per_patient=df.groupby('patient_id')['isic_id'].transform('count'))

    # カテゴリカル変数の型変換
    df[cat_cols] = df[cat_cols].astype('category')

    return df



def feature_engeneering_for_cnn(df, err=1e-5):
    df_for_cnn = df.copy()

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

def prepare_data_for_training(df, cat_cols, logdir):
    # OneHotEncodingの適用
    encoder = CustomOneHotEncoder(cat_cols)
    encoded_cat_data = encoder.fit_transform(df[cat_cols])

    new_cat_cols = encoded_cat_data.columns.to_list()
    df[new_cat_cols] = encoded_cat_data
    
    # エンコーダーの保存
    encoder.save(os.path.join(logdir, 'onehot_encoder.joblib'))
    
    return df, new_cat_cols

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


def get_feature_cols():
    num_cols = [
        'age_approx',                        # Approximate age of patient at time of imaging.
        'clin_size_long_diam_mm',            # Maximum diameter of the lesion (mm).+
        'tbp_lv_A',                          # A inside  lesion.+
        'tbp_lv_Aext',                       # A outside lesion.+
        'tbp_lv_B',                          # B inside  lesion.+
        'tbp_lv_Bext',                       # B outside lesion.+ 
        'tbp_lv_C',                          # Chroma inside  lesion.+
        'tbp_lv_Cext',                       # Chroma outside lesion.+
        'tbp_lv_H',                          # Hue inside the lesion; calculated as the angle of A* and B* in LAB* color space. Typical values range from 25 (red) to 75 (brown).+
        'tbp_lv_Hext',                       # Hue outside lesion.+
        'tbp_lv_L',                          # L inside lesion.+
        'tbp_lv_Lext',                       # L outside lesion.+
        'tbp_lv_areaMM2',                    # Area of lesion (mm^2).+
        'tbp_lv_area_perim_ratio',           # Border jaggedness, the ratio between lesions perimeter and area. Circular lesions will have low values; irregular shaped lesions will have higher values. Values range 0-10.+
        'tbp_lv_color_std_mean',             # Color irregularity, calculated as the variance of colors within the lesion's boundary.
        'tbp_lv_deltaA',                     # Average A contrast (inside vs. outside lesion).+
        'tbp_lv_deltaB',                     # Average B contrast (inside vs. outside lesion).+
        'tbp_lv_deltaL',                     # Average L contrast (inside vs. outside lesion).+
        'tbp_lv_deltaLB',                    #
        'tbp_lv_deltaLBnorm',                # Contrast between the lesion and its immediate surrounding skin. Low contrast lesions tend to be faintly visible such as freckles; high contrast lesions tend to be those with darker pigment. Calculated as the average delta LB of the lesion relative to its immediate background in LAB* color space. Typical values range from 5.5 to 25.+
        'tbp_lv_eccentricity',               # Eccentricity.+
        'tbp_lv_minorAxisMM',                # Smallest lesion diameter (mm).+
        'tbp_lv_nevi_confidence',            # Nevus confidence score (0-100 scale) is a convolutional neural network classifier estimated probability that the lesion is a nevus. The neural network was trained on approximately 57,000 lesions that were classified and labeled by a dermatologist.+,++
        'tbp_lv_norm_border',                # Border irregularity (0-10 scale); the normalized average of border jaggedness and asymmetry.+
        'tbp_lv_norm_color',                 # Color variation (0-10 scale); the normalized average of color asymmetry and color irregularity.+
        'tbp_lv_perimeterMM',                # Perimeter of lesion (mm).+
        'tbp_lv_radial_color_std_max',       # Color asymmetry, a measure of asymmetry of the spatial distribution of color within the lesion. This score is calculated by looking at the average standard deviation in LAB* color space within concentric rings originating from the lesion center. Values range 0-10.+
        'tbp_lv_stdL',                       # Standard deviation of L inside  lesion.+
        'tbp_lv_stdLExt',                    # Standard deviation of L outside lesion.+
        'tbp_lv_symm_2axis',                 # Border asymmetry; a measure of asymmetry of the lesion's contour about an axis perpendicular to the lesion's most symmetric axis. Lesions with two axes of symmetry will therefore have low scores (more symmetric), while lesions with only one or zero axes of symmetry will have higher scores (less symmetric). This score is calculated by comparing opposite halves of the lesion contour over many degrees of rotation. The angle where the halves are most similar identifies the principal axis of symmetry, while the second axis of symmetry is perpendicular to the principal axis. Border asymmetry is reported as the asymmetry value about this second axis. Values range 0-10.+
        'tbp_lv_symm_2axis_angle',           # Lesion border asymmetry angle.+
        'tbp_lv_x',                          # X-coordinate of the lesion on 3D TBP.+
        'tbp_lv_y',                          # Y-coordinate of the lesion on 3D TBP.+
        'tbp_lv_z',                          # Z-coordinate of the lesion on 3D TBP.+
    ]

    new_num_cols = [
        'lesion_size_ratio',             # tbp_lv_minorAxisMM      / clin_size_long_diam_mm
        'lesion_shape_index',            # tbp_lv_areaMM2          / tbp_lv_perimeterMM **2
        'hue_contrast',                  # tbp_lv_H                - tbp_lv_Hext              abs
        'luminance_contrast',            # tbp_lv_L                - tbp_lv_Lext              abs
        'lesion_color_difference',       # tbp_lv_deltaA **2       + tbp_lv_deltaB **2 + tbp_lv_deltaL **2  sqrt  
        'border_complexity',             # tbp_lv_norm_border      + tbp_lv_symm_2axis
        'color_uniformity',              # tbp_lv_color_std_mean   / tbp_lv_radial_color_std_max

        'position_distance_3d',          # tbp_lv_x **2 + tbp_lv_y **2 + tbp_lv_z **2  sqrt
        'perimeter_to_area_ratio',       # tbp_lv_perimeterMM      / tbp_lv_areaMM2
        'area_to_perimeter_ratio',       # tbp_lv_areaMM2          / tbp_lv_perimeterMM
        'lesion_visibility_score',       # tbp_lv_deltaLBnorm      + tbp_lv_norm_color
        'symmetry_border_consistency',   # tbp_lv_symm_2axis       * tbp_lv_norm_border
        'consistency_symmetry_border',   # tbp_lv_symm_2axis       * tbp_lv_norm_border / (tbp_lv_symm_2axis + tbp_lv_norm_border)

        'color_consistency',             # tbp_lv_stdL             / tbp_lv_Lext
        'consistency_color',             # tbp_lv_stdL*tbp_lv_Lext / tbp_lv_stdL + tbp_lv_Lext
        'size_age_interaction',          # clin_size_long_diam_mm  * age_approx
        'hue_color_std_interaction',     # tbp_lv_H                * tbp_lv_color_std_mean
        'lesion_severity_index',         # tbp_lv_norm_border      + tbp_lv_norm_color + tbp_lv_eccentricity / 3
        'shape_complexity_index',        # border_complexity       + lesion_shape_index
        'color_contrast_index',          # tbp_lv_deltaA + tbp_lv_deltaB + tbp_lv_deltaL + tbp_lv_deltaLBnorm

        'log_lesion_area',               # tbp_lv_areaMM2          + 1  np.log
        'normalized_lesion_size',        # clin_size_long_diam_mm  / age_approx
        'mean_hue_difference',           # tbp_lv_H                + tbp_lv_Hext    / 2
        'std_dev_contrast',              # tbp_lv_deltaA **2 + tbp_lv_deltaB **2 + tbp_lv_deltaL **2   / 3  np.sqrt
        'color_shape_composite_index',   # tbp_lv_color_std_mean   + bp_lv_area_perim_ratio + tbp_lv_symm_2axis   / 3
        'lesion_orientation_3d',         # tbp_lv_y                , tbp_lv_x  np.arctan2
        'overall_color_difference',      # tbp_lv_deltaA           + tbp_lv_deltaB + tbp_lv_deltaL   / 3

        'symmetry_perimeter_interaction',# tbp_lv_symm_2axis       * tbp_lv_perimeterMM
        'comprehensive_lesion_index',    # tbp_lv_area_perim_ratio + tbp_lv_eccentricity + bp_lv_norm_color + tbp_lv_symm_2axis   / 4
        'color_variance_ratio',          # tbp_lv_color_std_mean   / tbp_lv_stdLExt
        'border_color_interaction',      # tbp_lv_norm_border      * tbp_lv_norm_color
        'border_color_interaction_2',
        'size_color_contrast_ratio',     # clin_size_long_diam_mm  / tbp_lv_deltaLBnorm
        'age_normalized_nevi_confidence',# tbp_lv_nevi_confidence  / age_approx
        'age_normalized_nevi_confidence_2',
        'color_asymmetry_index',         # tbp_lv_symm_2axis       * tbp_lv_radial_color_std_max

        'volume_approximation_3d',       # tbp_lv_areaMM2          * sqrt(tbp_lv_x**2 + tbp_lv_y**2 + tbp_lv_z**2)
        'color_range',                   # abs(tbp_lv_L - tbp_lv_Lext) + abs(tbp_lv_A - tbp_lv_Aext) + abs(tbp_lv_B - tbp_lv_Bext)
        'shape_color_consistency',       # tbp_lv_eccentricity     * tbp_lv_color_std_mean
        'border_length_ratio',           # tbp_lv_perimeterMM      / pi * sqrt(tbp_lv_areaMM2 / pi)
        'age_size_symmetry_index',       # age_approx              * clin_size_long_diam_mm * tbp_lv_symm_2axis
        'index_age_size_symmetry',       # age_approx              * tbp_lv_areaMM2 * tbp_lv_symm_2axis
    ]

    cat_cols = ['sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location', 'tbp_lv_location_simple', 'attribution']
    norm_cols = [f'{col}_patient_norm' for col in num_cols + new_num_cols]
    special_cols = ['count_per_patient']
    return cat_cols, num_cols, new_num_cols, norm_cols + special_cols