import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import LocalOutlierFactor
import scipy

def load_data(path):
    data_configs = {}

    data = pd.read_csv(path, encoding='cp949', index_col=0, low_memory=False)
    
    data_configs['data'] = data
    data_configs['numeric_features'] = data.select_dtypes(['int64', 'float64']).columns
    data_configs['object_features'] = data.select_dtypes(['object']).columns

    return data_configs

def check_fail_rate(data):
    num_pass = len(data[data['passorfail'] == 0])
    num_fail = len(data[data['passorfail'] == 1])

    print(f"합격 데이터 수 : {num_pass}")
    print(f"불합격 데이터 수 : {num_fail}")
    print(f"불합격 데이터 비율 : {round(num_fail / (num_pass + num_fail) * 100, 2)} %")

def check_nan(data, percent=False):
    if percent:
        print(data.isna().sum() / len(data))
    else:
        print(data.isna().sum())

class NanProcessor:
    def __init__(self, nan_grid):
        self.drop_features = nan_grid['drop_features']
        self.simple_fill_dict = nan_grid['simple_fill_dict']
        self.mode_fill_features = nan_grid['mode_fill_features']
        self.mode_criterion = nan_grid['mode_criterion']
    
    def process(self, data):
        data = data.dropna(subset=['passorfail'])

        data = data.drop(columns=self.drop_features)

        for feature, fill_val in self.simple_fill_dict.items():
            data[feature] = data[feature].fillna(fill_val)
        
        for feature in self.mode_fill_features:
            data[feature] = data.groupby(self.mode_criterion)[feature].transform(
                lambda x : x.fillna(x.mode()[0] if not x.mode().empty else x.mean())
            )
        
        data = data.reset_index(drop=True)

        return data

class CatFeatureEncoder:
    def __init__(self, encode_grid):
        self.encode_grid = encode_grid
    
    def process(self, data):
        for feature, ordinal in self.encode_grid.items():
            encoder = OrdinalEncoder(categories=[ordinal])
            data[[feature]] = encoder.fit_transform(data[[feature]])
        
        return data

def remove_outlier_by_lof(data, features):
    for feature in features:
        lof = LocalOutlierFactor(n_neighbors=10, n_jobs=-1)

        y_pred = lof.fit_predict(data[[feature]])
    
        data['outlier'] = y_pred
        
        data = data[data['outlier'] == 1].reset_index(drop=True)

        data = data.drop(columns=['outlier'])
    
    return data

class T_Testor:
    def __init__(self, p_threshold=0.05):
        self.p_threshold = p_threshold

    def test(self, data):
        self.t_test_configs = {}

        self.t_test = []
        self.useful_features = []

        t_test_features = data.columns
        t_test_features = [feature for feature in t_test_features if feature != 'passorfail']

        for feature in t_test_features:
            t=scipy.stats.ttest_ind(data[data['passorfail']==1][feature], 
                                    data[data['passorfail']==0][feature],
                                    equal_var=False)
            
            self.t_test.append([feature, t[0], t[1]])
            
        self.t_test = pd.DataFrame(self.t_test, columns=['col', 'tvalue', 'pvalue'])
        
        for idx in range(len(self.t_test)):
            if self.t_test['pvalue'][idx] < self.p_threshold:
                self.useful_features.append(self.t_test['col'][idx])

        self.t_test_configs['t_test'] = self.t_test
        self.t_test_configs['useful_features'] = self.useful_features

        return self.t_test_configs
    
    def get_useful_data(self, data):
        return data[self.useful_features]

class KampDataLoader:
    def __init__(self, path, nan_grid, encode_grid, p_threshold=0.05, get_useful_p_data=False):
        self.path = path
        self.nan_grid = nan_grid
        self.encode_grid = encode_grid
        self.p_threshold = p_threshold
        self.get_useful_p_data = get_useful_p_data
    
    def process(self):
        # load raw data
        print("[process Log] Loading Raw Data...")
        data_configs = load_data(self.path)
        print("[process Log] Done\n")

        data = data_configs['data']
        numeric_features = data_configs['numeric_features']
        object_features = data_configs['object_features']

        print("[process Log] Processing Nan Value...")
        data = NanProcessor(nan_grid=self.nan_grid).process(data)
        print("[process Log] Done\n")
        
        print("[process Log] Encoding Categorical Features...")
        data = CatFeatureEncoder(encode_grid=self.encode_grid).process(data)
        print("[process Log] Done\n")

        print("[process Log] Removing Outliers (LOF)...")
        numeric_features = [feature for feature in numeric_features 
                    if feature not in ['count', 'molten_volume', 'passorfail', 'mold_code']]
        data = remove_outlier_by_lof(data, numeric_features)
        print("[process Log] Done\n")

        if self.get_useful_p_data:
            print("[process Log] T-Testing...")
            t_test = T_Testor(p_threshold=self.p_threshold)
            t_test_configs = t_test.test(data)
            data = t_test.get_useful_data(data)
            print("[process Log] Done\n")
        
        self.data = data

    def load(self):
        return self.data
    
    def save(self, path):
        self.data.to_csv(path, encoding='cp949', index=False)