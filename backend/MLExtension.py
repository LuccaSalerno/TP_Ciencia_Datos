import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class EnhancedRobustCornPipeline:
    """
    Enhanced ML Pipeline with infinity handling and improved feature engineering
    """
    
    def __init__(self, eda_instance, confidence_level=0.95):
        self.eda = eda_instance
        self.df_ml = None
        self.features = None
        self.target = None
        self.imputers = {}
        self.scalers = {}
        self.models = {}
        self.confidence_level = confidence_level
        self.feature_importance = {}
        self.feature_selector = None
        
    def clean_infinity_values(self, df):
        """Clean infinity and extreme values from dataframe"""
        print("🧹 Cleaning infinity and extreme values...")
        
        df_clean = df.copy()
        
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != self.target:
                q99 = df_clean[col].quantile(0.999)
                q01 = df_clean[col].quantile(0.001)
                
                df_clean[col] = df_clean[col].clip(lower=q01, upper=q99)
                
                if df_clean[col].isnull().any() or np.isinf(df_clean[col]).any():
                    print(f"  ⚠️ Still problematic values in {col}, filling with median")
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
        
        return df_clean
    
    def engineer_enhanced_features(self, df):
        """Enhanced feature engineering with better handling of edge cases"""
        print(f"\n🛠️ ENHANCED FEATURE ENGINEERING")
        print("="*40)
        
        df_features = df.copy()
        
        df_features['year'] = df_features.index.year
        df_features['month'] = df_features.index.month
        df_features['quarter'] = df_features.index.quarter
        
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
        df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['quarter'] / 4)
        
        df_features['time_trend'] = range(len(df_features))
        df_features['time_trend_norm'] = df_features['time_trend'] / len(df_features)
        
        if len(df_features) > 12:
            for lag in [1, 2, 3, 6, 12]:
                if len(df_features) > lag:
                    df_features[f'precio_lag_{lag}'] = df_features['precio_maiz'].shift(lag)
            
            for window in [3, 6, 12]:
                if len(df_features) > window:
                    min_periods = max(2, window // 3)
                    
                    df_features[f'precio_ma_{window}'] = (
                        df_features['precio_maiz'].shift(1)
                        .rolling(window, min_periods=min_periods)
                        .mean()
                    )
                    
                    df_features[f'precio_std_{window}'] = (
                        df_features['precio_maiz'].shift(1)
                        .rolling(window, min_periods=min_periods)
                        .std()
                    )

                    rolling_min = (
                        df_features['precio_maiz'].shift(1)
                        .rolling(window, min_periods=min_periods)
                        .min()
                    )
                    rolling_max = (
                        df_features['precio_maiz'].shift(1)
                        .rolling(window, min_periods=min_periods)
                        .max()
                    )
                    
                    range_val = rolling_max - rolling_min
                    range_val = range_val.replace(0, np.nan)
                    
                    df_features[f'precio_position_{window}'] = (
                        (df_features['precio_maiz'] - rolling_min) / range_val
                    ).fillna(0.5)
            
            for period in [1, 3, 12]:
                if len(df_features) > period:
                    pct_change = df_features['precio_maiz'].pct_change(period)
                    pct_change = pct_change.clip(-0.5, 0.5)
                    df_features[f'precio_pct_{period}m'] = pct_change.fillna(0)
            
            if 'precio_ma_3' in df_features.columns:
                ma_3 = df_features['precio_ma_3'].replace(0, np.nan)
                momentum = (df_features['precio_maiz'] / ma_3 - 1).fillna(0)
                df_features['precio_momentum_3m'] = momentum.clip(-0.5, 0.5)
        
        if 'tipo_cambio' in df_features.columns:
            tc_pct = df_features['tipo_cambio'].pct_change(1).fillna(0)
            df_features['tc_pct_1m'] = tc_pct.clip(-0.3, 0.3)
            
            if len(df_features) > 3:
                df_features['tc_ma_3'] = (
                    df_features['tipo_cambio'].shift(1)
                    .rolling(3, min_periods=1)
                    .mean()
                )
            
            tc_safe = df_features['tipo_cambio'].replace(0, np.nan)
            ratio = (df_features['precio_maiz'] / tc_safe).fillna(method='ffill')
            df_features['precio_tc_ratio'] = ratio
            
            if len(df_features) > 3:
                df_features['precio_tc_ratio_ma'] = (
                    ratio.shift(1).rolling(3, min_periods=1).mean()
                )
        
        if 'precio_volatility' in df_features.columns:
            vol_ma = df_features['precio_volatility'].shift(1).rolling(3, min_periods=1).mean()
            df_features['volatility_ma_3'] = vol_ma
            
            vol_ratio = (df_features['precio_volatility'] / vol_ma.replace(0, np.nan)).fillna(1)
            df_features['volatility_ratio'] = vol_ratio.clip(0, 5)
        
        if all(col in df_features.columns for col in ['precio_ma_3', 'precio_ma_12']):
            ma_3_safe = df_features['precio_ma_3'].replace(0, np.nan)
            ma_12_safe = df_features['precio_ma_12'].replace(0, np.nan)
            
            trend_strength = (ma_3_safe / ma_12_safe - 1).fillna(0)
            df_features['trend_strength'] = trend_strength.clip(-0.3, 0.3)
            
            df_features['bull_market'] = (df_features['precio_maiz'] > df_features['precio_ma_12']).astype(int)
        
        df_features['harvest_season'] = df_features['month'].isin([3, 4, 5, 6]).astype(int)
        df_features['planting_season'] = df_features['month'].isin([9, 10, 11, 12]).astype(int)
        
        print(f"✅ Enhanced features: {df_features.shape[1]} columns")
        
        return df_features
    
    def smart_feature_selection(self, X, y, max_features=20):
        """Intelligent feature selection to avoid overfitting"""
        print(f"\n🎯 SMART FEATURE SELECTION")
        print("="*30)
        
        variance_threshold = 0.01
        variances = X.var()
        low_variance_features = variances[variances < variance_threshold].index
        
        if len(low_variance_features) > 0:
            print(f"❌ Removing {len(low_variance_features)} low-variance features")
            X = X.drop(columns=low_variance_features)
        
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > 0.95)
        ]
        
        if len(high_corr_features) > 0:
            print(f"❌ Removing {len(high_corr_features)} highly correlated features")
            X = X.drop(columns=high_corr_features)
        
        if len(X.columns) > max_features:
            print(f"🔍 Selecting top {max_features} features using F-test")
            selector = SelectKBest(score_func=f_regression, k=max_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            X = X[selected_features]
            self.feature_selector = selector
        
        print(f"✅ Final features selected: {len(X.columns)}")
        
        return X
    
    def train_enhanced_models(self, test_size=0.2):
        """Train models with enhanced robustness"""
        print(f"\n🤖 TRAINING ENHANCED MODELS")
        print("="*50)
        
        df = self.df_ml.copy()

        train_size = int(len(df) * (1 - test_size))
        train = df.iloc[:train_size]
        test = df.iloc[train_size:]

        print(f"📈 Training: {len(train)} months")
        print(f"🧪 Testing: {len(test)} months")

        X_train = train[self.features].copy()
        y_train = train[self.target].copy()
        X_test = test[self.features].copy()
        y_test = test[self.target].copy()

        X_train_selected = self.smart_feature_selection(X_train, y_train)
        X_test_selected = X_test[X_train_selected.columns]

        self.features = X_train_selected.columns.tolist()

        X_train_clean = self.clean_infinity_values(X_train_selected)
        X_test_clean = self.clean_infinity_values(X_test_selected)

        self.feature_medians = X_train_clean.median()

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_clean)
        X_test_scaled = scaler.transform(X_test_clean)
        self.scalers['robust'] = scaler
        
        print(f"🔍 Final data verification:")
        print(f"  • Training shape: {X_train_clean.shape}")
        print(f"  • Testing shape: {X_test_clean.shape}")
        print(f"  • Infinity values: {np.isinf(X_train_clean.values).sum()}")
        print(f"  • Missing values: {np.isnan(X_train_clean.values).sum()}")
        
        models_config = {
            'RandomForest': {
                'model': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                ),
                'use_scaling': False
            },
            'HistGradientBoosting': {
                'model': HistGradientBoostingRegressor(
                    max_iter=200,
                    max_depth=6,
                    learning_rate=0.05,
                    min_samples_leaf=10,
                    random_state=42
                ),
                'use_scaling': False
            },
            'Ridge': {
                'model': Ridge(alpha=10.0, random_state=42),
                'use_scaling': True
            },
            'Lasso': {
                'model': Lasso(alpha=1.0, random_state=42, max_iter=2000),
                'use_scaling': True
            },
            'ElasticNet': {
                'model': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000),
                'use_scaling': True
            }
        }
        
        try:
            from xgboost import XGBRegressor
            models_config['XGBoost'] = {
                'model': XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0
                ),
                'use_scaling': False
            }
        except ImportError:
            print("⚠️ XGBoost not available")

        results = {}
        
        for name, config in models_config.items():
            print(f"\n🔄 Training {name}...")
            
            try:
                model = config['model']
                use_scaling = config['use_scaling']
                
                if use_scaling:
                    X_train_model = X_train_scaled
                    X_test_model = X_test_scaled
                else:
                    X_train_model = X_train_clean.values
                    X_test_model = X_test_clean.values
                
                model.fit(X_train_model, y_train.values)
                
                y_pred_train = model.predict(X_train_model)
                y_pred_test = model.predict(X_test_model)
                
                train_mape = np.mean(np.abs((y_train.values - y_pred_train) / y_train.values)) * 100
                test_mape = np.mean(np.abs((y_test.values - y_pred_test) / y_test.values)) * 100
                test_rmse = np.sqrt(mean_squared_error(y_test.values, y_pred_test))
                test_r2 = r2_score(y_test.values, y_pred_test)
                test_mae = mean_absolute_error(y_test.values, y_pred_test)
                
                residuals = y_test.values - y_pred_test
                pred_std = np.std(residuals)
                
                results[name] = {
                    'model': model,
                    'config': config,
                    'train_mape': train_mape,
                    'test_mape': test_mape,
                    'test_rmse': test_rmse,
                    'test_r2': test_r2,
                    'test_mae': test_mae,
                    'predictions_train': y_pred_train,
                    'predictions_test': y_pred_test,
                    'y_train': y_train.values,
                    'y_test': y_test.values,
                    'residuals': residuals,
                    'pred_std': pred_std
                }
                
                print(f"  ✅ {name}:")
                print(f"     • MAPE: {test_mape:.2f}%")
                print(f"     • R²: {test_r2:.3f}")
                print(f"     • RMSE: ${test_rmse:.2f}")
                print(f"     • MAE: ${test_mae:.2f}")
                
                if hasattr(model, 'feature_importances_'):
                    importance = pd.Series(model.feature_importances_, index=self.features)
                    self.feature_importance[name] = importance.sort_values(ascending=False)
                elif hasattr(model, 'coef_'):
                    importance = pd.Series(np.abs(model.coef_), index=self.features)
                    self.feature_importance[name] = importance.sort_values(ascending=False)
                
            except Exception as e:
                print(f"  ❌ {name} failed: {str(e)}")
                continue
        
        if results:
            good_models = {k: v for k, v in results.items() if v['test_r2'] > 0}
            
            if good_models:
                best_model_name = min(good_models.keys(), key=lambda x: good_models[x]['test_mape'])
                print(f"\n🏆 Best model: {best_model_name}")
                print(f"   • MAPE: {results[best_model_name]['test_mape']:.2f}%")
                print(f"   • R²: {results[best_model_name]['test_r2']:.3f}")
            else:
                best_model_name = min(results.keys(), key=lambda x: results[x]['test_mape'])
                print(f"\n⚠️ Best model (by MAPE): {best_model_name}")
                print(f"   • MAPE: {results[best_model_name]['test_mape']:.2f}%")
                print(f"   • R²: {results[best_model_name]['test_r2']:.3f}")
        
        self.models = results
        return results
    
    def create_robust_dataset(self):
        """Create dataset with enhanced robustness"""
        print(f"\n🔧 CREATING ENHANCED ROBUST DATASET")
        print("="*50)
        
        if 'agrofy_maiz' not in self.eda.datasets_procesados:
            raise ValueError("No corn price data available")
        
        df_raw = self.eda.datasets_procesados['agrofy_maiz'].copy()
        
        print("📅 Creating enhanced monthly aggregation...")
        
        price_monthly = df_raw['precio_numerico'].resample('M').agg({
            'mean': 'mean',
            'median': 'median',
            'std': 'std',
            'min': 'min',
            'max': 'max',
            'count': 'count',
            'q25': lambda x: x.quantile(0.25),
            'q75': lambda x: x.quantile(0.75)
        })
        
        df_monthly = pd.DataFrame()
        
        df_monthly['precio_maiz'] = price_monthly['median'].fillna(price_monthly['mean'])
        df_monthly['precio_volatility'] = price_monthly['std'].fillna(0)
        df_monthly['precio_range'] = price_monthly['max'] - price_monthly['min']
        df_monthly['precio_iqr'] = price_monthly['q75'] - price_monthly['q25']
        df_monthly['data_quality'] = price_monthly['count']
        
        df_monthly = df_monthly[df_monthly['data_quality'] >= 1].copy()
        
        if 'tipo_cambio' in self.eda.datasets_procesados:
            print("💱 Adding enhanced exchange rate data...")
            tc_data = self.eda.datasets_procesados['tipo_cambio']
            if 'dolar_principal' in tc_data.columns:
                tc_monthly = tc_data['dolar_principal'].resample('M').agg({
                    'mean': 'mean',
                    'last': 'last',
                    'std': 'std',
                    'min': 'min',
                    'max': 'max'
                })
                
                df_monthly['tipo_cambio'] = tc_monthly['last'].fillna(tc_monthly['mean'])
                df_monthly['tc_volatility'] = tc_monthly['std'].fillna(0)
                df_monthly['tc_range'] = tc_monthly['max'] - tc_monthly['min']
        
        print(f"✅ Enhanced dataset created: {df_monthly.shape}")
        print(f"📅 Date range: {df_monthly.index.min()} to {df_monthly.index.max()}")
        
        return df_monthly
    
    def run_complete_analysis(self):
        """Run the complete enhanced analysis"""
        print("🚀 RUNNING ENHANCED ROBUST ANALYSIS")
        print("="*70)
        
        try:
            df_monthly = self.create_robust_dataset()
            
            df_features = self.engineer_enhanced_features(df_monthly)
            
            self.target = 'precio_maiz'
            exclude_cols = [
                self.target, 
                'data_quality',
                'precio_range', 
                'precio_iqr',
                'tc_range'
            ]
            
            potential_features = [col for col in df_features.columns if col not in exclude_cols]
            
            df_clean = df_features[[self.target] + potential_features].copy()
            df_clean = self.clean_infinity_values(df_clean)
            
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
            df_clean = df_clean.dropna(subset=[self.target])
            
            self.features = [col for col in df_clean.columns if col != self.target]
            self.df_ml = df_clean
            
            print(f"✅ Data preparation complete: {df_clean.shape}")
            print(f"✅ Features: {len(self.features)}")
            
            models = self.train_enhanced_models()
            
            print(f"\n✅ ENHANCED ANALYSIS COMPLETE!")
            
        except Exception as e:
            print(f"❌ Error in enhanced analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
