"""
DATAFEST 2026 - Prediccion de Propension de Conversion Bancaria
================================================================
Modelo: Ensemble (LightGBM + XGBoost + CatBoost)
Metrica: Gini = 2 * AUC - 1
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================

print("Cargando datos...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

print(f"  Train: {train.shape}")
print(f"  Test:  {test.shape}")
print(f"  Tasa de conversion: {train['objetivo'].mean():.4f}")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

print("\nGenerando features...")

def create_features(train_df, test_df):
    """Genera features avanzadas usando el panel de datos."""
    
    test_df = test_df.copy()
    test_df['objetivo'] = -1
    combined = pd.concat([train_df, test_df], ignore_index=True)
    
    train_only = train_df.copy()
    
    # Estadisticas historicas por cliente
    client_months = train_only.groupby('id_cliente').agg(
        n_meses_observados=('mes', 'nunique'),
        primer_mes=('mes', 'min'),
        ultimo_mes=('mes', 'max'),
        media_dias_transaccion=('dias_ultima_transaccion', 'mean'),
        media_dias_interaccion=('dias_ultima_interaccion', 'mean'),
        std_dias_transaccion=('dias_ultima_transaccion', 'std'),
        media_saldo=('saldo_promedio', 'mean'),
        std_saldo=('saldo_promedio', 'std'),
        max_productos=('numero_productos', 'max'),
        media_productos=('numero_productos', 'mean'),
        media_ingresos=('ingresos', 'mean'),
        media_ratio_deuda=('ratio_deuda_ingresos', 'mean'),
        media_visitas_web=('visitas_web_ultimos_90_dias', 'mean'),
    ).reset_index()
    
    client_months['std_dias_transaccion'] = client_months['std_dias_transaccion'].fillna(0)
    client_months['std_saldo'] = client_months['std_saldo'].fillna(0)
    
    # Tendencias temporales
    first_obs = train_only.sort_values('mes').groupby('id_cliente').first().reset_index()
    last_obs = train_only.sort_values('mes').groupby('id_cliente').last().reset_index()
    
    trend_df = pd.DataFrame({
        'id_cliente': first_obs['id_cliente'],
        'trend_saldo': last_obs['saldo_promedio'].values - first_obs['saldo_promedio'].values,
        'trend_dias_transaccion': last_obs['dias_ultima_transaccion'].values - first_obs['dias_ultima_transaccion'].values,
        'cambio_productos': last_obs['numero_productos'].values - first_obs['numero_productos'].values,
    })
    
    client_features = client_months.merge(trend_df, on='id_cliente', how='left')
    combined = combined.merge(client_features, on='id_cliente', how='left')
    
    # Rellenar NaN para clientes nuevos
    combined['n_meses_observados'] = combined['n_meses_observados'].fillna(0)
    combined['trend_saldo'] = combined['trend_saldo'].fillna(0)
    combined['trend_dias_transaccion'] = combined['trend_dias_transaccion'].fillna(0)
    combined['cambio_productos'] = combined['cambio_productos'].fillna(0)
    for col in ['media_dias_transaccion', 'media_dias_interaccion', 'std_dias_transaccion',
                'media_saldo', 'std_saldo', 'max_productos', 'media_productos',
                'media_ingresos', 'media_ratio_deuda', 'media_visitas_web']:
        combined[col] = combined[col].fillna(combined[col].median())
    
    # Features de interaccion
    combined['ratio_trans_interac'] = combined['dias_ultima_transaccion'] / (combined['dias_ultima_interaccion'] + 1)
    combined['productos_x_activo_movil'] = combined['numero_productos'] * combined['activo_movil'].astype(int)
    combined['ingresos_x_productos'] = combined['ingresos'] * combined['numero_productos']
    combined['saldo_sobre_ingresos'] = combined['saldo_promedio'] / (combined['ingresos'] + 1)
    combined['deuda_total_est'] = combined['ingresos'] * combined['ratio_deuda_ingresos']
    combined['capacidad_financiera'] = combined['ingresos'] * (1 - combined['ratio_deuda_ingresos'])
    
    combined['engagement_score'] = (
        combined['activo_movil'].astype(int) * 2 +
        combined['tiene_tarjeta_credito'].astype(int) +
        combined['tiene_prestamo'].astype(int) +
        combined['tiene_seguro'].astype(int) +
        (combined['visitas_web_ultimos_90_dias'] > combined['visitas_web_ultimos_90_dias'].median()).astype(int)
    )
    
    combined['recency_score'] = combined['dias_ultima_transaccion'] + combined['dias_ultima_interaccion']
    combined['diff_trans_interac'] = combined['dias_ultima_transaccion'] - combined['dias_ultima_interaccion']
    combined['ratio_antiguedad'] = combined['antiguedad_cuenta_meses'] / (combined['antiguedad_direccion_meses'] + 1)
    combined['total_productos_bool'] = (
        combined['tiene_tarjeta_credito'].astype(int) +
        combined['tiene_prestamo'].astype(int) +
        combined['tiene_seguro'].astype(int)
    )
    combined['es_cliente_nuevo_real'] = (combined['n_meses_observados'] == 0).astype(int)
    combined['saldo_vs_media'] = combined['saldo_promedio'] - combined['media_saldo']
    combined['dias_trans_vs_media'] = combined['dias_ultima_transaccion'] - combined['media_dias_transaccion']
    
    # Encoding de categoricas
    cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
    
    for col in cat_cols:
        le = LabelEncoder()
        combined[col + '_encoded'] = le.fit_transform(combined[col])
    
    # Target encoding con smoothing
    train_mask = combined['objetivo'] != -1
    for col in cat_cols:
        global_mean = combined.loc[train_mask, 'objetivo'].mean()
        smooth = 10
        agg = combined.loc[train_mask].groupby(col)['objetivo'].agg(['mean', 'count'])
        agg['te'] = (agg['count'] * agg['mean'] + smooth * global_mean) / (agg['count'] + smooth)
        combined[col + '_te'] = combined[col].map(agg['te'])
    
    train_feat = combined[combined['objetivo'] != -1].copy()
    test_feat = combined[combined['objetivo'] == -1].copy()
    
    return train_feat, test_feat

train_feat, test_feat = create_features(train, test)

# ============================================================================
# 3. DEFINICION DE FEATURES
# ============================================================================

feature_cols = [
    'edad', 'ingresos', 'ratio_deuda_ingresos', 'antiguedad_cuenta_meses',
    'numero_productos', 'saldo_promedio', 'dias_ultima_transaccion',
    'antiguedad_direccion_meses', 'visitas_web_ultimos_90_dias',
    'distancia_sucursal_km', 'dia_preferido_pago', 'dias_ultima_interaccion',
    'tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente',
    'tiene_prestamo', 'tiene_seguro',
    'ocupacion_encoded', 'region_encoded', 'canal_adquisicion_encoded',
    'banda_riesgo_encoded', 'dispositivo_principal_encoded',
    'ocupacion_te', 'region_te', 'canal_adquisicion_te',
    'banda_riesgo_te', 'dispositivo_principal_te',
    'n_meses_observados', 'media_dias_transaccion', 'media_dias_interaccion',
    'std_dias_transaccion', 'media_saldo', 'std_saldo',
    'max_productos', 'media_productos', 'media_ingresos',
    'media_ratio_deuda', 'media_visitas_web',
    'trend_saldo', 'trend_dias_transaccion', 'cambio_productos',
    'ratio_trans_interac', 'productos_x_activo_movil',
    'ingresos_x_productos', 'saldo_sobre_ingresos',
    'deuda_total_est', 'capacidad_financiera',
    'engagement_score', 'recency_score', 'diff_trans_interac',
    'ratio_antiguedad', 'total_productos_bool',
    'es_cliente_nuevo_real', 'saldo_vs_media', 'dias_trans_vs_media',
]

for col in ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente',
            'tiene_prestamo', 'tiene_seguro']:
    train_feat[col] = train_feat[col].astype(int)
    test_feat[col] = test_feat[col].astype(int)

X = train_feat[feature_cols].values
y = train_feat['objetivo'].values.astype(int)
X_test = test_feat[feature_cols].values

print(f"  Features generadas: {len(feature_cols)}")

# ============================================================================
# 4. ENTRENAMIENTO
# ============================================================================

print("\nEntrenando modelos...")

# Validacion temporal
train_temporal = train_feat[train_feat['mes'] <= 202610]
val_temporal = train_feat[train_feat['mes'] == 202611]
X_train_t = train_temporal[feature_cols].values
y_train_t = train_temporal['objetivo'].values.astype(int)
X_val_t = val_temporal[feature_cols].values
y_val_t = val_temporal['objetivo'].values.astype(int)

print(f"  Train: {X_train_t.shape[0]} obs | Val: {X_val_t.shape[0]} obs")

HAS_LGB = False
HAS_XGB = False
HAS_CB = False

# LightGBM
try:
    import lightgbm as lgb
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'n_estimators': 1500, 'learning_rate': 0.03, 'max_depth': 7,
        'num_leaves': 63, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_weight': 5,
        'min_child_samples': 20,
        'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
        'random_state': 42, 'verbose': -1, 'n_jobs': -1,
    }
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train_t, y_train_t, eval_set=[(X_val_t, y_val_t)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    lgb_val_pred = lgb_model.predict_proba(X_val_t)[:, 1]
    lgb_auc = roc_auc_score(y_val_t, lgb_val_pred)
    lgb_gini = 2 * lgb_auc - 1
    print(f"  LightGBM - AUC: {lgb_auc:.6f} | Gini: {lgb_gini:.6f}")
    lgb_full = lgb.LGBMClassifier(**{**lgb_params, 'n_estimators': lgb_model.best_iteration_})
    lgb_full.fit(X, y)
    lgb_test_pred = lgb_full.predict_proba(X_test)[:, 1]
    HAS_LGB = True
except ImportError:
    print("  LightGBM no disponible")

# XGBoost
try:
    import xgboost as xgb
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'n_estimators': 1500, 'learning_rate': 0.03, 'max_depth': 7,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_weight': 5,
        'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
        'random_state': 42, 'n_jobs': -1, 'verbosity': 0, 'tree_method': 'hist',
    }
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train_t, y_train_t, eval_set=[(X_val_t, y_val_t)], verbose=False)
    xgb_val_pred = xgb_model.predict_proba(X_val_t)[:, 1]
    xgb_auc = roc_auc_score(y_val_t, xgb_val_pred)
    xgb_gini = 2 * xgb_auc - 1
    print(f"  XGBoost  - AUC: {xgb_auc:.6f} | Gini: {xgb_gini:.6f}")
    xgb_full = xgb.XGBClassifier(**{**xgb_params, 'n_estimators': xgb_model.best_iteration})
    xgb_full.fit(X, y, verbose=False)
    xgb_test_pred = xgb_full.predict_proba(X_test)[:, 1]
    HAS_XGB = True
except ImportError:
    print("  XGBoost no disponible")

# CatBoost
try:
    from catboost import CatBoostClassifier
    cb_model = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=7, l2_leaf_reg=3.0,
        subsample=0.8, auto_class_weights='Balanced', eval_metric='AUC',
        random_seed=42, verbose=0, early_stopping_rounds=100,
    )
    cb_model.fit(X_train_t, y_train_t, eval_set=(X_val_t, y_val_t))
    cb_val_pred = cb_model.predict_proba(X_val_t)[:, 1]
    cb_auc = roc_auc_score(y_val_t, cb_val_pred)
    cb_gini = 2 * cb_auc - 1
    print(f"  CatBoost - AUC: {cb_auc:.6f} | Gini: {cb_gini:.6f}")
    cb_full = CatBoostClassifier(
        iterations=cb_model.best_iteration_, learning_rate=0.03, depth=7,
        l2_leaf_reg=3.0, subsample=0.8, auto_class_weights='Balanced',
        random_seed=42, verbose=0,
    )
    cb_full.fit(X, y)
    cb_test_pred = cb_full.predict_proba(X_test)[:, 1]
    HAS_CB = True
except ImportError:
    print("  CatBoost no disponible")

# ============================================================================
# 5. ENSEMBLE
# ============================================================================

print("\nGenerando ensemble...")
models_preds, models_weights = {}, {}
if HAS_LGB:
    models_preds['lgb'] = lgb_test_pred
    models_weights['lgb'] = lgb_gini
if HAS_XGB:
    models_preds['xgb'] = xgb_test_pred
    models_weights['xgb'] = xgb_gini
if HAS_CB:
    models_preds['cb'] = cb_test_pred
    models_weights['cb'] = cb_gini

if len(models_preds) == 0:
    raise RuntimeError("Ningun modelo disponible. Instala al menos: pip install lightgbm xgboost")

total_weight = sum(models_weights.values())
final_pred = np.zeros(X_test.shape[0])
for name, pred in models_preds.items():
    w = models_weights[name] / total_weight
    final_pred += w * pred
    print(f"  {name}: peso = {w:.4f}")

print(f"\n  Prediccion media: {final_pred.mean():.4f}")
print(f"  Prediccion min:   {final_pred.min():.4f}")
print(f"  Prediccion max:   {final_pred.max():.4f}")

# ============================================================================
# 6. GENERAR SUBMISSION
# ============================================================================

submission = pd.DataFrame({
    'id_cliente': test_feat['id_cliente'].values,
    'prediccion': final_pred,
})

# Validaciones
assert submission.shape[0] == sample_sub.shape[0], f"Filas: {submission.shape[0]} vs {sample_sub.shape[0]}"
assert list(submission.columns) == list(sample_sub.columns), "Columnas no coinciden"
assert submission['prediccion'].between(0, 1).all(), "Predicciones fuera de [0,1]"

submission.to_csv('submission.csv', index=False)
print(f"\nGuardado: submission.csv ({submission.shape[0]} filas)")

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================

if HAS_LGB:
    print("\nTop 15 features mas importantes (LightGBM):")
    imp = pd.DataFrame({'feature': feature_cols, 'importance': lgb_full.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    for _, row in imp.head(15).iterrows():
        print(f"  {row['importance']:6.0f} | {row['feature']}")

print("\n=== Solucion completada exitosamente! ===")
