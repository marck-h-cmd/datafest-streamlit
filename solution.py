"""
DATAFEST 2026 - Prediccion de Propension de Conversion Bancaria
================================================================
Modelo: Ensemble Optimizado (CatBoost + XGBoost + LightGBM)
Metrica: Gini = 2 * AUC - 1
Sin Data Leakage temporal + Features financieras + Rank Averaging
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================

print("=" * 60)
print("1. CARGANDO DATOS")
print("=" * 60)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

print(f"  Train: {train.shape[0]:,} filas, {train.shape[1]} columnas")
print(f"  Test:  {test.shape[0]:,} filas, {test.shape[1]} columnas")
print(f"  Meses Train: {train['mes'].min()} a {train['mes'].max()}")
print(f"  Mes Test:    {test['mes'].unique()[0]}")
print(f"  Tasa global de conversion: {train['objetivo'].mean():.4f}")

# ============================================================================
# 2. FEATURE ENGINEERING (ESTRICTAMENTE TEMPORAL, SIN LOOKAHEAD LEAKAGE)
# ============================================================================

print("\n" + "=" * 60)
print("2. GENERANDO FEATURES LIMPIAS (SIN FUGA TEMPORAL)")
print("=" * 60)

def build_features(train_df, test_df):
    train_c = train_df.copy()
    test_c = test_df.copy()
    
    train_c['is_test'] = 0
    test_c['is_test'] = 1
    test_c['objetivo'] = np.nan
    
    combined = pd.concat([train_c, test_c], ignore_index=True)
    
    # Orden cronologico por cliente
    combined = combined.sort_values(['id_cliente', 'mes']).reset_index(drop=True)
    
    # 1. Antigüedad en observaciones pasadas (estrictamente acumulada hasta el mes actual)
    combined['meses_en_historial'] = combined.groupby('id_cliente').cumcount()
    combined['es_cliente_recurrente'] = (combined['meses_en_historial'] > 0).astype(int)
    
    # 2. Ratios y variables financieras de alto valor predictivo
    combined['deuda_estimada'] = combined['ingresos'] * combined['ratio_deuda_ingresos']
    combined['capacidad_financiera'] = combined['ingresos'] * (1 - combined['ratio_deuda_ingresos'])
    combined['saldo_sobre_ingresos'] = combined['saldo_promedio'] / (combined['ingresos'] + 1.0)
    combined['saldo_por_producto'] = combined['saldo_promedio'] / (combined['numero_productos'] + 1.0)
    combined['saldo_sobre_deuda'] = combined['saldo_promedio'] / (combined['deuda_estimada'] + 1.0)
    
    # 3. Interacciones y recencia
    combined['ratio_trans_interac'] = combined['dias_ultima_transaccion'] / (combined['dias_ultima_interaccion'] + 1.0)
    combined['diff_trans_interac'] = combined['dias_ultima_transaccion'] - combined['dias_ultima_interaccion']
    combined['visitas_por_mes_cuenta'] = combined['visitas_web_ultimos_90_dias'] / (combined['antiguedad_cuenta_meses'] + 1.0)
    combined['ratio_antiguedad'] = combined['antiguedad_cuenta_meses'] / (combined['antiguedad_direccion_meses'] + 1.0)
    
    # 4. Flags booleanas e indices de tenencia de productos
    bool_cols = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']
    for c in bool_cols:
        combined[c] = combined[c].astype(int)
        
    combined['total_productos_bool'] = (
        combined['tiene_tarjeta_credito'] +
        combined['activo_movil'] +
        combined['tiene_prestamo'] +
        combined['tiene_seguro']
    )
    combined['productos_totales_banco'] = combined['numero_productos'] + combined['total_productos_bool']
    
    combined['engagement_score'] = (
        combined['activo_movil'] * 2 +
        combined['tiene_tarjeta_credito'] +
        combined['tiene_prestamo'] +
        combined['tiene_seguro'] +
        (combined['visitas_web_ultimos_90_dias'] > 5).astype(int)
    )
    
    # 5. Mapeo ordinal de riesgo
    risk_map = {'low': 2, 'medium': 1, 'high': 0}
    combined['banda_riesgo_num'] = combined['banda_riesgo'].map(risk_map).fillna(1).astype(int)
    
    # Separar train y test manteniendo el orden original
    train_res = combined[combined['is_test'] == 0].copy().sort_values(['mes', 'id_cliente']).reset_index(drop=True)
    test_res = combined[combined['is_test'] == 1].copy().sort_values('id_cliente').reset_index(drop=True)
    
    return train_res, test_res

train_feat, test_feat = build_features(train, test)

cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
ignore_cols = ['id_cliente', 'mes', 'objetivo', 'is_test']
feature_cols = [c for c in train_feat.columns if c not in ignore_cols]

print(f"  Total de features generadas: {len(feature_cols)}")
print(f"  Features: {', '.join(feature_cols[:8])}...")

# ============================================================================
# 3. VALIDACION TEMPORAL (Train <= 202610, Val == 202611)
# ============================================================================

print("\n" + "=" * 60)
print("3. VALIDACION TEMPORAL (Train: Ene-Oct 2026 | Val: Nov 2026)")
print("=" * 60)

train_t = train_feat[train_feat['mes'] <= 202610].copy()
val_t = train_feat[train_feat['mes'] == 202611].copy()

X_train_t = train_t[feature_cols].copy()
y_train_t = train_t['objetivo'].astype(int).values
X_val_t = val_t[feature_cols].copy()
y_val_t = val_t['objetivo'].astype(int).values

print(f"  Train split: {len(y_train_t):,} filas (Tasa conv: {y_train_t.mean():.4f})")
print(f"  Val split:   {len(y_val_t):,} filas (Tasa conv: {y_val_t.mean():.4f})")

# Preparacion de datos por tipo de modelo
# 1) CatBoost: columnas categoricas como string
X_tr_cb = X_train_t.copy()
X_va_cb = X_val_t.copy()
for c in cat_cols:
    X_tr_cb[c] = X_tr_cb[c].astype(str)
    X_va_cb[c] = X_va_cb[c].astype(str)

# 2) LightGBM: columnas categoricas como tipo category
X_tr_lgb = X_train_t.copy()
X_va_lgb = X_val_t.copy()
for c in cat_cols:
    X_tr_lgb[c] = X_tr_lgb[c].astype('category')
    X_va_lgb[c] = X_va_lgb[c].astype('category')

# 3) XGBoost: one-hot encoding
X_tr_xgb = pd.get_dummies(X_train_t, columns=cat_cols)
X_va_xgb = pd.get_dummies(X_val_t, columns=cat_cols)
X_va_xgb = X_va_xgb.reindex(columns=X_tr_xgb.columns, fill_value=0)

models_val_preds = {}
models_best_iter = {}

# --- A. CATBOOST ---
from catboost import CatBoostClassifier
print("\n[1/3] Entrenando CatBoost...")
cb_val = CatBoostClassifier(
    iterations=1200,
    learning_rate=0.035,
    depth=6,
    l2_leaf_reg=4.0,
    cat_features=cat_cols,
    auto_class_weights='Balanced',
    eval_metric='AUC',
    random_seed=42,
    verbose=0,
    early_stopping_rounds=100
)
cb_val.fit(X_tr_cb, y_train_t, eval_set=(X_va_cb, y_val_t))
cb_val_pred = cb_val.predict_proba(X_va_cb)[:, 1]
cb_auc = roc_auc_score(y_val_t, cb_val_pred)
cb_gini = 2 * cb_auc - 1
best_iter_cb = cb_val.get_best_iteration()
models_val_preds['CatBoost'] = cb_val_pred
models_best_iter['CatBoost'] = best_iter_cb
print(f"  -> CatBoost  | AUC: {cb_auc:.5f} | Gini: {cb_gini:.5f} (Best iter: {best_iter_cb})")

# --- B. XGBOOST ---
from xgboost import XGBClassifier
print("\n[2/3] Entrenando XGBoost...")
xgb_val = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.025,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.75,
    reg_alpha=0.5,
    reg_lambda=2.0,
    tree_method='hist',
    eval_metric='auc',
    random_state=42,
    early_stopping_rounds=100
)
xgb_val.fit(X_tr_xgb, y_train_t, eval_set=[(X_va_xgb, y_val_t)], verbose=False)
xgb_val_pred = xgb_val.predict_proba(X_va_xgb)[:, 1]
xgb_auc = roc_auc_score(y_val_t, xgb_val_pred)
xgb_gini = 2 * xgb_auc - 1
best_iter_xgb = xgb_val.best_iteration
models_val_preds['XGBoost'] = xgb_val_pred
models_best_iter['XGBoost'] = best_iter_xgb
print(f"  -> XGBoost   | AUC: {xgb_auc:.5f} | Gini: {xgb_gini:.5f} (Best iter: {best_iter_xgb})")

# --- C. LIGHTGBM ---
from lightgbm import LGBMClassifier
print("\n[3/3] Entrenando LightGBM...")
lgb_val = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.025,
    num_leaves=28,
    max_depth=5,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.75,
    reg_alpha=0.2,
    reg_lambda=1.5,
    random_state=42,
    verbose=-1
)
lgb_val.fit(X_tr_lgb, y_train_t)
lgb_val_pred = lgb_val.predict_proba(X_va_lgb)[:, 1]
lgb_auc = roc_auc_score(y_val_t, lgb_val_pred)
lgb_gini = 2 * lgb_auc - 1
models_val_preds['LightGBM'] = lgb_val_pred
models_best_iter['LightGBM'] = 600
print(f"  -> LightGBM  | AUC: {lgb_auc:.5f} | Gini: {lgb_gini:.5f}")

# --- ENSEMBLE EN VALIDACION ---
print("\n" + "-" * 50)
print("EVALUACION DEL ENSEMBLE EN VALIDACION:")
print("-" * 50)
# Ranking percentil normalizado
r_cb = rankdata(cb_val_pred) / len(cb_val_pred)
r_xgb = rankdata(xgb_val_pred) / len(xgb_val_pred)
r_lgb = rankdata(lgb_val_pred) / len(lgb_val_pred)

# Ponderacion: 45% CatBoost, 35% XGBoost, 20% LightGBM
val_ensemble_rank = 0.45 * r_cb + 0.35 * r_xgb + 0.20 * r_lgb
ens_auc = roc_auc_score(y_val_t, val_ensemble_rank)
ens_gini = 2 * ens_auc - 1
print(f"  [+] ENSEMBLE RANK AVERAGING | AUC: {ens_auc:.5f} | Gini: {ens_gini:.5f}")

# ============================================================================
# 4. RE-ENTRENAMIENTO COMPLETO CON LOS 11 MESES DE TRAIN
# ============================================================================

print("\n" + "=" * 60)
print("4. ENTRENAMIENTO FINAL (TODO TRAIN: 110,100 REGISTROS)")
print("=" * 60)

X_train_full = train_feat[feature_cols].copy()
y_train_full = train_feat['objetivo'].astype(int).values
X_test_full = test_feat[feature_cols].copy()

# A. CatBoost Full
print("  Entrenando CatBoost Full...")
X_full_cb = X_train_full.copy()
X_test_cb = X_test_full.copy()
for c in cat_cols:
    X_full_cb[c] = X_full_cb[c].astype(str)
    X_test_cb[c] = X_test_cb[c].astype(str)

cb_full = CatBoostClassifier(
    iterations=int(best_iter_cb * 1.15) if best_iter_cb else 700,
    learning_rate=0.035,
    depth=6,
    l2_leaf_reg=4.0,
    cat_features=cat_cols,
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=0
)
cb_full.fit(X_full_cb, y_train_full)
test_pred_cb = cb_full.predict_proba(X_test_cb)[:, 1]

# B. XGBoost Full
print("  Entrenando XGBoost Full...")
X_full_xgb = pd.get_dummies(X_train_full, columns=cat_cols)
X_test_xgb = pd.get_dummies(X_test_full, columns=cat_cols)
X_test_xgb = X_test_xgb.reindex(columns=X_full_xgb.columns, fill_value=0)

xgb_full = XGBClassifier(
    n_estimators=max(int(best_iter_xgb * 1.15), 350) if best_iter_xgb else 450,
    learning_rate=0.025,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.75,
    reg_alpha=0.5,
    reg_lambda=2.0,
    tree_method='hist',
    random_state=42
)
xgb_full.fit(X_full_xgb, y_train_full, verbose=False)
test_pred_xgb = xgb_full.predict_proba(X_test_xgb)[:, 1]

# C. LightGBM Full
print("  Entrenando LightGBM Full...")
X_full_lgb = X_train_full.copy()
X_test_lgb = X_test_full.copy()
for c in cat_cols:
    X_full_lgb[c] = X_full_lgb[c].astype('category')
    X_test_lgb[c] = X_test_lgb[c].astype('category')

lgb_full = LGBMClassifier(
    n_estimators=700,
    learning_rate=0.025,
    num_leaves=28,
    max_depth=5,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.75,
    reg_alpha=0.2,
    reg_lambda=1.5,
    random_state=42,
    verbose=-1
)
lgb_full.fit(X_full_lgb, y_train_full)
test_pred_lgb = lgb_full.predict_proba(X_test_lgb)[:, 1]

# ============================================================================
# 5. ENSEMBLE FINAL Y GENERACION DE SUBMISSION
# ============================================================================

print("\n" + "=" * 60)
print("5. GENERANDO SUBMISSION FINAL")
print("=" * 60)

# Rank Averaging en Test
rank_test_cb = rankdata(test_pred_cb) / len(test_pred_cb)
rank_test_xgb = rankdata(test_pred_xgb) / len(test_pred_xgb)
rank_test_lgb = rankdata(test_pred_lgb) / len(test_pred_lgb)

final_ranks = 0.45 * rank_test_cb + 0.35 * rank_test_xgb + 0.20 * rank_test_lgb

# Escalado a probabilidad calibrada [0.01, 0.99]
pred_calibrada = (final_ranks - final_ranks.min()) / (final_ranks.max() - final_ranks.min())
pred_calibrada = 0.01 + 0.98 * pred_calibrada

# DataFrame de submission
sub_df = pd.DataFrame({
    'id_cliente': test_feat['id_cliente'].values,
    'prediccion': pred_calibrada
})

# Asegurar orden exacto de sample_submission
sub_df = sample_sub[['id_cliente']].merge(sub_df, on='id_cliente', how='left')

# Verificaciones de calidad requeridas por la competencia
assert sub_df.shape[0] == sample_sub.shape[0], f"Error: {sub_df.shape[0]} != {sample_sub.shape[0]}"
assert list(sub_df.columns) == ['id_cliente', 'prediccion'], f"Error columnas: {sub_df.columns}"
assert not sub_df['prediccion'].isna().any(), "Error: Existen valores NaN en la prediccion"
assert sub_df['prediccion'].between(0, 1).all(), "Error: Predicciones fuera del rango [0, 1]"
assert (sub_df['id_cliente'].values == sample_sub['id_cliente'].values).all(), "Error: Desalineacion de IDs"

sub_df.to_csv('submission.csv', index=False)

print(f"  Submission guardada exitosamente en 'submission.csv'")
print(f"  Filas: {len(sub_df):,}")
print(f"  Distribucion predicciones:")
print(f"    Min:    {sub_df['prediccion'].min():.4f}")
print(f"    Media:  {sub_df['prediccion'].mean():.4f}")
print(f"    Mediana:{sub_df['prediccion'].median():.4f}")
print(f"    Max:    {sub_df['prediccion'].max():.4f}")

# ============================================================================
# 6. IMPORTANCIA DE VARIABLES (TOP 15)
# ============================================================================

print("\n" + "=" * 60)
print("6. TOP 15 VARIABLES MAS IMPORTANTES (CATBOOST)")
print("=" * 60)

imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': cb_full.get_feature_importance()
}).sort_values('importance', ascending=False)

for rank, (_, row) in enumerate(imp_df.head(15).iterrows(), 1):
    print(f"  {rank:2d}. {row['feature']:<30} | {row['importance']:.2f}%")

print("\n" + "=" * 60)
print("PROCESO COMPLETADO EXITOSAMENTE!")
print("=" * 60)
