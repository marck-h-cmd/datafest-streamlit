"""
DATAFEST 2026 - Solucion simplificada (solo scikit-learn)
Usar si no puedes instalar lightgbm/xgboost/catboost
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

print("Cargando datos...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature engineering basico
cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
bool_cols = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']

for col in bool_cols:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# Features adicionales
for df in [train, test]:
    df['engagement_score'] = df['activo_movil'] + df['tiene_tarjeta_credito'] + df['tiene_prestamo'] + df['tiene_seguro']
    df['recency_score'] = df['dias_ultima_transaccion'] + df['dias_ultima_interaccion']
    df['capacidad_financiera'] = df['ingresos'] * (1 - df['ratio_deuda_ingresos'])
    df['saldo_sobre_ingresos'] = df['saldo_promedio'] / (df['ingresos'] + 1)

feature_cols = [
    'edad', 'ingresos', 'ratio_deuda_ingresos', 'antiguedad_cuenta_meses',
    'numero_productos', 'saldo_promedio', 'dias_ultima_transaccion',
    'antiguedad_direccion_meses', 'visitas_web_ultimos_90_dias',
    'distancia_sucursal_km', 'dia_preferido_pago', 'dias_ultima_interaccion',
] + cat_cols + bool_cols + [
    'engagement_score', 'recency_score', 'capacidad_financiera', 'saldo_sobre_ingresos'
]

print("Entrenando modelo...")
model = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    random_state=42,
)
model.fit(train[feature_cols], train['objetivo'])

# Validacion temporal
val_mask = train['mes'] == 202611
val_pred = model.predict_proba(train.loc[val_mask, feature_cols])[:, 1]
gini = 2 * roc_auc_score(train.loc[val_mask, 'objetivo'], val_pred) - 1
print(f"Gini validacion (mes 202611): {gini:.4f}")

# Prediccion
submission = pd.DataFrame({
    'id_cliente': test['id_cliente'],
    'prediccion': model.predict_proba(test[feature_cols])[:, 1],
})
submission.to_csv('submission.csv', index=False)
print(f"Guardado: submission.csv ({len(submission)} filas)")
print("Solucion completada!")
