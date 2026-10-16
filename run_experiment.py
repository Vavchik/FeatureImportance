#!/usr/bin/env python3
"""
Feature Importance Audit: MDI vs Permutation vs TreeSHAP
Empirical benchmark demonstrating:
1. Cardinality bias of MDI (Mean Decrease in Impurity / Split Count in boosting).
2. Multicollinearity splitting effect.
3. Downstream impact on model performance during automated feature pruning.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fix directories for sandboxed headless runs
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
os.environ["SKB_DATA_DIRECTORY"] = "/tmp/skrub_data"
os.environ["SCIKIT_LEARN_DATA"] = "/tmp/scikit_learn_data"

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import shap

# Visualization setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)


def experiment_1_cardinality_bias():
    print("=== [1/3] Running Experiment 1: Cardinality Bias ===")
    np.random.seed(42)
    n_samples = 3000

    # True informative continuous features
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    
    # Low-cardinality true signal (binary informative flag, e.g. vip client: 50% positive, +2.0 boost)
    x3_binary = (np.random.uniform(0, 1, n_samples) > 0.5).astype(float)
    
    # Noise features:
    # High cardinality categorical / numeric noise (user_id: 1000 unique integers)
    noise_high_card = np.random.randint(0, 1000, n_samples)
    # Continuous noise
    noise_cont = np.random.normal(0, 1, n_samples)
    # Low cardinality binary noise (coin toss)
    noise_binary = np.random.binomial(1, 0.5, n_samples)

    # Ground truth: y depends on x1, x2, x3_binary. NO DEPENDENCE on noise!
    y = 1.5 * x1 + 1.5 * x2 + 2.0 * x3_binary + np.random.normal(0, 1.0, n_samples)

    df = pd.DataFrame({
        'signal_cont_1': x1,
        'signal_cont_2': x2,
        'signal_binary_flag': x3_binary,
        'noise_high_card (user_id)': noise_high_card,
        'noise_continuous': noise_cont,
        'noise_binary': noise_binary
    })

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

    # Standard LightGBM Regressor (default split importance)
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1, importance_type='split')
    model.fit(X_train, y_train)

    # 1. Default MDI (Split Count in LightGBM)
    mdi_importances = model.feature_importances_

    # 2. Permutation Importance on Train (Overfitting trap)
    perm_train = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1)
    perm_train_importances = perm_train.importances_mean

    # 3. Permutation Importance on Test (Out-of-fold validation)
    perm_test = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_test_importances = perm_test.importances_mean

    # 4. TreeSHAP on Test
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_importances = np.mean(np.abs(shap_values), axis=0)

    # Normalize each to percentage of sum for apples-to-apples comparison
    feature_names = df.columns.tolist()
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'MDI (Default rf.feature_importances_)': (mdi_importances / mdi_importances.sum()) * 100,
        'Permutation (Train)': (np.maximum(0, perm_train_importances) / np.maximum(0, perm_train_importances).sum()) * 100,
        'Permutation (Test/OOF)': (np.maximum(0, perm_test_importances) / np.maximum(0, perm_test_importances).sum()) * 100,
        'TreeSHAP (Mean |SHAP| Test)': (shap_importances / shap_importances.sum()) * 100
    })

    print("Experiment 1 Results (% Share):")
    print(results_df.round(2).to_string(index=False))

    # Plot Figure 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Subplot A: Grouped bar chart
    plot_df = results_df.melt(id_vars='Feature', var_name='Method', value_name='Importance (%)')
    palette = {'MDI (Default rf.feature_importances_)': '#e74c3c',
               'Permutation (Train)': '#f39c12',
               'Permutation (Test/OOF)': '#2ecc71',
               'TreeSHAP (Mean |SHAP| Test)': '#3498db'}

    sns.barplot(data=plot_df, y='Feature', x='Importance (%)', hue='Method', palette=palette, ax=axes[0])
    axes[0].set_title('Искажение важности признаков при разной мощности\n(Cardinality Bias в бустинге)', fontsize=11, fontweight='bold', pad=10)
    axes[0].set_xlabel('Нормализованная важность (%)', fontsize=11)
    axes[0].set_ylabel('')
    axes[0].legend(title='Метод расчета', loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    axes[0].axvline(0, color='black', lw=0.8)

    # Subplot B: Summary comparison specifically for noise_high_card
    methods = ['MDI (Split)', 'Permutation (Train)', 'Permutation (Test)', 'TreeSHAP (Test)']
    high_card_vals = [
        results_df.loc[results_df['Feature'] == 'noise_high_card (user_id)', 'MDI (Default rf.feature_importances_)'].values[0],
        results_df.loc[results_df['Feature'] == 'noise_high_card (user_id)', 'Permutation (Train)'].values[0],
        results_df.loc[results_df['Feature'] == 'noise_high_card (user_id)', 'Permutation (Test/OOF)'].values[0],
        results_df.loc[results_df['Feature'] == 'noise_high_card (user_id)', 'TreeSHAP (Mean |SHAP| Test)'].values[0]
    ]
    colors = ['#e74c3c', '#e67e22', '#27ae60', '#2980b9']
    bars = axes[1].bar(methods, high_card_vals, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    axes[1].set_title('Какую долю важности алгоритм\nотдает чистому шуму (user_id)?', fontsize=11, fontweight='bold', pad=10)
    axes[1].set_ylabel('Приписанная важность (%)', fontsize=11)
    max_val = max(high_card_vals)
    axes[1].set_ylim(0, max_val * 1.25 + 0.5)
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    axes[1].axhline(0, color='gray', linestyle='--', lw=1)
    axes[1].text(0.5, 0.88, 'Идеальное значение для шума = 0.00%', transform=axes[1].transAxes,
                 ha='center', fontsize=10, style='italic', color='#1e8449',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#2ecc71", lw=1))

    plt.tight_layout()
    fig1_path = os.path.join(RESULTS_DIR, "fig1_cardinality_bias.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")
    return results_df


def experiment_2_multicollinearity_split():
    print("\n=== [2/3] Running Experiment 2: Multicollinearity Splitting ===")
    np.random.seed(42)
    n_samples = 3000

    # True informative base
    x_base = np.random.normal(0, 1, n_samples)
    # Collinear twin with correlation > 0.999
    x_twin = x_base + np.random.normal(0, 0.01, n_samples)
    
    # Independent informative feature
    x_indep = np.random.normal(0, 1, n_samples)
    # Noise feature
    noise = np.random.normal(0, 1, n_samples)

    # Target depends on x_base and x_indep
    y = 2.5 * x_base + 1.5 * x_indep + np.random.normal(0, 0.5, n_samples)

    # Scenario A: Only x_base, x_indep, noise (No collinearity)
    df_no_collin = pd.DataFrame({'feature_A': x_base, 'feature_B': x_indep, 'noise': noise})
    
    # Scenario B: x_base, x_twin, x_indep, noise (With multicollinearity)
    df_collin = pd.DataFrame({'feature_A': x_base, 'feature_A_twin': x_twin, 'feature_B': x_indep, 'noise': noise})

    m_a = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1, importance_type='split')
    m_a.fit(df_no_collin, y)

    m_b = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1, importance_type='split')
    m_b.fit(df_collin, y)

    # MDI
    mdi_a = m_a.feature_importances_
    mdi_b = m_b.feature_importances_

    # TreeSHAP
    explainer_b = shap.TreeExplainer(m_b)
    shap_b = np.mean(np.abs(explainer_b.shap_values(df_collin)), axis=0)

    # Permutation on B
    perm_b = permutation_importance(m_b, df_collin, y, n_repeats=10, random_state=42, n_jobs=-1).importances_mean

    collin_results = {
        'scenario_single': dict(zip(df_no_collin.columns, (mdi_a / mdi_a.sum()) * 100)),
        'scenario_twins_mdi': dict(zip(df_collin.columns, (mdi_b / mdi_b.sum()) * 100)),
        'scenario_twins_shap': dict(zip(df_collin.columns, (shap_b / shap_b.sum()) * 100)),
        'scenario_twins_perm': dict(zip(df_collin.columns, (np.maximum(0, perm_b) / np.maximum(0, perm_b).sum()) * 100))
    }

    print("Experiment 2 Results (%):")
    for k, v in collin_results.items():
        print(f"  {k}: { {feat: round(val, 2) for feat, val in v.items()} }")

    # Plot Figure 2: The Splitting / Dilution Effect
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Subplot A: Single feature vs Twins in MDI
    categories = ['Без дубликата\n(feature_A)', 'С дубликатом\n(feature_A)', 'С дубликатом\n(feature_A_twin)', 'Сумма близнецов\n(A + A_twin)']
    vals_mdi = [
        collin_results['scenario_single']['feature_A'],
        collin_results['scenario_twins_mdi']['feature_A'],
        collin_results['scenario_twins_mdi']['feature_A_twin'],
        collin_results['scenario_twins_mdi']['feature_A'] + collin_results['scenario_twins_mdi']['feature_A_twin']
    ]
    bar_colors = ['#2980b9', '#e74c3c', '#e67e22', '#27ae60']
    bars_a = axes[0].bar(categories, vals_mdi, color=bar_colors, width=0.55, edgecolor='black', linewidth=0.8)
    axes[0].set_title('Эффект размытия (Splitting) в MDI при сильной корреляции', fontsize=12, fontweight='bold', pad=12)
    axes[0].set_ylabel('MDI Importance (%)', fontsize=11)
    axes[0].set_ylim(0, max(vals_mdi) * 1.25)
    for bar in bars_a:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Subplot B: Method Comparison on Collinear Dataset
    method_labels = ['MDI (Default)', 'Permutation (OOF)', 'TreeSHAP (Test)']
    feat_a = [collin_results['scenario_twins_mdi']['feature_A'], collin_results['scenario_twins_perm']['feature_A'], collin_results['scenario_twins_shap']['feature_A']]
    feat_twin = [collin_results['scenario_twins_mdi']['feature_A_twin'], collin_results['scenario_twins_perm']['feature_A_twin'], collin_results['scenario_twins_shap']['feature_A_twin']]
    feat_b = [collin_results['scenario_twins_mdi']['feature_B'], collin_results['scenario_twins_perm']['feature_B'], collin_results['scenario_twins_shap']['feature_B']]

    x = np.arange(len(method_labels))
    w = 0.25
    axes[1].bar(x - w, feat_a, width=w, label='feature_A (основа)', color='#3498db', edgecolor='black', linewidth=0.8)
    axes[1].bar(x, feat_twin, width=w, label='feature_A_twin (копия)', color='#f39c12', edgecolor='black', linewidth=0.8)
    axes[1].bar(x + w, feat_b, width=w, label='feature_B (независимый)', color='#2ecc71', edgecolor='black', linewidth=0.8)

    axes[1].set_title('Как разные методы интерпретируют коллинеарные признаки', fontsize=12, fontweight='bold', pad=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(method_labels, fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Относительная важность (%)', fontsize=11)
    axes[1].legend(frameon=True, facecolor='white', fontsize=9)
    axes[1].set_ylim(0, max(max(feat_a), max(feat_b)) * 1.3)

    plt.tight_layout()
    fig2_path = os.path.join(RESULTS_DIR, "fig2_multicollinearity_split.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")
    return collin_results


def experiment_3_pruning_performance():
    print("\n=== [3/3] Running Experiment 3: Automated Feature Pruning Pipeline ===")
    np.random.seed(42)
    n_samples = 3000

    # Core continuous features
    f_score = np.random.normal(0, 1, size=n_samples)
    f_income = np.random.normal(0, 1, size=n_samples)
    
    # Crucial low-cardinality binary flag (vip client / risk status: 25% positive, +2.2 impact)
    f_vip = (np.random.uniform(0, 1, size=n_samples) > 0.75).astype(float)
    
    # Collinear clone of income (correlation > 0.99)
    f_income_clone = f_income + np.random.normal(0, 0.02, size=n_samples)

    # 3 Pure noise features:
    # High cardinality pseudo client ID
    f_client_id = np.random.randint(1000, 9999, size=n_samples)
    # Continuous normal noise
    f_noise_cont = np.random.normal(0, 1, size=n_samples)
    # Uniform noise
    f_noise_unif = np.random.uniform(-1, 1, size=n_samples)

    # Ground truth non-linear target
    y = 1.6 * f_income + 1.6 * f_score + 2.2 * f_vip + np.random.normal(0, 0.5, size=n_samples)

    X = pd.DataFrame({
        'credit_score': f_score,
        'income': f_income,
        'vip_flag (binary)': f_vip,
        'income_clone': f_income_clone,
        'noise_uniform': f_noise_unif,
        'noise_continuous': f_noise_cont,
        'client_id (noise)': f_client_id
    })

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Split train into fit and validation for ranking
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Fit reference LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1, importance_type='split')
    lgb_model.fit(X_fit, y_fit)

    # 1. MDI ranking (Split Count)
    mdi_ranks = pd.Series(lgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    # 2. Permutation on Validation Set
    perm_val = permutation_importance(lgb_model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
    perm_ranks = pd.Series(perm_val.importances_mean, index=X.columns).sort_values(ascending=False)

    # 3. TreeSHAP on Validation Set
    explainer = shap.TreeExplainer(lgb_model)
    shap_vals = explainer.shap_values(X_val)
    shap_ranks = pd.Series(np.mean(np.abs(shap_vals), axis=0), index=X.columns).sort_values(ascending=False)

    # Benchmark Pruning Strategy:
    # We prune the bottom K features (from K=0 to K=4) according to each ranking
    # and retrain the final model on full X_train, then evaluate RMSE on held-out X_test!
    num_features_to_drop = list(range(0, 5))
    total_features = len(X.columns)

    rmse_mdi = []
    rmse_perm = []
    rmse_shap = []

    dropped_features_log = {'mdi': {}, 'perm': {}, 'shap': {}}

    for k in num_features_to_drop:
        # Features to keep
        keep_mdi = mdi_ranks.index[:total_features - k].tolist()
        keep_perm = perm_ranks.index[:total_features - k].tolist()
        keep_shap = shap_ranks.index[:total_features - k].tolist()

        dropped_features_log['mdi'][k] = [c for c in X.columns if c not in keep_mdi]
        dropped_features_log['perm'][k] = [c for c in X.columns if c not in keep_perm]
        dropped_features_log['shap'][k] = [c for c in X.columns if c not in keep_shap]

        # Train and evaluate MDI
        m_mdi = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
        m_mdi.fit(X_train[keep_mdi], y_train)
        rmse_mdi.append(np.sqrt(mean_squared_error(y_test, m_mdi.predict(X_test[keep_mdi]))))

        # Train and evaluate Permutation
        m_perm = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
        m_perm.fit(X_train[keep_perm], y_train)
        rmse_perm.append(np.sqrt(mean_squared_error(y_test, m_perm.predict(X_test[keep_perm]))))

        # Train and evaluate TreeSHAP
        m_shap = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
        m_shap.fit(X_train[keep_shap], y_train)
        rmse_shap.append(np.sqrt(mean_squared_error(y_test, m_shap.predict(X_test[keep_shap]))))

    pruning_df = pd.DataFrame({
        'Dropped_Features': num_features_to_drop,
        'Remaining_Features': [total_features - k for k in num_features_to_drop],
        'RMSE_MDI': rmse_mdi,
        'RMSE_Permutation': rmse_perm,
        'RMSE_TreeSHAP': rmse_shap
    })

    print("Feature Pruning Results (Held-Out Test RMSE):")
    print(pruning_df.round(4).to_string(index=False))

    # Plot Figure 3: Pruning Curve & Feature Retention Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Subplot A: RMSE vs Number of Dropped Features
    axes[0].plot(pruning_df['Dropped_Features'], pruning_df['RMSE_MDI'], 'o--', color='#e74c3c', lw=2.4, label='MDI (Дефолтный Split в LightGBM)')
    axes[0].plot(pruning_df['Dropped_Features'], pruning_df['RMSE_Permutation'], 's-', color='#2ecc71', lw=2.2, label='Permutation Importance (Val)')
    axes[0].plot(pruning_df['Dropped_Features'], pruning_df['RMSE_TreeSHAP'], '^-.', color='#3498db', lw=2.2, label='TreeSHAP Magnitude (Val)')

    axes[0].axvline(3, color='gray', linestyle=':', lw=1.2, label='Оптимум (все 3 шума удалены)')
    axes[0].set_title('Качество модели на отложенном тесте при отсечении признаков', fontsize=12, fontweight='bold', pad=12)
    axes[0].set_xlabel('Количество удаленных признаков (наименее важных)', fontsize=11)
    axes[0].set_ylabel('RMSE на валидации (меньше — лучше)', fontsize=11)
    axes[0].legend(frameon=True, facecolor='white', fontsize=9.5)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Subplot B: Rank Comparison for Top vs Noise features
    comparison_ranks = pd.DataFrame({
        'MDI Rank': mdi_ranks.rank(ascending=False).astype(int),
        'Permutation Rank': perm_ranks.rank(ascending=False).astype(int),
        'TreeSHAP Rank': shap_ranks.rank(ascending=False).astype(int)
    })
    
    # Sort by SHAP rank
    comparison_ranks = comparison_ranks.sort_values(by='TreeSHAP Rank')

    axes[1].axis('off')
    table_data = []
    for idx, row in comparison_ranks.iterrows():
        name = idx
        if len(name) > 22:
            name = name[:20] + '..'
        table_data.append([name, row['MDI Rank'], row['Permutation Rank'], row['TreeSHAP Rank']])

    table = axes[1].table(
        cellText=table_data,
        colLabels=['Признак', 'Ранг MDI\n(LightGBM)', 'Ранг Permutation\n(OOF Val)', 'Ранг TreeSHAP\n(OOF Val)'],
        cellLoc='center',
        loc='center',
        colColours=['#eaeded', '#f9e79f', '#d5f5e3', '#d6eaf8']
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1.0, 1.55)
    axes[1].set_title('Сравнительная таблица рангов признаков (1 = самый важный)', fontsize=12, fontweight='bold', pad=12)

    plt.tight_layout()
    fig3_path = os.path.join(RESULTS_DIR, "fig3_pruning_performance.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    return {
        'pruning_df': pruning_df.to_dict(orient='records'),
        'ranks': comparison_ranks.to_dict(orient='index')
    }


if __name__ == '__main__':
    r1 = experiment_1_cardinality_bias()
    r2 = experiment_2_multicollinearity_split()
    r3 = experiment_3_pruning_performance()

    metrics = {
        'experiment_1_cardinality_bias': r1.to_dict(orient='records'),
        'experiment_2_multicollinearity_split': r2,
        'experiment_3_pruning_performance': r3
    }
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nAll experiments completed! Metrics saved to {metrics_path}")
