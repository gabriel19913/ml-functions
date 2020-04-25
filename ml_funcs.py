import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def convert_numbers(valor):
    return float('.'.join(valor.split(',')))

def verificar_correlação_saida(var, out):
    return np.corrcoef([df[var], df[out]])

def standarize_data(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X =  pd.DataFrame(X)
    return X

def pca(X, n_components, plot = False):
    pca = PCA(n_components = n_components)
    pca.fit(X)
    variance = pca.explained_variance_ratio_
    var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 7) * 100)
    if plot:
        fig, ax1 = plt.subplots(ncols = 1, figsize=(10, 8))
        ax1.set(xlabel = 'Número de Componentes', ylabel= 'Variância Explicada (%)', 
                title = 'Análise do PCA', ylim = [68,101], xticks = list(range(1, n_components + 1)))
        sns.lineplot(x = list(range(1, n_components + 1)), y = var, ax = ax1)
        plt.show()
    print(f'Percentual de variância explicada por cada componente:\n{np.round(var, 3)}')
    keep = int(input('Quantas componentes você gostaria de manter? '))
    pca1 = PCA(n_components = keep)
    pca1.fit(X)
    X = pca1.transform(X)
    print('=-=' * 24)
    print(f'Percentual de variância explicada por cada componente:\n{np.round(pca1.explained_variance_ratio_ * 100, 3)}')
    print(f'Percentual total de variância explicada pelas componente: {np.round(pca1.explained_variance_ratio_ * 100, 3).sum():.3f}%')
    return pd.DataFrame(X)

def high_corr(X_cor, threshold = 0.7, annot = True, size = (16,16), font_scale = 1, cmap = "RdBu_r"):
    high_cor = X_cor[X_cor >= threshold]
    mask = np.zeros_like(high_cor, dtype = np.bool)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize = size)
    sns.set(font_scale = font_scale)
    sns.heatmap(high_cor,cmap = cmap, annot = annot, linecolor = 'black', vmin = -1, vmax = 1, mask = mask, ax = ax, linewidths = .5)
    plt.tight_layout()
    plt.show()

def correlation(X, annot = False, save = False, size = (16,16), font_scale = 1, cmap = "RdBu_r", corr_table = False):
    X_cor = X.corr()
    X_cor = X_cor.round(2)
    mask = np.zeros_like(X_cor, dtype = np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.set(font_scale = font_scale)
    fig, ax = plt.subplots(figsize = size)
    sns.heatmap(X_cor, cmap = cmap, annot = annot, linecolor = 'white', vmin = -1, vmax = 1, mask = mask, ax = ax)
    plt.tight_layout()
    if save:
        plt.savefig('correlation.tif', dpi=300)
    else:
        plt.show()
    if corr_table:
        return X_cor

def cm_analysis(y_true, y_pred, filename, cmap = 'Blues', percent = True, font = 10, figsize = (10,10)):
    sns.set_context("paper")
    sns.set(font_scale = 2)
    data = {'True':    y_true,
            'Predicted': y_pred}
    df = pd.DataFrame(data, columns=['True','Predicted'])
    confusion_matrix = pd.crosstab(df['True'], df['Predicted'], rownames = ['True'], colnames = ['Predicted'], margins = True)
    if percent:
        all_values = confusion_matrix['All']['All']
        confusion_matrix = np.around((confusion_matrix / all_values) * 100, 2)
    fig, ax = plt.subplots(figsize = figsize)
    ax.set_yticklabels(confusion_matrix.index, rotation = 0, fontsize = "24", va = "center")
    ax.set_xticklabels(confusion_matrix.columns, rotation = 0, fontsize = "24", va = "center")
    sns.heatmap(confusion_matrix, annot = True, fmt = 'g', annot_kws = {"size": font}, cmap = cmap)
    plt.savefig(filename, dpi = 300)

def print_score(scores, metric):
    print(f'\n{metric.title()} in each fold:')
    for i in range(len(scores)):
        print(f'Fold {i+1:>2}: {100 * scores[i]:.2f}%')
    print(f'\nMean {metric} of all folds: {100* np.mean(scores):.2f}%')
    print(f'\nStandard deviation {metric} of all folds: {100 * np.std(scores):.2f}%')

def cross_validation_model_class(X, y, model, model_name, data_name, seed, save = False, print_scores = False):
    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)
    scores = []
    predicoes = []
    past_score = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predicoes.append(y_pred)
        score = accuracy_score(y_pred, y_test)
        scores.append(score)
        if score > past_score:
            print(f'The current accuracy is {100 * score:.2f}%, the previous accuracy was {100 * past_score:.2f}%.')
            filename = f'{model_name}_{data_name}_model.pkl'
            if save:
                with open(model_path + filename,'wb') as f:
                    pickle.dump(model, f)
            past_score = score
    if print_scores:
        print_score(scores, 'accuracy')
    return scores

def cross_validation_model_regr(X, y, model, model_name, data_name, seed, save = False, print_scores = False):
    kf = KFold(n_splits = 10, shuffle = True, random_state = seed)
    scores = []
    erros = []
    predicoes = []
    past_score = 0
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predicoes.append(y_pred)
        score = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(score)
        erros.append(rmse)
        if score > past_score:
            print(f'The current R2 is {score:.3f}, the past R2 was {past_score:.3f}.')
            filename = f'{model_name}_{data_name}_model.pkl'
            if save:
                with open(model_path + filename,'wb') as f:
                    pickle.dump(model, f)
            past_score = score
    if print_scores:
        print_score(scores, 'r2')
        print_score(erros, 'rmse')
    return scores, erros