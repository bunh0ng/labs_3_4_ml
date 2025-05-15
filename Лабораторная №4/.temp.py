import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Загрузка данных
data = pd.read_csv('mnist_small.csv')

# Разделение данных
zeros = data[data['label'] == 0].drop('label', axis=1)
sixes = data[data['label'] == 6].drop('label', axis=1)

# Масштабирование
scaler = StandardScaler()
X_zero = scaler.fit_transform(zeros)
X_six = scaler.transform(sixes)

# One-Class SVM с параметрами, способствующими появлению ошибок
ocsvm = OneClassSVM(nu=0.15, kernel='rbf', gamma=0.01)  # Увеличили nu для большего количества ошибок
ocsvm.fit(X_zero)

# Получаем оценки аномальности
zero_scores = ocsvm.score_samples(X_zero)
six_scores = ocsvm.score_samples(X_six)

# Намеренно выбираем порог, чтобы получить и FP, и FN
# Берем среднее между медианой нулей и медианой шестерок
median_zero = np.median(zero_scores)
median_six = np.median(six_scores)
selected_threshold = (median_zero + median_six) / 2

# Рассчитываем ошибки
fp = np.sum(six_scores > selected_threshold)  # False Positive
fn = np.sum(zero_scores < selected_threshold)  # False Negative

# Искусственно увеличиваем ошибку, если нужно
if fp == 0 or fn == 0:
    # Сдвигаем порог ближе к медиане нулей, чтобы получить больше FP
    selected_threshold = median_zero * 0.9 + median_six * 0.1
    fp = np.sum(six_scores > selected_threshold)
    fn = np.sum(zero_scores < selected_threshold)

# Пересчитываем ошибку
total = len(zero_scores) + len(six_scores)
err = (fp + fn) / total

print(f"Выбранный порог: {selected_threshold:.2f}")
print(f"Общая ошибка (ERR): {err:.4f}")
print(f"False Positive: {fp} ({(fp/len(six_scores)*100):.1f}% шестерок)")
print(f"False Negative: {fn} ({(fn/len(zero_scores)*100):.1f}% нулей)")

# Функция для отображения цифры
def plot_digit(digit, title):
    digit = scaler.inverse_transform(digit.reshape(1, -1)).reshape(28, 28)
    plt.imshow(digit, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')
    plt.colorbar()
    plt.show()

# Находим примеры с наибольшей ошибкой
# False Positive - шестерки с самыми высокими scores
fp_indices = np.argsort(-six_scores)[:3]  # Топ-3 самых "нормальных" шестерок

# False Negative - нули с самыми низкими scores
fn_indices = np.argsort(zero_scores)[:3]  # Топ-3 самых "аномальных" нулей

# True Negative - самые типичные нули
tn_idx = np.argmax(zero_scores)

# True Positive - самые аномальные шестерки
tp_idx = np.argmin(six_scores)

# Отображаем примеры
plot_digit(X_zero[tn_idx], f"True Negative: типичный 0\nScore: {zero_scores[tn_idx]:.2f}")

for i, idx in enumerate(fp_indices):
    plot_digit(X_six[idx], f"False Positive #{i+1}: 6 как 0\nScore: {six_scores[idx]:.2f} > {selected_threshold:.2f}")

for i, idx in enumerate(fn_indices):
    plot_digit(X_zero[idx], f"False Negative #{i+1}: 0 как аномалия\nScore: {zero_scores[idx]:.2f} < {selected_threshold:.2f}")

plot_digit(X_six[tp_idx], f"True Positive: аномальная 6\nScore: {six_scores[tp_idx]:.2f}")

# Дополнительная статистика
print("\nДополнительная статистика:")
print(f"Средний score для нулей: {np.mean(zero_scores):.2f}")
print(f"Средний score для шестерок: {np.mean(six_scores):.2f}")
print(f"Минимальный score для нулей: {np.min(zero_scores):.2f}")
print(f"Максимальный score для шестерок: {np.max(six_scores):.2f}")