import pandas as pd

df = pd.read_csv('customer_churn.csv', encoding='utf-8')

# 1. находим все колонки с текстовым типом данных
categorical_cols = df.select_dtypes(include=['object', 'str']).columns

print("Количество скрытых пропусков (unknown) по колонкам:")
for col in categorical_cols:
    # считаем, сколько раз встречается слово unknown
    unknown_count = (df[col] == 'unknown').sum()
    percentage = (unknown_count / len(df)) * 100
    print(f"- {col}: {unknown_count} строк ({percentage:.2f}%)")

print("\nДисбаланас классов в таргете")
print(df['y'].value_counts(normalize=True) * 100)

# создаем копию датасета, чтобы не портить исходник
df_clean = df.copy()

# 1.1 удаляем poutcome, 80% пропусков это борщ
df_clean = df_clean.drop(columns=['poutcome'])

# 1.2 мода, заполняем частыми значениями, где пропусков мало
most_frequent_job = df_clean['job'].mode()[0]
most_frequent_edu = df_clean['education'].mode()[0]

df_clean['job'] = df_clean['job'].replace('unknown', most_frequent_job)
df_clean['education'] = df_clean['education'].replace('unknown', most_frequent_edu)

# 2. переводим нет/да(таргет) из датасета в цифры
df_clean['y'] = df_clean['y'].map({'yes': 1, 'no': 0})

# чекаем, что получилось
print(f"Новый размер после удаления колонки: {df_clean.shape}")
print("\nПроверим пропуски еще раз в одной из колонок:")
print((df_clean['job'] == 'unknown').sum())
print("\nКак теперь выглядит таргет y:")
print(df_clean['y'].value_counts())

# 3. кодирование признаков
# 3.1 бинарное кодирование для бинарных колонок (да/нет)
binary_cols = ['default', 'housing', 'loan']
for col in binary_cols:
    df_clean[col] = df_clean[col].map({'yes': 1, 'no': 0})

# 3.2 находим оставшихся текстовых партизанов 
text_cols = df_clean.select_dtypes(include=['object']).columns

# 3.3 применяем One-Hot Encoding к текстовым колонкам
df_encoded = pd.get_dummies(df_clean, columns=text_cols, drop_first=True)

# на всякий случай пропишем, что хотим видеть 0 и 1, а не True/False
df_encoded = df_encoded.astype(int)

# проверяем результат
print(f"Размер таблицы до кодирования: {df_clean.shape}")
print(f"Размер после: {df_encoded.shape}")
print("\nСписок новый колонок:")
print(df_encoded.columns[:10].tolist())

# 4. Разделение данных и обучение модели
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 4.1 выделяем признаки(Х) и таргет(у)
X = df_encoded.drop(columns=['y']) 
y = df_encoded['y']

# 4.2 делим данные на обучение и тест
# stratify=y гарантирует, что повсюду будет одинаково ушедших клиентов
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4.3 создаем и обучаем модель Случайного леса
# добавили баланса
model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
model.fit(X_train, y_train)

# 4.4 делаем предсказание на тестовых данных
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] #вероятности для расчета ROC-AUC

# смотрим на результаты
print("\nОтчет о качестве модели")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# модель упускает много клиентов из-за дисбаланса классов. добавили штрафы

# 5. важность признаков(из-за чего люди уходят?)
import numpy as np

# 5.1 получаем важность признаков из обученной модели
importance = model.feature_importances_
feature_names = X.columns

# 5.2 сортируем их по убыванию
indices = np.argsort(importance)[::-1]

print("Топ-10 самых важных признаков для модели:")
for i in range(10):
    print(f"{i+1}. {feature_names[indices[i]]}: {importance[indices[i]]:.4f}")