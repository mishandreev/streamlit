import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from typing import Dict, List, Tuple

# Загрузка данных
@st.cache_data
def load_data() -> pd.DataFrame:
    excel_file = pd.ExcelFile('processed_with_params.xlsx')
    materials = excel_file.parse('Материалы')
    equipment = excel_file.parse('Оборудование')
    mechanisms = excel_file.parse('Механизмы')

    materials['Тип ресурса'] = 'Материалы'
    equipment['Тип ресурса'] = 'Оборудование'
    mechanisms['Тип ресурса'] = 'Механизмы'

    df = pd.concat([materials, equipment, mechanisms], ignore_index=True)

    # Переименование столбцов
    df = df.rename(columns={
        'Код ресурса': 'Код',
        'Наименование': 'Наименование',
        'Ед.изм.': 'Единица измерения',
        'Код ОКПД2': 'ОКПД2',
        'tokens': 'Теги'
    })

    # Заполнение пропущенных значений
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Не указано')
        elif df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(0)

    # Флаг "Имеет параметры"
    param_cols = [col for col in df.columns if col not in ['Код', 'Наименование', 'Единица измерения', 'ОКПД2', 'Теги', 'Тип ресурса']]
    df['Имеет параметры'] = df[param_cols].apply(lambda row: any(val not in [0, 'Не указано'] for val in row), axis=1)

    return df

df = load_data()

# Индекс для автодополнения
@st.cache_data
def build_autocomplete_index(df: pd.DataFrame) -> Tuple[Dict[str, set], Dict[int, str]]:
    token_index = defaultdict(set)
    name_map = {}
    for i, row in df.iterrows():
        name = str(row['Наименование']).lower()
        name_map[i] = name
        tokens = re.findall(r'\w+', name)
        for token in tokens:
            token_index[token].add(i)
    return token_index, name_map

token_index, name_map = build_autocomplete_index(df)

# Функция для автоматического определения фильтров
def get_available_filters(df: pd.DataFrame, resource_type: str) -> Dict[str, str]:
    """Возвращает словарь {Название параметра: имя столбца} для заданного типа ресурса."""
    filtered_df = df[df['Тип ресурса'] == resource_type]
    available_params = {}

    for col in filtered_df.columns:
        if col not in ['Код', 'Наименование', 'Единица измерения', 'ОКПД2', 'Теги', 'Тип ресурса', 'Имеет параметры']:
            # Красивое название параметра (убираем '_mm' и т.д.)
            pretty_name = col.replace('_', ' ').title().replace('Mpa', 'MPa').replace('Kw', 'kW')
            available_params[pretty_name] = col

    return available_params

# Сайдбар с фильтрами
st.sidebar.header("🔍 Фильтры ресурсов")

# Поиск по автодополнению
st.sidebar.subheader("Поиск по названию")
query = st.sidebar.text_input("Введите часть названия")
suggestions = []

if len(query) >= 3:
    q_tokens = query.lower().split()
    matched_indices = set()
    for token in q_tokens:
        for k in token_index:
            if token in k:
                matched_indices.update(token_index[k])

    for i in matched_indices:
        row = df.iloc[i]
        label = f"{row['Наименование']} ({row['Тип ресурса']})"
        suggestions.append((label, row['Код']))

    suggestions = sorted(suggestions, key=lambda x: x[0])[:10]
    options = [label for label, code in suggestions]
    selected_option = st.sidebar.selectbox("Варианты", options) if options else None
    if selected_option:
        selected_code = [code for label, code in suggestions if label == selected_option][0]
        filtered_df = df[df['Код'] == selected_code]
    else:
        filtered_df = df.copy()
else:
    filtered_df = df.copy()

# Фильтр по типу ресурса
resource_type = st.sidebar.multiselect(
    "Тип ресурса",
    options=df['Тип ресурса'].unique(),
    default=df['Тип ресурса'].unique()
)
if resource_type:
    filtered_df = filtered_df[filtered_df['Тип ресурса'].isin(resource_type)]

# Поиск по тегам и названию
search_query = st.sidebar.text_input("Поиск по тегам или названию")
if search_query:
    filtered_df = filtered_df[
        filtered_df['Наименование'].str.contains(search_query, case=False, na=False) |
        filtered_df['Теги'].str.contains(search_query, case=False, na=False)
    ]

# Фильтр по наличию параметров
has_params = st.sidebar.checkbox("Только ресурсы с параметрами", value=False)
if has_params:
    filtered_df = filtered_df[filtered_df['Имеет параметры']]

# Фильтры параметров (автоматические)
if len(resource_type) == 1:  # Фильтры показываем только если выбран один тип
    st.sidebar.subheader("⚙️ Фильтры параметров")
    available_params = get_available_filters(df, resource_type[0])

    for param_name, param_col in available_params.items():
        if param_col in filtered_df.columns:
            col_data = filtered_df[param_col]

            # Числовой параметр
            if pd.api.types.is_numeric_dtype(col_data):
                clean_series = col_data.replace(['Не указано', 'NaN', 'nan'], pd.NA).dropna()
                if not clean_series.empty:
                    min_val, max_val = clean_series.min(), clean_series.max()

                    if min_val != max_val:
                        st.sidebar.markdown(f"**{param_name}**")
                        filter_type = st.sidebar.radio(
                            f"Тип фильтра для {param_name}",
                            ["Ползунок", "Ручной ввод"],
                            key=f"filter_type_{param_col}"
                        )

                        if filter_type == "Ползунок":
                            val_range = st.sidebar.slider(
                                f"Диапазон {param_name}",
                                min_val, max_val, (min_val, max_val),
                                key=f"slider_{param_col}"
                            )
                            if val_range != (min_val, max_val):
                                filtered_df = filtered_df[
                                    (pd.to_numeric(filtered_df[param_col], errors='coerce') >= val_range[0]) &
                                    (pd.to_numeric(filtered_df[param_col], errors='coerce') <= val_range[1])
                                ]
                        else:
                            col1, col2 = st.sidebar.columns(2)
                            with col1:
                                min_input = col1.number_input(
                                    f"Мин. {param_name}",
                                    min_val, max_val, min_val,
                                    key=f"min_{param_col}"
                                )
                            with col2:
                                max_input = col2.number_input(
                                    f"Макс. {param_name}",
                                    min_val, max_val, max_val,
                                    key=f"max_{param_col}"
                                )
                            if min_input != min_val or max_input != max_val:
                                filtered_df = filtered_df[
                                    (pd.to_numeric(filtered_df[param_col], errors='coerce') >= min_input) &
                                    (pd.to_numeric(filtered_df[param_col], errors='coerce') <= max_input)
                                ]
            # Категориальный параметр
            elif col_data.nunique() > 1:
                unique_values = col_data.dropna().unique().tolist()
                try:
                    # Пробуем преобразовать в числа (если это возможно)
                    unique_values = [float(x) if str(x).replace('.', '').isdigit() else x for x in unique_values]
                    options = ['Все'] + sorted(unique_values, key=lambda x: str(x))  # Сортировка как строки
                except:
                    options = ['Все'] + sorted(unique_values, key=lambda x: str(x))  # Резервный вариант

                selected = st.sidebar.multiselect(
                    f"Значения {param_name}",
                    options=options,
                    default='Все',
                    key=f"select_{param_col}"
                )
                if 'Все' not in selected:
                    filtered_df = filtered_df[filtered_df[param_col].isin(selected)]

# Вкладки
tab1, tab2 = st.tabs(["Данные", "Визуализация"])

with tab1:
    # Основной интерфейс
    st.title("📊 Маркетплейс КСР")
    st.subheader(f"Найдено ресурсов: {len(filtered_df)}")

    # Отображение таблицы
    st.dataframe(
        filtered_df,
        column_config={
            "Код": "Код",
            "Наименование": "Название",
            "Единица измерения": "Ед. изм.",
            "ОКПД2": "ОКПД2",
            "Теги": "Теги",
            "Тип ресурса": "Тип",
            "Имеет параметры": st.column_config.CheckboxColumn("Параметры")
        },
        use_container_width=True,
        height=600
    )

    # Детализация выбранного ресурса
    if not filtered_df.empty:
        selected_resource = st.selectbox(
            "Выберите ресурс для детального просмотра",
            options=filtered_df['Код'] + " - " + filtered_df['Наименование']
        )
        if selected_resource:
            resource_code = selected_resource.split(" - ")[0]
            resource_details = filtered_df[filtered_df['Код'] == resource_code].iloc[0]

            st.subheader("🔍 Детали ресурса")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Код:** `{resource_details['Код']}`")
                st.markdown(f"**Тип:** {resource_details['Тип ресурса']}")
                st.markdown(f"**Название:** {resource_details['Наименование']}")
                st.markdown(f"**Ед. изм.:** {resource_details['Единица измерения']}")
                st.markdown(f"**ОКПД2:** {resource_details['ОКПД2']}")
                if resource_details['Теги'] != 'Не указано':
                    st.markdown(f"**Теги:** `{resource_details['Теги']}`")

            with col2:
                st.markdown("**📌 Параметры:**")
                available_params = get_available_filters(df, resource_details['Тип ресурса'])
                for param_name, param_col in available_params.items():
                    if param_col in resource_details and resource_details[param_col] not in [0, 'Не указано']:
                        st.markdown(f"- **{param_name}:** `{resource_details[param_col]}`")

    # Экспорт
    if not filtered_df.empty:
        st.download_button(
            label="📥 Скачать CSV",
            data=filtered_df.to_csv(index=False, sep=';', encoding='utf-8-sig'),
            file_name='resources_export.csv',
            mime='text/csv'
        )

with tab2:
    # Визуализация
    if not filtered_df.empty:
        st.header("Аналитика")
        tab1, tab2, tab3 = st.tabs(["Распределение по типам", "Параметры", "Частота параметров"])

        with tab1:
            fig, ax = plt.subplots()
            filtered_df['Тип ресурса'].dropna().value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_title("Доля типов ресурсов")
            ax.set_ylabel("")
            st.pyplot(fig)

        with tab2:
            if len(resource_type) == 1:
                available_params = get_available_filters(df, resource_type[0])
                numeric_params = {
                    name: col for name, col in available_params.items()
                    if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col])
                }

                selected_param = st.selectbox(
                    "Выберите параметр для анализа",
                    options=list(numeric_params.keys())
                )
                if selected_param:
                    param_col = numeric_params[selected_param]
                    fig, ax = plt.subplots(figsize=(10, 4))
                    filtered_df[param_col].plot.hist(bins=20, ax=ax, color='skyblue')
                    ax.set_title(f"Распределение {selected_param}")
                    ax.set_xlabel(selected_param)
                    ax.set_ylabel("Количество")
                    st.pyplot(fig)

        with tab3:
            df = pd.read_excel('processed_with_params.xlsx')
            cpc = df.loc[:, 'thickness_mm':].notna().sum()
            plt.figure(figsize=(14, 6))
            bars = plt.bar(cpc.index, cpc.values, width=1)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.5)
            plt.tight_layout()
            st.pyplot(plt)
