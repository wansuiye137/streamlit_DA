import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import json
from datetime import datetime
import altair as alt

# 页面配置
st.set_page_config(
    page_title="零售数据分析平台",
    page_icon="👩🏻‍💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 数据根目录
DATA_ROOT = "data/"


# 列出所有可用数据集
def list_datasets():
    """获取所有可用数据集"""
    datasets = []
    for name in os.listdir(DATA_ROOT):
        path = os.path.join(DATA_ROOT, name)
        if os.path.isdir(path):
            if glob.glob(os.path.join(path, f"{name}_data_*.csv")):
                datasets.append(name)
    return sorted(datasets)


# 从侧边栏获取选中的数据集
def get_selected_dataset():
    """从会话状态获取选中的数据集"""
    datasets = list_datasets()

    # 初始化会话状态
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = datasets[0] if datasets else None

    # 在侧边栏显示数据集选择器
    with st.sidebar:
        st.header("📁 数据集选择")
        selected = st.selectbox(
            "选择数据集",
            options=datasets,
            index=datasets.index(st.session_state.selected_dataset) if datasets else 0,
            key="dataset_selector"
        )

        if selected != st.session_state.selected_dataset:
            st.session_state.selected_dataset = selected
            # 清除缓存，强制重新加载数据
            st.cache_data.clear()

        st.markdown("---")
        st.info(f"当前分析: **{st.session_state.selected_dataset}**")

    return st.session_state.selected_dataset


# 封装文件处理逻辑 - 通用版本
def process_data_files(dataset_name):
    """处理所有数据文件并计算统计指标"""
    dataset_dir = os.path.join(DATA_ROOT, dataset_name)
    files = sorted(glob.glob(os.path.join(dataset_dir, f"{dataset_name}_data_*.csv")))
    stats_files = {
        'category': os.path.join(dataset_dir, f"{dataset_name}_category_stats.jsonl"),
        'null': os.path.join(dataset_dir, f"{dataset_name}_null_stats.jsonl"),
        'unique': os.path.join(dataset_dir, f"{dataset_name}_unique_stats.jsonl")
    }

    # 检查已处理的日期
    processed_dates = {key: set() for key in stats_files}
    for key, file_path in stats_files.items():
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    processed_dates[key].add(data['date'])

    # 处理新文件
    for file in files:
        try:
            # 从文件名解析日期
            filename = os.path.basename(file)
            date_str = filename.split('_')[-1].replace('.csv', '')

            # 读取CSV
            df = pd.read_csv(file, dtype=str, keep_default_na=False)

            # 分类统计
            if date_str not in processed_dates['category']:
                count_df = df.groupby(['department', 'category', 'subcategory']).size().reset_index(name='count')
                with open(stats_files['category'], 'a') as f:
                    for _, row in count_df.iterrows():
                        record = {
                            'date': date_str,
                            'department': str(row['department']),
                            'category': str(row['category']),
                            'subcategory': str(row['subcategory']),
                            'count': int(row['count'])
                        }
                        f.write(json.dumps(record) + '\n')
                processed_dates['category'].add(date_str)

            # 空值统计
            if date_str not in processed_dates['null']:
                null_counts = {}
                for col in df.columns:
                    null_counts[col] = int((df[col] == '').sum())
                null_counts['date'] = date_str
                with open(stats_files['null'], 'a') as f:
                    f.write(json.dumps(null_counts) + '\n')
                processed_dates['null'].add(date_str)

            # 唯一值统计
            if date_str not in processed_dates['unique']:
                unique_counts = {}
                for col in df.columns:
                    # 计算非空唯一值数量
                    non_empty = df[col][df[col] != '']
                    unique_counts[col] = int(non_empty.nunique())
                unique_counts['date'] = date_str
                with open(stats_files['unique'], 'a') as f:
                    f.write(json.dumps(unique_counts) + '\n')
                processed_dates['unique'].add(date_str)

        except Exception as e:
            st.error(f"处理文件 {filename} 时出错: {str(e)}")

    # 返回所有可用日期
    return sorted(processed_dates['category'])


# 封装数据加载逻辑
@st.cache_data
def load_stat_data(file_name):
    """加载统计JSONL文件数据"""
    if os.path.exists(file_name):
        data = []
        with open(file_name, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)
    return pd.DataFrame()


# 加载单日原始数据用于异常检测 - 通用版本
@st.cache_data
def load_daily_data(dataset_name, date_str):
    dataset_dir = os.path.join(DATA_ROOT, dataset_name)
    file = os.path.join(dataset_dir, f"{dataset_name}_data_{date_str}.csv")
    if os.path.exists(file):
        df = pd.read_csv(file, dtype=str, keep_default_na=False)
        # 转换价格列为数值类型
        if 'current_price' in df.columns:
            df['current_price'] = pd.to_numeric(df['current_price'], errors='coerce')
        if 'retail_price' in df.columns:
            df['retail_price'] = pd.to_numeric(df['retail_price'], errors='coerce')
        return df
    return pd.DataFrame()


# 主应用函数
def main():
    # 获取选中的数据集
    dataset_name = get_selected_dataset()

    if not dataset_name:
        st.warning("没有找到可用数据集。请确保在data/目录下有数据集文件夹")
        return

    # 设置页面标题
    st.title(f"📊 {dataset_name.capitalize()} 产品数据分析")

    # 预计算统计结果
    with st.spinner("处理数据文件中..."):
        available_dates = process_data_files(dataset_name)

    if not available_dates:
        st.warning("没有可用的日期数据进行展示")
        return

    # 加载统计数据
    dataset_dir = os.path.join(DATA_ROOT, dataset_name)
    stats_df = load_stat_data(os.path.join(dataset_dir, f"{dataset_name}_category_stats.jsonl"))
    null_stats_df = load_stat_data(os.path.join(dataset_dir, f"{dataset_name}_null_stats.jsonl"))
    unique_stats_df = load_stat_data(os.path.join(dataset_dir, f"{dataset_name}_unique_stats.jsonl"))

    # ====================== 第一部分：概览表格和折线图 ======================
    st.header("📊 1. 产品分类概览")

    if not stats_df.empty:
        # 创建透视表
        pivot_df = stats_df.pivot_table(
            index=['department', 'category', 'subcategory'],
            columns='date',
            values='count',
            fill_value=0
        ).reset_index()

        # 添加历史趋势列
        pivot_df['历史趋势'] = pivot_df[available_dates].apply(
            lambda row: [int(x) for x in row],
            axis=1
        )

        st.subheader("分类产品数量统计")

        # 创建列配置
        column_config = {
            "department": "部门",
            "category": "类别",
            "subcategory": "子分类",
            "历史趋势": st.column_config.LineChartColumn(
                "数量变化趋势",
                help="各日期产品数量变化",
                width="medium"
            )
        }

        # 添加日期列的配置
        for date in available_dates:
            if date in pivot_df.columns:
                column_config[date] = st.column_config.NumberColumn(
                    date,
                    format="%d件"
                )

        # 显示表格
        st.dataframe(
            pivot_df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True
        )

        # ====================== 交互式折线图 ======================
        st.subheader("产品数量变化趋势图")

        # 准备绘图数据
        trend_data = stats_df.copy()
        trend_data['日期'] = pd.to_datetime(trend_data['date'])

        # 创建多级选择器
        col1, col2, col3 = st.columns(3)

        with col1:
            departments = st.multiselect(
                "选择部门",
                options=trend_data['department'].unique(),
                default=trend_data['department'].unique()[0] if len(trend_data['department'].unique()) > 0 else []
            )

        with col2:
            if departments:
                categories_options = trend_data[trend_data['department'].isin(departments)]['category'].unique()
            else:
                categories_options = trend_data['category'].unique()

            categories = st.multiselect(
                "选择类别",
                options=categories_options,
                default=categories_options[0] if len(categories_options) > 0 else []
            )

        with col3:
            if departments and categories:
                subcategories_options = trend_data[
                    (trend_data['department'].isin(departments)) &
                    (trend_data['category'].isin(categories))
                    ]['subcategory'].unique()
            elif departments:
                subcategories_options = trend_data[trend_data['department'].isin(departments)]['subcategory'].unique()
            elif categories:
                subcategories_options = trend_data[trend_data['category'].isin(categories)]['subcategory'].unique()
            else:
                subcategories_options = trend_data['subcategory'].unique()

            subcategories = st.multiselect(
                "选择子分类",
                options=subcategories_options,
                default=subcategories_options[:min(3, len(subcategories_options))] if len(
                    subcategories_options) > 0 else []
            )

        # 筛选数据
        filtered_data = trend_data.copy()
        if departments:
            filtered_data = filtered_data[filtered_data['department'].isin(departments)]
        if categories:
            filtered_data = filtered_data[filtered_data['category'].isin(categories)]
        if subcategories:
            filtered_data = filtered_data[filtered_data['subcategory'].isin(subcategories)]

        # 交互式折线图
        if not filtered_data.empty:
            # 计算Y轴范围
            y_min_val = filtered_data['count'].min()
            y_max_val = filtered_data['count'].max()
            range_buffer = max(1, (y_max_val - y_min_val) * 0.1) if y_max_val != y_min_val else 1
            y_min_val = max(0, y_min_val - range_buffer)
            y_max_val += range_buffer

            chart = alt.Chart(filtered_data).mark_line(point=True).encode(
                x=alt.X('日期:T', title='日期'),
                y=alt.Y('count:Q', title='产品数量', scale=alt.Scale(domain=[y_min_val, y_max_val])),
                color=alt.Color('subcategory:N', legend=alt.Legend(title="子分类")),
                tooltip=[
                    alt.Tooltip('department:N', title='部门'),
                    alt.Tooltip('category:N', title='类别'),
                    alt.Tooltip('subcategory:N', title='子分类'),
                    alt.Tooltip('日期:T', title='日期', format='%Y-%m-%d'),
                    alt.Tooltip('count:Q', title='数量', format='.0f')
                ]
            ).properties(
                height=500,
                title='产品数量随时间变化趋势'
            ).interactive()

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("请至少选择一个分类进行可视化")
    else:
        st.warning("没有可用的统计数据进行展示")

    # ====================== 第二部分：详细表格与日期筛选 ======================
    st.header("📅 2. 详细分类数据")

    if not stats_df.empty:
        # 日期选择滑块
        selected_date = st.select_slider("选择查看日期", options=available_dates)

        # 筛选数据
        detailed_df = stats_df[stats_df['date'] == selected_date]
        detailed_df = detailed_df.sort_values('count', ascending=False)

        st.subheader(f"{selected_date} 产品分类统计")

        # 创建列配置
        max_value = int(detailed_df['count'].max()) if not detailed_df.empty else 1
        detailed_config = {
            "department": "部门",
            "category": "类别",
            "subcategory": "子分类",
            "count": st.column_config.ProgressColumn(
                "产品数量",
                help="该子分类的产品数量",
                format="%d",
                min_value=0,
                max_value=max_value
            )
        }

        st.dataframe(
            detailed_df[['department', 'category', 'subcategory', 'count']],
            column_config=detailed_config,
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("没有可用的统计数据进行展示")

    # ====================== 第三部分：异常价格检测 ======================
    st.header("⚠️ 3. 异常价格检测")

    if not available_dates:
        st.warning("没有可用的数据进行异常检测")
    else:
        # 选择日期进行异常检测
        anomaly_date = st.selectbox("选择检测日期", options=available_dates, index=len(available_dates) - 1)
        daily_df = load_daily_data(dataset_name, anomaly_date)

        if daily_df.empty:
            st.warning(f"没有找到 {anomaly_date} 的数据")
        else:
            # 自定义阈值输入
            st.subheader("设置异常价格阈值")
            col1, col2 = st.columns(2)

            with col1:
                low_threshold_percent = st.number_input(
                    "价格过低阈值（相对于中位数的百分比）",
                    min_value=0.1, max_value=100.0, value=20.0, step=0.1,
                    help="例如：输入20表示价格低于中位数的20%被视为异常"
                )

            with col2:
                high_threshold_multiple = st.number_input(
                    "价格过高阈值（相对于中位数的倍数）",
                    min_value=1.0, max_value=100.0, value=5.0, step=0.1,
                    help="例如：输入5表示价格高于中位数的5倍被视为异常"
                )

            # 显示异常值检测标准
            st.subheader("异常值检测标准")
            st.write(f"""
            我们使用以下标准检测异常价格：
            - **缺失值**：价格数据为空
            - **非正值**：价格 ≤ 0
            - **价格过低**：价格 < 中位数的{low_threshold_percent}%
            - **价格过高**：价格 > 中位数的{high_threshold_multiple}倍
            """)

            # 检测异常价格函数
            def detect_price_anomalies(price_series, low_threshold_percent, high_threshold_multiple):
                anomalies = pd.Series(False, index=price_series.index)
                too_low = pd.Series(False, index=price_series.index)
                too_high = pd.Series(False, index=price_series.index)

                median_price = price_series.median()

                # 检测缺失值
                missing = price_series.isna()

                # 检测零/负值
                non_positive = price_series <= 0

                # 检测过低价格
                if median_price > 0:
                    low_threshold = median_price * (low_threshold_percent / 100)
                    too_low = (price_series < low_threshold) & (price_series > 0)
                else:
                    too_low = pd.Series(False, index=price_series.index)

                # 检测过高价格
                if median_price > 0:
                    high_threshold = median_price * high_threshold_multiple
                    too_high = price_series > high_threshold
                else:
                    too_high = pd.Series(False, index=price_series.index)

                # 组合所有异常条件
                anomalies = missing | non_positive | too_low | too_high

                return anomalies, too_low, too_high, median_price

            # 识别异常值
            current_median = retail_median = None
            current_anomalies = retail_anomalies = pd.Series(False)

            if 'current_price' in daily_df.columns:
                (current_anomalies, current_too_low, current_too_high, current_median) = detect_price_anomalies(
                    daily_df['current_price'], low_threshold_percent, high_threshold_multiple
                )
                daily_df['current_price_anomaly'] = current_anomalies

            if 'retail_price' in daily_df.columns:
                (retail_anomalies, retail_too_low, retail_too_high, retail_median) = detect_price_anomalies(
                    daily_df['retail_price'], low_threshold_percent, high_threshold_multiple
                )
                daily_df['retail_price_anomaly'] = retail_anomalies

            # 获取异常数据
            price_columns = [c for c in ['current_price_anomaly', 'retail_price_anomaly'] if c in daily_df.columns]

            if not price_columns:
                st.warning("数据集中没有找到价格列")
                return

            anomaly_condition = daily_df[price_columns[0]]
            for col in price_columns[1:]:
                anomaly_condition = anomaly_condition | daily_df[col]

            price_anomalies = daily_df[anomaly_condition]

            # 显示异常值统计
            st.subheader("异常价格统计")
            col1, col2 = st.columns(2)

            with col1:
                if current_median is not None:
                    st.metric("当前价格中位数", f"${current_median:.2f}" if pd.notna(current_median) else "N/A")
                    st.metric("当前价格异常总数", f"{current_anomalies.sum()}条")
                    st.metric("当前价格缺失", f"{daily_df['current_price'].isna().sum()}条")
                    st.metric("当前价格非正", f"{(daily_df['current_price'] <= 0).sum()}条")
                    if 'current_too_low' in locals():
                        st.metric("当前价格过低", f"{current_too_low.sum()}条")
                else:
                    st.info("没有当前价格数据")

            with col2:
                if retail_median is not None:
                    st.metric("零售价格中位数", f"${retail_median:.2f}" if pd.notna(retail_median) else "N/A")
                    st.metric("零售价格异常总数", f"{retail_anomalies.sum()}条")
                    st.metric("零售价格缺失", f"{daily_df['retail_price'].isna().sum()}条")
                    st.metric("零售价格非正", f"{(daily_df['retail_price'] <= 0).sum()}条")
                    if 'retail_too_high' in locals():
                        st.metric("零售价格过高", f"{retail_too_high.sum()}条")
                else:
                    st.info("没有零售价格数据")

            # 显示异常数据表格
            st.subheader("异常价格记录")

            if not price_anomalies.empty:
                # 添加异常类型列
                price_anomalies['异常类型'] = ""

                # 当前价格异常
                if 'current_price' in daily_df.columns:
                    price_anomalies.loc[price_anomalies['current_price'].isna(), '异常类型'] += "当前价格缺失; "
                    price_anomalies.loc[price_anomalies['current_price'] <= 0, '异常类型'] += "当前价格非正; "
                    if 'current_too_low' in locals():
                        price_anomalies.loc[current_too_low[price_anomalies.index], '异常类型'] += "当前价格过低; "
                    if 'current_too_high' in locals():
                        price_anomalies.loc[current_too_high[price_anomalies.index], '异常类型'] += "当前价格过高; "

                # 零售价格异常
                if 'retail_price' in daily_df.columns:
                    price_anomalies.loc[price_anomalies['retail_price'].isna(), '异常类型'] += "零售价格缺失; "
                    price_anomalies.loc[price_anomalies['retail_price'] <= 0, '异常类型'] += "零售价格非正; "
                    if 'retail_too_low' in locals():
                        price_anomalies.loc[retail_too_low[price_anomalies.index], '异常类型'] += "零售价格过低; "
                    if 'retail_too_high' in locals():
                        price_anomalies.loc[retail_too_high[price_anomalies.index], '异常类型'] += "零售价格过高; "

                # 添加价格比较列
                def format_price_comparison(row):
                    parts = []
                    if 'current_price' in row and pd.notna(row['current_price']):
                        parts.append(f"当前价: ${row['current_price']:.2f}")
                    if 'retail_price' in row and pd.notna(row['retail_price']):
                        parts.append(f"零售价: ${row['retail_price']:.2f}")
                    return " | ".join(parts) if parts else "价格数据不完整"

                price_anomalies['价格比较'] = price_anomalies.apply(format_price_comparison, axis=1)

                # 创建列配置 - 根据数据集调整显示的列
                display_columns = ['product_name', '价格比较', '异常类型', 'product_url']
                anomaly_config = {
                    "product_name": "产品名称",
                    "价格比较": st.column_config.TextColumn("价格比较"),
                    "异常类型": "异常类型",
                    "product_url": st.column_config.LinkColumn("产品链接")
                }

                if 'brand' in daily_df.columns:
                    display_columns.insert(1, 'brand')
                    anomaly_config['brand'] = "品牌"
                if 'department' in daily_df.columns:
                    display_columns.insert(1, 'department')
                    anomaly_config['department'] = "部门"

                st.dataframe(
                    price_anomalies[display_columns],
                    column_config=anomaly_config,
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.success("🎉 未检测到异常价格记录")

    # ====================== 第四部分：空值统计板块 ======================
    st.header("📉 4. 数据质量分析 - 空值统计")

    if not null_stats_df.empty:
        st.subheader("各日期文件中的空值数量")

        # 确保日期是字符串类型
        null_stats_df['date'] = null_stats_df['date'].astype(str)

        # 选择要展示的列 (排除日期列)
        columns_to_show = [col for col in null_stats_df.columns if col != 'date' and not col.startswith('Unnamed')]

        # 创建列配置
        null_column_config = {
            "date": st.column_config.TextColumn("日期")
        }

        # 为每列添加配置
        for col in columns_to_show:
            null_column_config[col] = st.column_config.NumberColumn(
                col,
                help=f"{col}列的空值数量",
                format="%d"
            )

        # 显示表格
        st.dataframe(
            null_stats_df[['date'] + columns_to_show],
            column_config=null_column_config,
            hide_index=True,
            use_container_width=True
        )

        # 空值变化趋势图
        st.subheader("空值数量变化趋势")

        # 准备绘图数据
        null_trend_data = null_stats_df.melt(
            id_vars=['date'],
            value_vars=columns_to_show,
            var_name='column',
            value_name='null_count'
        )

        # 转换为日期格式
        null_trend_data['date'] = pd.to_datetime(null_trend_data['date'])

        # 选择要展示的列
        selected_columns = st.multiselect(
            "选择要分析的列 (空值)",
            options=columns_to_show,
            default=columns_to_show[:min(5, len(columns_to_show))]
        )

        if selected_columns:
            filtered_null_data = null_trend_data[null_trend_data['column'].isin(selected_columns)]

            # 创建折线图
            chart = alt.Chart(filtered_null_data).mark_line(point=True).encode(
                x=alt.X('date:T', title='日期'),
                y=alt.Y('null_count:Q', title='空值数量'),
                color=alt.Color('column:N', legend=alt.Legend(title="列名")),
                tooltip=[
                    alt.Tooltip('date:T', title='日期', format='%Y-%m-%d'),
                    alt.Tooltip('column:N', title='列名'),
                    alt.Tooltip('null_count:Q', title='空值数量')
                ]
            ).properties(
                height=400,
                title='各列空值数量随时间变化'
            ).interactive()

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("请至少选择一列进行可视化")
    else:
        st.warning("没有可用的空值统计数据")

    # ====================== 第五部分：唯一值统计板块 ======================
    st.header("🔍 5. 数据质量分析 - 唯一值统计")

    if not unique_stats_df.empty:
        st.subheader("各日期文件中的唯一值数量")

        # 确保日期是字符串类型
        unique_stats_df['date'] = unique_stats_df['date'].astype(str)

        # 选择要展示的列 (排除日期列)
        columns_to_show = [col for col in unique_stats_df.columns if col != 'date' and not col.startswith('Unnamed')]

        # 创建列配置
        unique_column_config = {
            "date": st.column_config.TextColumn("日期")
        }

        # 为每列添加配置
        for col in columns_to_show:
            unique_column_config[col] = st.column_config.NumberColumn(
                col,
                help=f"{col}列的唯一值数量",
                format="%d"
            )

        # 显示表格
        st.dataframe(
            unique_stats_df[['date'] + columns_to_show],
            column_config=unique_column_config,
            hide_index=True,
            use_container_width=True
        )

        # 唯一值变化趋势图
        st.subheader("唯一值数量变化趋势")

        # 准备绘图数据
        unique_trend_data = unique_stats_df.melt(
            id_vars=['date'],
            value_vars=columns_to_show,
            var_name='column',
            value_name='unique_count'
        )

        # 转换为日期格式
        unique_trend_data['date'] = pd.to_datetime(unique_trend_data['date'])

        # 选择要展示的列
        selected_columns = st.multiselect(
            "选择要分析的列 (唯一值)",
            options=columns_to_show,
            default=columns_to_show[:min(5, len(columns_to_show))]
        )

        if selected_columns:
            filtered_unique_data = unique_trend_data[unique_trend_data['column'].isin(selected_columns)]

            # 创建折线图
            chart = alt.Chart(filtered_unique_data).mark_line(point=True).encode(
                x=alt.X('date:T', title='日期'),
                y=alt.Y('unique_count:Q', title='唯一值数量'),
                color=alt.Color('column:N', legend=alt.Legend(title="列名")),
                tooltip=[
                    alt.Tooltip('date:T', title='日期', format='%Y-%m-%d'),
                    alt.Tooltip('column:N', title='列名'),
                    alt.Tooltip('unique_count:Q', title='唯一值数量')
                ]
            ).properties(
                height=400,
                title='各列唯一值数量随时间变化'
            ).interactive()

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("请至少选择一列进行可视化")
    else:
        st.warning("没有可用的唯一值统计数据")


# 运行应用
if __name__ == "__main__":
    main()