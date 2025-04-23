import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.io as pio

st.set_page_config(layout="wide")

names = ['Adim']
usernames = ['admin']
passwords = ['bigdata']

hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    authenticator.logout('Logout', 'main')
    # st.write(f'Welcome *{name}*!')
    st.title('Penetration Dashboard')

    # Set up Plotly template
    pio.templates.default = 'plotly'

    st.sidebar.title("Filter and Display Selection")

    @st.cache_data
    def getdata():
        df = pd.read_parquet(r'sonlt9_support_v2.parquet')
        continuous_columns = df.select_dtypes(include=['float64', 'int64']).columns
        category_columns = df.select_dtypes(include=['object']).columns
        df[continuous_columns] = df[continuous_columns].fillna(0)
        df[category_columns] = df[category_columns].fillna('Unknown')
        df['age_group'] = pd.cut(df['age'], bins=[18, 23, 31, 41, 51, max(df['age']) + 10], right=False, include_lowest=True).astype(str)
        # df.drop('age', axis = 1, inplace=True)
        df['use_product'] = df['use_product'].map({
            0: "Never Used",
            1: "Used Now",
            2: "Not Used Now - Used in the past"
        })
        df['age'] = df['age'].astype('Int64')
        df.rename(columns={
            'use_product': 'Product Used',
            'thu_nhap_bq_3_thang': 'Average Income Last 3 Months',
            'age': 'Age',
            'gender': 'Gender',
            'city': 'City',
            'occupation': 'Occupation',
            'mtd_casa': 'CASA Balance by Month',
            'qtd_casa': 'CASA Balance by Quarter',
            'ytd_casa': 'CASA Balance by Year',
            'age_group': 'Age Group'
        }, inplace=True)
        df['CASA Balance by Month'] = df['CASA Balance by Month']*1e6
        df['CASA Balance by Quarter'] = df['CASA Balance by Quarter']*1e6
        df['CASA Balance by Year'] = df['CASA Balance by Year']*1e6
        return df

    # Load data
    df_tmp = getdata()

    # Add filter for 'Product Used'
    st.sidebar.markdown(f"**Filter by Product Used**")
    product_used_values = df_tmp['Product Used'].unique()
    selected_product_used = st.sidebar.multiselect(f"Select Product Used", product_used_values, default=product_used_values.tolist())

    # Apply the filter for 'Product Used' if selected
    if selected_product_used:
        df_tmp = df_tmp[df_tmp['Product Used'].isin(selected_product_used)]

    # Categorical filters
    for col in df_tmp.select_dtypes(include=['object']).columns:
        if col == 'Product Used':  # Skip 'Product Used' as it's already filtered
            continue

        st.sidebar.markdown(f"**Filter by {col}**")
        selected_values = st.sidebar.multiselect(f"Select {col}", df_tmp[col].unique(), default=None)
        if selected_values:
            df_tmp = df_tmp[df_tmp[col].isin(selected_values)]
        
        st.sidebar.markdown("----------")  # Add a separator line between filters

    # Numeric filters (only apply after binning)
    for col in df_tmp.select_dtypes(include=['float64', 'int64', 'int32']).columns:
        st.sidebar.markdown(f"**Filter by {col}**")
        bin_edges_input = st.sidebar.text_input(
            f"Enter bin edges for {col} (comma-separated) or leave blank for default",
            value=""
        )
        try:
            # Use user-defined bins if provided, else create default bins
            if bin_edges_input.strip():
                bin_edges = list(map(float, bin_edges_input.split(',')))
                
            else:
                # Default bins: divide data into 6 equal intervals
                col_min = df_tmp[col].min()
                col_max = df_tmp[col].max()
                col_median = df_tmp[col].median()
                col_1 = round((col_median - col_min)/100, ndigits=0)
                col_2 = round((col_max - col_median)/100, ndigits=0)

                bin_edges = [col_min, col_min + col_1 , col_median, col_median + col_2, col_max]
                bin_edges = np.array(bin_edges)
            # st.write(col)
            # st.write(bin_edges)
            df_tmp[f'{col}_bin'] = pd.cut(df_tmp[col], bins=bin_edges, right=False, include_lowest=True, duplicates='drop')
            df_tmp[f'{col}_bin'] = df_tmp[f'{col}_bin'].apply(lambda x: f'[{x.left:,.0f} --  {x.right:,.0f})')
            df_tmp[f'{col}_bin'] = df_tmp[f'{col}_bin'].astype(str)
            # st.write(df_tmp[f'{col}_bin'].value_counts())
            # Add filter for binned data
            selected_bins = st.sidebar.multiselect(f"Select bins for {col}", df_tmp[f'{col}_bin'].unique(), default=None)
            if selected_bins:
                df_tmp = df_tmp[df_tmp[f'{col}_bin'].isin(selected_bins)]
            
        except ValueError:
            # st.dataframe(df_tmp)
            st.sidebar.error(f"Invalid input for {col}. Please enter numeric values separated by commas.")
        
        st.sidebar.markdown("----------")  # Add a separator line between filters

    # Bar chart for 'Product Used'
    product_counts = df_tmp['Product Used'].value_counts().reset_index()
    product_counts.columns = ['Product Used', 'Count']
    product_counts = product_counts.sort_values(by='Count', ascending=False)  # Sort by count descending

    fig_product = px.bar(
        product_counts,
        x='Product Used',
        y='Count',
        title='Customer Usage of Product',
        text='Count',
        color='Product Used'
    )
    fig_product.update_traces(texttemplate='%{text}', textposition='outside')
    st.plotly_chart(fig_product, use_container_width=True)

    # Loop through all columns and create charts
    columns_for_filter = df_tmp.select_dtypes(include=['object']).columns.tolist() + df_tmp.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()

    for col in columns_for_filter:
        if col == 'Product Used':  # Skip 'Product Used'
            continue

        if df_tmp[col].dtype == 'object':  # Categorical columns
            # Kiểm tra nếu tên cột có '_bin'
            if '_bin' in col:
                # Tên cột gốc (trước khi thêm '_bin')
                original_col = col.replace('_bin', '')
                
                if original_col in df_tmp.columns:  # Đảm bảo cột gốc tồn tại
                    mean_value = df_tmp[original_col].mean()
                    median_value = df_tmp[original_col].median()

                    # Hiển thị thống kê
                    st.markdown(f"### Statistics for {original_col}")
                    st.write(f"**Mean:** {mean_value:,.2f}")
                    st.write(f"**Median:** {median_value:,.2f}")
            
            # Nhóm dữ liệu và tạo biểu đồ
            grouped_data = df_tmp.groupby([col, 'Product Used']).size().reset_index(name='Count')
            grouped_data = grouped_data.sort_values(by='Count', ascending=False)
            
            fig = px.bar(
                grouped_data,
                x=col,
                y='Count',
                color='Product Used',
                title=f'Distribution of {col} by Product Usage',
                text='Count',
                barmode='stack'
            )
            st.plotly_chart(fig, use_container_width=True)


        elif col in df_tmp.select_dtypes(include=['float64', 'int64', 'int32']).columns:  # Continuous columns
            if f'{col}_bin' in df_tmp.columns:
                grouped_data = df_tmp.groupby([f'{col}_bin', 'Product Used']).size().reset_index(name='Count')
                grouped_data = grouped_data.sort_values(by='Count', ascending=False)

                # Calculate mean and median
                mean_value = df_tmp[col].mean()
                median_value = df_tmp[col].median()

                fig = px.bar(
                    grouped_data,
                    x=f'{col}_bin',
                    y='Count',
                    color='Product Used',
                    title=f'Distribution of {col} (Binned) by Product Usage\nMean: {mean_value:,.2f} | Median: {median_value:,.2f}',
                    text='Count',
                    barmode='stack'
                )
                # st.plotly_chart(fig, use_container_width=True)
                # st.markdown(f"### Statistics for {col}")
                # st.write(f"**Mean:** {mean_value:,.2f}")
                # st.write(f"**Median:** {median_value:,.2f}")
            else:
                # If no binning, show statistics only
                mean_value = df_tmp[col].mean()
                median_value = df_tmp[col].median()

                # st.markdown(f"### Statistics for {col}")
                # st.write(f"**Mean:** {mean_value:,.2f}")
                # st.write(f"**Median:** {median_value:,.2f}")
