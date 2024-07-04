import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

# Function to load and combine data from multiple CSV files
def load_and_combine_data(file_paths):
    data_frames = []
    for file in file_paths:
        if isinstance(file, str):
            data_frames.append(pd.read_csv(file))
        else:
            data_frames.append(pd.read_csv(StringIO(file.getvalue().decode("utf-8"))))
    df = pd.concat(data_frames, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Function to update the graph
def update_graph(G, current_date, df):
    # Filter the data up to the current date
    df_filtered = df[df['Date'] <= current_date]

    # Add nodes and store data corresponding to each stock in the node itself
    stock_symbols = df_filtered['Stock_Symbol'].unique()
    for stock_symbol in stock_symbols:
        stock_data = df_filtered[df_filtered['Stock_Symbol'] == stock_symbol]
        if stock_symbol not in G:
            G.add_node(stock_symbol, data=stock_data)
        else:
            G.nodes[stock_symbol]['data'] = stock_data

    # Remove nodes that no longer have data
    for node in list(G.nodes):
        if node not in stock_symbols:
            G.remove_node(node)

    # Clear existing edges
    G.remove_edges_from(list(G.edges))

    # Compute the correlation between the stock prices of each pair of stocks
    for i in range(len(stock_symbols)):
        for j in range(i + 1, len(stock_symbols)):
            stock_i = stock_symbols[i]
            stock_j = stock_symbols[j]

            prices_i = df_filtered[df_filtered['Stock_Symbol'] == stock_i]['Price'].values
            prices_j = df_filtered[df_filtered['Stock_Symbol'] == stock_j]['Price'].values

            if len(prices_i) > 1 and len(prices_j) > 1:
                correlation = np.corrcoef(prices_i, prices_j)[0, 1]
                weight = abs(correlation)  # Use the absolute value of the correlation as the edge weight

                G.add_edge(stock_i, stock_j, weight=weight)
                G.add_edge(stock_j, stock_i, weight=weight)

# Function to visualize the graph
def visualize_graph(G, current_date):
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G)
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] * 10 for edge in edges]  # Scale weights for better visualization

    nx.draw(G, pos, with_labels=True, node_size=7000, node_color="skyblue", edge_color="gray", linewidths=2, font_size=10, width=weights)

    # Add edge labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f"Graph at date: {current_date.date()}")
    st.pyplot(plt)

# Function to remove a node and its data from the DataFrame and graph
def remove_stock_data(df, stock_symbol):
    df = df[df['Stock_Symbol'] != stock_symbol]
    return df

# Streamlit app
st.title('Stock Graph Visualization')

# File upload for initial data
uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    df = load_and_combine_data(uploaded_files)
    st.write(df)

    # Initialize the graph
    G = nx.DiGraph()

    # Get the unique dates from the DataFrame
    dates = pd.to_datetime(df['Date']).unique()

    # Update and visualize the graph for all dates
    if st.button('Update Graph'):
        for current_date in dates:
            st.write(f"Graph at date: {current_date}")
            update_graph(G, current_date, df)
            visualize_graph(G, current_date)

    # Node removal
    node_to_remove = st.text_input("Enter Stock Symbol to Remove")
    if st.button('Remove Node'):
        df = remove_stock_data(df, node_to_remove)
        for current_date in dates:
            st.write(f"Graph at date: {current_date} (After Removing {node_to_remove})")
            update_graph(G, current_date, df)
            visualize_graph(G, current_date)

    # New file upload for additional data
    new_file = st.file_uploader("Upload New CSV for Node Addition", type=["csv"])
    if new_file:
        new_df = pd.read_csv(new_file)
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        df_combined = pd.concat([df, new_df], ignore_index=True)

        # Update and visualize the graph with new data
        if st.button('Add New Nodes and Update Graph'):
            for current_date in dates:
                st.write(f"Graph at date: {current_date} (With New Nodes)")
                update_graph(G, current_date, df_combined)
                visualize_graph(G, current_date)

    # Option to visualize the entire evolution over time
    if st.button('Visualize Evolution Over Time'):
        for current_date in dates:
            st.write(f"Graph at date: {current_date} (Evolution Over Time)")
            update_graph(G, current_date, df)
            visualize_graph(G, current_date)
