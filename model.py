import numpy as np
import pandas as pd
.from faker import Faker  # for creating fake data
import random  # modules
import plotly.graph_objects as go  # for 3D graphs and it is a module
from itertools import combinations  # it combines
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # it helps for dimension reduction e.g., from 30 dimensions to 2D
import plotly.express as px  # for plotting graphs
import warnings  # we will have irrelevant data for those we will get warnings
from tqdm import tqdm  # for progress bar

warnings.filterwarnings("ignore", category=DeprecationWarning)  # with this, we can see useful errors

# "seeding" means e.g., we have a die then we want the same number every time 
np.random.seed(44)
random.seed(44)  # disclosing

# Faker makes fake data 
fake = Faker()  # it is like a function

# we use def for which type of data we want

def generate_data(num_products=10, num_customers=100, num_transactions=500):
    products = [fake.word() for _ in range(num_products)]
    transactions = []
    for _ in range(num_transactions):
        customer_id = random.randint(1, num_customers)  # here it creates 1 to 100 customer IDs
        basket_size = random.randint(1, 5)  # it was depending on the number of products
        basket = random.sample(products, basket_size)  # here it creates a sample based on the number of products according to basket size
        transactions.append({
            'customer_id': customer_id,
            'products': basket
        })
    df = pd.DataFrame(transactions)  # for converting into raw 
    df_encoded = df.explode("products").pivot_table(  # example: packing gift packs, it will filter category-wise
        index='customer_id',
        columns='products',
        aggfunc=lambda x: 1,  # values will be from 0 to 1
        fill_value=0
    )
    return df_encoded

# APRIORI ALGO! INTRO: it will search common pairs e.g., chicken and beer
def simple_apriori(df, min_support=0.1, min_confidence=0.5):
    def support(item_set):
        return (df[list(item_set)].sum(axis=1) == len(item_set)).mean()

    items = set(df.columns)
    item_sets = [frozenset([item]) for item in items]  # frozen it will search for connecting pairs
    rules = []

    for k in range(2, len(items) + 1):
        item_sets = [s for s in combinations(items, k) if support(s) >= min_support]
        for item_set in item_sets:
            item_set = frozenset(item_set)
            for i in range(1, len(item_set)):
                for antecedent in combinations(item_set, i):
                    antecedent = frozenset(antecedent)
                    consequent = item_set - antecedent
                    confidence = support(item_set) / support(antecedent)
                    if confidence >= min_confidence:
                        lift = confidence / support(consequent)
                        rules.append({
                            'antecedent': ','.join(antecedent),
                            'consequent': ','.join(consequent),
                            'support': support(item_set),
                            'confidence': confidence,
                            'lift': lift
                        })
                        if len(rules) >= 10:  # if we have at least 10 rules
                            break
        if len(rules) >= 10:
            break

    return pd.DataFrame(rules).sort_values('lift', ascending=False)

# K-means starting
def perform_kmeans_with_progress(df, n_clusters=3, update_interval=5):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=44, max_iter=100)

    with tqdm(total=kmeans.max_iter, desc="K-means clustering") as pbar:
        for i in range(kmeans.max_iter):  # it will run for max iter times
            kmeans.fit(df_scaled)
            pbar.update(1)
            if i % update_interval == 0:
                yield kmeans.labels_
            if kmeans.n_iter_ <= i + 1:
                break

    return kmeans.labels_

# Visualize the data
def visualize_apriori_rules(rules, top_n=10):
    top_rules = rules.head(top_n)
    fig = px.scatter_3d(
        top_rules, x="support", y="confidence", z="lift",
        color="lift", size="support",
        hover_data=["antecedent", "consequent"],
        labels={"support": "support", "confidence": "confidence", "lift": "lift"},
        title=f"Top {top_n} Association Rules"
    )
    fig.show()
    return fig

# Visualization for clusters
def visualize_kmeans_clusters(df, cluster_labels):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df)
    fig = px.scatter_3d(
        x=pca_result[:, 0], y=pca_result[:, 1], z=pca_result[:, 2],
        color=cluster_labels,
        title="Customer Cluster Visualization"
    )
    fig.show()
    return fig

def main():
    print("Gathering synthetic data...")
    df_encoded = generate_data(num_products=10, num_customers=100, num_transactions=500)
    print("Data generation complete!")
    print(f"Dataset shape: {df_encoded.shape}")

    print("Performing Apriori algorithm...")
    rules = simple_apriori(df_encoded, min_support=0.1, min_confidence=0.5)

    if not rules.empty:
        print(f"Apriori algorithm completed. Found {len(rules)} rules")
        viz = visualize_apriori_rules(rules)
        viz.write_html("apriori3d.html")
        print("Apriori rules visualization saved as 'apriori3d.html'")
    else:
        print("Error: Apriori failed")

    print("Performing K-means clustering...")
    kmeans_generator = perform_kmeans_with_progress(df_encoded, n_clusters=3, update_interval=5)
    for i, labels in enumerate(kmeans_generator):
        print(f"K-means iteration {i * 5}")
        viz = visualize_kmeans_clusters(df_encoded, labels)
        viz.write_html(f"customer_cluster_3d_step_{i}.html")
        print(f"Intermediate visualization saved as 'customer_cluster_3d_step_{i}.html'")

    final_labels = labels
    print("K-means clustering completed.")
    final_viz = visualize_kmeans_clusters(df_encoded, final_labels)
    final_viz.write_html("customer_cluster_3dfinal.html")
    print("Final customer cluster visualization saved")
    print("Analysis completed.")

if __name__ == "__main__":
    main()