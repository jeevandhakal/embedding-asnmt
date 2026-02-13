from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from huggingface_hub import login

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
import csv
import umap





# token to login huggingface
login("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")

project_path = "./"

# Read attendees and their responses from a CSV file
attendees_map = {}
with open(project_path+ 'classmates.csv', newline='') as csvfile:
    attendees = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(attendees)  # Skip the header row
    for row in attendees:
        name, paragraph = row
        attendees_map[paragraph] = name

# Generate sentence embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
paragraphs = list(attendees_map.keys())
embeddings = model.encode(paragraphs)



# Create a dictionary to store embeddings for each person
person_embeddings = {attendees_map[paragraph]: embedding for paragraph, embedding in zip(paragraphs, embeddings)}

# save dictionary as csv
embeddings_df= pd.DataFrame(person_embeddings)
embeddings_df.to_csv('embeddings.csv', index=False)

# Reducing dimensionality of embedding data, scaling to coordinate domain/range
scaler = StandardScaler()
scaled_data = scaler.fit_transform(list(person_embeddings.values()))

for i in range(5):

    reducer = umap.UMAP(random_state=i)
    reduced_data = reducer.fit_transform(scaled_data)

    # Creating lists of coordinates with accompanying labels
    x = [row[0] for row in reduced_data]
    y = [row[1] for row in reduced_data]
    label = list(person_embeddings.keys())

    # Plotting and annotating data points
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y)
    for j, name in enumerate(label):
        plt.annotate(name, (x[j], y[j]), fontsize="3")

    # Clean-up and Export
    plt.axis('off')
    plt.savefig(project_path + 'a_visualization_' + str(i) + '.png', dpi=800)
    plt.close()



# We use cosine similarity as the basis for the ranking in 384D
sim_matrix_384d = cosine_similarity(scaled_data)


def objective(trial):
    # Hyperparameters to tune
    n_neighbors = trial.suggest_int('n_neighbors', 2, len(embeddings) - 1)
    min_dist = trial.suggest_float('min_dist', 0.0, 0.5)


    # We use metric='cosine' to match our embedding similarity
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=42,
        n_components=2
    )
    embedding_2d = reducer.fit_transform(scaled_data)#embeddings)

    # Compute 2D Euclidean distances
    dist_matrix_2d = squareform(pdist(embedding_2d, metric='euclidean'))

    # Calculate Average Spearman Rank Correlation
    correlations = []
    num_students = embeddings.shape[0]

    for i in range(num_students):

        high_d_scores = sim_matrix_384d[i]
        low_d_scores = -dist_matrix_2d[i]
        rho, _ = spearmanr(high_d_scores, low_d_scores)
        correlations.append(rho)

    return np.mean(correlations)


# Run the study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print(f"Best Score: {study.best_value}")
print(f"Best Params: {study.best_params}")

correlation_df = pd.DataFrame(columns=['random_seed', 'mean_rho'])
correlation=[]

for i in range(5):

    best_reducer = umap.UMAP(
        **study.best_params,
        metric='cosine',
        random_state=i
    )

    reduced_data = best_reducer.fit_transform(scaled_data)

    # Creating lists of coordinates with accompanying labels
    x = [row[0] for row in reduced_data]
    y = [row[1] for row in reduced_data]
    label = list(person_embeddings.keys())

    # Plotting and annotating data points
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y)
    for j, name in enumerate(label):
        plt.annotate(name, (x[j], y[j]), fontsize="3")



    # Clean-up and Export
    plt.axis('off')
    plt.savefig(project_path + 'a_visualization_best_param'+ str(i) + '.png', dpi=800)
    plt.close()

    # Compute 2D Euclidean distances for this SPECIFIC seed
    dist_matrix_2d = squareform(pdist(reduced_data, metric='euclidean'))

    # Calculate the average Spearman correlation (ρ) for THIS version
    actual_correlations = []
    for k in range(len(scaled_data)):

        rho, _ = spearmanr(sim_matrix_384d[k], -dist_matrix_2d[k])
        actual_correlations.append(rho)

    current_rho = np.mean(actual_correlations)

    print(f"The Spearman Rho for this seed is: {current_rho}")
    new_row_data = {'random_seed':i, 'mean_rho': current_rho}
    correlation.append(new_row_data)

correlation_df = pd.DataFrame.from_records(correlation)
correlation_df.to_csv('rho_per_seed.csv', index=False)

