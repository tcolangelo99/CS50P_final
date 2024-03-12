import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from numpy.random import default_rng
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
import numpy as np

def check_dataset_size(data, random_seed=46):
    print()
    max_obs = 1000
    current_obs = len(data)
    if current_obs > max_obs:
        print(f"Dataset contains {current_obs} observations, which exceeds the maximum limit of {max_obs}.")
        num_obs_to_remove = current_obs - max_obs
        print(f"Removing {num_obs_to_remove} random observations...")
        # Set a fixed random seed for reproducibility
        rng = np.random.default_rng(random_seed)
        # Randomly select max_obs observations to keep
        arr_indices_to_keep = rng.choice(current_obs, size=max_obs, replace=False)
        data = data.iloc[arr_indices_to_keep]
        data.reset_index(drop=True, inplace=True) 
        print(f"Dataset size reduced to {max_obs} observations.")
        return data
    else:
        print("Dataset size is within the limit of 10 thousand observations.")
        return data


def nullcheck(data):
    print()

    missing_values = data.isnull().sum()

    if missing_values.sum() == 0:
        print("No missing values found.")
        return None
    elif missing_values.sum() > 0:
        for column_name, num_missing in missing_values.items():
            if num_missing > 0:
                null_indices = data[data[column_name].isnull()].index
                for index in null_indices:
                    
                    rowname = data.iloc[index, 0]
                    missing_column = column_name
                    print(f"Missing value for row '{rowname}' at column '{missing_column}'")
        nullchecked = data
        return nullchecked


def nullreplacer(nullchecked):
    print()
    """
    Replace null values with zeroes or mean of other observations and convert categorical columns to numerical values.
    Returns the DataFrame after null replacement and list of categorical columns.
    """
    while True:
        try:
            choice = input("Do you want to replace null values with zeroes or should they be replaced with the mean of the other observations? (Zero/Mean): ")
            choice = choice.lower()
            if choice == "mean":
                # Replace null values with the mean of other observations for each column
                nullchecked.fillna(nullchecked.mean(numeric_only=True), inplace=True)
                print("Null values replaced with the mean of other observations for each column")
                nullchecked.to_csv('Cleaned_Dataset.csv', index=False)
                break
            elif choice == "zero":
                nullchecked.fillna(0, inplace=True)
                missing_values_recheck = nullchecked.isnull().sum()
                if missing_values_recheck.sum() == 0:
                    print("Successfully set all null values to zero")
                    nullchecked.to_csv('Cleaned_Dataset.csv', index=False)
                    break
                else:
                    print(f"Failed to replace {missing_values_recheck.sum()} values. Program will exit now.")
                    sys.exit("Critical Error")
            else:
                print("Invalid input. Please enter 'Zero' or 'Mean'.")
        except ValueError:
            print("Invalid input. Please enter 'Zero' or 'Mean'.")
        except Exception as e:
            sys.exit("An error occurred:", e)
    data = nullchecked
    categorical_columns = None
    return data, categorical_columns


def categorical_replacer(data):
    print()
    categorical_columns = list(data.select_dtypes(include=['object']).columns)
    
    if len(categorical_columns) > 0:
        # Split the dataset into categorical and non-categorical parts
        categorical_data = data[categorical_columns]
        non_categorical_data = data.drop(categorical_columns, axis=1)
        categorical_data = categorical_data.astype(str)

        # Store original values of categorical variables
        original_values = {col: sorted(set(categorical_data[col])) for col in categorical_columns}

        # Convert categorical columns to numerical values
        
        mapping_dicts = {}
        for col_name, col in categorical_data.items():
            mapping_dict = mapping_dicts[col_name] = {value: index for index, value in enumerate(sorted(set(col)))}
            categorical_data[col_name] = col.map(mapping_dict)

        # Concatenate categorical and non-categorical parts back together
        data = pd.concat([non_categorical_data, categorical_data], axis=1)

        for col_name in categorical_columns:
            print(f"Found categorical variable {col_name} with value(s): ")
            original_values_list = (original_values[col_name])
            lengthtest = len(original_values_list)
            remaining = lengthtest-5
            if lengthtest > 5:
                original_values_list_shortened = original_values_list[0:5]
                original_values_list_shortened.append(f"... and {remaining} more")
                print(', '.join(original_values_list_shortened))
                print()

            elif lengthtest < 5:
                print(', '.join(original_values_list))
                print()

            else:
                print("An error has occured.")
                print()

        print("Categorical columns converted to numerical values")
        print("Transformed dataset:")
        print(data.head())
        
        data.to_csv('Cleaned_Dataset.csv', index=False)
        return data, categorical_columns, original_values
    
    else:
        print("No categorical values found")
        data.to_csv('Cleaned_Dataset.csv', index=False)
        return data, categorical_columns, {}


def scaler(data, categorical_columns):
    print()
  
    while True:
        try:
            choice = input("Do you want to scale your data? (Yes/No): ").lower()
            if choice == "no":
                return None
            elif choice == "yes":
                # Separate numerical and categorical data
                numerical_data = data.drop(columns=categorical_columns)
                converted_categorical_data = data[categorical_columns]

                # Scaling the numerical data
                scaler = StandardScaler()
                scaled_numerical_data = scaler.fit_transform(numerical_data)
                scaled_numerical_df = pd.DataFrame(scaled_numerical_data, columns=numerical_data.columns)

                data_scaled = pd.concat([scaled_numerical_df, converted_categorical_data.reset_index(drop=True)], axis=1)

                print("Successfully scaled the data. Here's the first few rows of the scaled dataframe:")
                print(tabulate(data_scaled.head(), headers='keys', tablefmt='grid'))
                return data_scaled
            else:
                print("Invalid input. Please enter 'Yes' or 'No'.")
        except ValueError as e:
            print("An error occurred:", e)


def feature_selection(data_scaled, target_variable):
    print()
    print("Performing feature selection...")

    # Extract features and target variable
    X = data_scaled.drop(columns=[target_variable])
    y = data_scaled[target_variable]

    try:

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(X, y)

        # Get the scores and feature names
        feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_})
        feature_scores = feature_scores.sort_values(by='Score', ascending=False)

        # Display the top features
        print("Top Features:")
        print(tabulate(feature_scores.head(), headers='keys', tablefmt='grid'))

        # Prompt the user to select the number of features to keep
        n_features = input("Enter the number of features to keep (press Enter to use default): ")
        if not n_features:
            n_features = len(feature_scores) // 2  # Default to only keep top features
        else:
            n_features = int(n_features)

        # Select the top k features
        selected_features = feature_scores['Feature'][:n_features].tolist()

        # Update the dataset with selected features
        selected_data = data_scaled[selected_features + [target_variable]]

        print(f"Selected {n_features} features based on feature selection.")
        return selected_data
    
    except RuntimeWarning as rw:
        print(f"There was an error: {rw}. Please use another variable")
        return None

    except Exception as e:
        print(f"There was an error: {e}")
        return None


def visualize_feature_scores(data, target_variable, selected_features, n_features):
    X = data[selected_features]
    y = data[target_variable]
    
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)

    if np.any(np.isnan(selector.scores_)) or np.any(np.isinf(selector.scores_)):
        print("Error: Feature scores contain null or infinite values.")
        return
    
    feature_scores = pd.DataFrame({'Feature': selected_features, 'Score': selector.scores_})
    feature_scores = feature_scores.sort_values(by='Score', ascending=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Score', y='Feature', data=feature_scores.head(n_features))
    plt.title(f'Top Features Importance for Predicting {target_variable}')
    plt.xlabel('F-score')
    plt.ylabel('Feature')
    plt.show()


def find_elbow(scores):
    differences = [scores[i] - scores[i-1] for i in range(1, len(scores))]
    max_diff_index = differences.index(max(differences))
    return max_diff_index + 1


def find_optimal_clusters_silhouette(data, cluster_number):
    s_scores = {}
    
    def calculate_silhouette_score(k):
        kmeans = KMeans(n_clusters=k, random_state=1).fit(data)
        labels = kmeans.predict(data)
        return silhouette_score(data, labels)
    
    # Use parallel processing to calculate silhouette scores for multiple cluster numbers
    results = Parallel(n_jobs=-1)(delayed(calculate_silhouette_score)(k) for k in range(2, cluster_number + 1))
    
    for k, score in zip(range(2, cluster_number + 1), results):
        s_scores[k] = score
        
    return max(s_scores, key=s_scores.get)


def clusterer(data_scaled, original_values, selected_features):
    print()
    data_orig = pd.read_csv('Cleaned_Dataset.csv')
    # Use a copy of the original dataset
    data_orig = data_orig.copy()
    data_selected = data_orig[selected_features]  # Select only the chosen features
    
    try:
        while True:
            clustering = input("Would you like to use k-means or k-medoids? (Means/Medoids): ").lower()
            if clustering in ["means", "medoids"]:
                break
            else:
                print("Please choose either 'means' or 'medoids'.")
        
        cluster_number = input("How many clusters would you like to test? Leave blank for default value of 10: ")

        if not cluster_number:
            cluster_number = 10
            print("Using default number of clusters: 10")
        else:
            cluster_number = int(cluster_number)
            print("Using user-specified number of clusters:", cluster_number)

        if cluster_number < 2:
            print("Number of clusters must be an integer greater than 1.")
            return

        # Store the score for k
        scores = {}

        for k in range(1, cluster_number + 1):
            kmeans = KMeans(n_clusters=k, random_state=1).fit(data_scaled)
            scores[k] = kmeans.inertia_

        # Find the elbow point
        elbow_point = find_elbow(list(scores.values()))
        while True:
            try:
                if elbow_point == cluster_number or (elbow_point < cluster_number and (scores[elbow_point] - scores[elbow_point - 1]) / scores[elbow_point - 1] < 0.1):
                    print("Elbow method result is inconclusive. Trying Silhouette")
                    optimal_clusters_silhouette = find_optimal_clusters_silhouette(data_scaled, cluster_number)
                    print("Optimal number of clusters (Silhouette method):", optimal_clusters_silhouette)
                    response = input("Would you like to use this number of clusters? (Yes/No): ").lower()
                    if response == "no":
                        cluster_number = int(input("Custom cluster number: "))
                        break
                    elif response == "yes":
                        cluster_number = optimal_clusters_silhouette
                        break
                    else:
                        print("Please respond with either 'Yes' or 'No'")
                        pass
                else:
                    print("Optimal number of clusters (Elbow method):", elbow_point)
                    response = input("Would you like to use this number of clusters? (Yes/No): ").lower()
                    if response == "no":
                        cluster_number = int(input("Custom cluster number: "))
                        break
                    elif response == "yes":
                        cluster_number = elbow_point
                        break
                    else:
                        print("Please respond with either 'Yes' or 'No'")
                        pass
            except Exception as e:
                print("An error occurred:", e)

        if clustering in ["means", "medoids"]:
            if clustering == "means":
                k_model = KMeans
            elif clustering == "medoids":
                k_model = KMedoids

            # Fit the selected clustering model
            k_clustering = k_model(n_clusters=cluster_number, random_state=1)
            k_clustering.fit(data_scaled)
            labels = k_clustering.predict(data_scaled)
            cluster_counts = pd.Series(labels).value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            print(tabulate(cluster_counts, headers='keys', tablefmt='grid'))

            
            # Calculate mean and median for each feature within each cluster group
            mean = data_selected.groupby(labels).mean()  # Use only selected features
            median = data_selected.groupby(labels).median()

            # Concatenate mean and median dataframes with proper labeling
            df_clusters = pd.concat([mean, median], axis=0)
            df_clusters.index = [f'Cluster_{i} Mean' if idx < cluster_number else f'Cluster_{i - cluster_number} Median' for idx, i in enumerate(range(2 * cluster_number))]

            print(df_clusters)


            plotcols = selected_features  # Use only selected features

            # Create subplots for each feature
            fig, axes = plt.subplots(len(plotcols), 1, figsize=(10, 5 * len(plotcols)))

            for i, column in enumerate(plotcols):
                # Create a boxplot for each feature
                if column == 'Year':
                    for j in range(cluster_number):
                        sns.boxplot(x=labels, y=data_orig[column], ax=axes[i], showfliers=False)
                else:
                    sns.boxplot(x=labels, y=data_orig[column], ax=axes[i], showfliers=False)
                
                axes[i].set_ylabel(column)
                axes[i].set_title(f'{column} by Cluster')

            plt.subplot_tool()
            plt.show()
            sys.exit()

    except ValueError as e:
        print("An error occurred:", e)
    except Exception as e:
        print("An error occurred:", e)


def main():
    print("This tool was made to clean and cluster datasets consisting mostly of numeric values. If your dataset contains phrases or is mostly comprised of strings, the tool might fail or produce inaccurate results.")
    file_path = input("Enter the path to the dataset file: ").replace('"', '')
    
    try:
        data = pd.read_csv(file_path)
        data = data.copy()
        data = check_dataset_size(data)
        
        # Convert categorical variables before handling null values
        data, categorical_columns, original_values = categorical_replacer(data)
        
        nullchecked = nullcheck(data)
        
        if nullchecked is not None:
            nullreplacer(nullchecked)

        data_scaled = scaler(data, categorical_columns)
        if data_scaled is not None:
            # Inform the user about the number of variables
            print(f"The dataset contains {len(data.columns)} variables.")
            
            # Ask if the user wants to perform feature selection
            perform_feature_selection = input("Do you want to perform feature selection? (Yes/No): ").lower()
            
            if perform_feature_selection == "yes":
                while True:
                    try:
                        target_variable = input("Enter the name of the target variable (Case sensitive!): ")
                        data_selected = feature_selection(data_scaled, target_variable)
                        
                        # Get the selected features
                        selected_features = data_selected.columns.tolist()[:-1]  # Exclude the target variable
                        
                        # Visualize feature scores
                        n_features = len(selected_features)
                        visualize_feature_scores(data_selected, target_variable, selected_features, n_features)
                        again = input("Do you want to try again with a different target variable? (Yes?/No): ").lower()
                        if again == "no":
                            # Perform clustering
                            clusterer(data_selected, original_values, selected_features)
                        else:
                            pass
                    except Exception:
                        print("Please enter a valid target variable")
            else:
                clusterer(data_scaled, original_values, [])
        else:
            print("Cleaned Dataset saved in directory")
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
    except Exception as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    main()

