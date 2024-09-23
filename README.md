# Customer Segmentation Using Clustering Algorithms

## Project Overview

This project performs customer segmentation using clustering algorithms on the **Customer Personality Analysis** dataset from Kaggle. The goal is to identify distinct customer groups based on their purchasing behavior and demographics, enabling businesses to tailor marketing strategies, improve customer satisfaction, and enhance profitability.

**Key Features:**

- **Data Preprocessing and Cleaning:** Handling missing values, converting data types, and removing outliers to ensure data quality.
- **Exploratory Data Analysis (EDA):** Understanding data distributions and relationships through statistical summaries and visualizations.
- **Feature Engineering:** Creating new features and simplifying categories to enhance the clustering process.
- **Dimensionality Reduction using Autoencoders:** Leveraging neural networks to reduce dimensionality while capturing nonlinear relationships.
- **Clustering Analysis:** Applying K-Means, Hierarchical Clustering, and Gaussian Mixture Models to identify customer segments.
- **Evaluation of Clustering Performance:** Using Silhouette Score and Calinski-Harabasz Index to assess cluster quality.
- **Visualization of Customer Segments:** Visualizing clusters to interpret and understand the customer segments.
- **Interpretation and Business Insights:** Providing actionable recommendations based on cluster characteristics.

## Dataset

The **Customer Personality Analysis** dataset contains detailed information about customers, including:

- **Demographics:** Age, education level, marital status, income, number of children and teenagers at home.
- **Purchasing Behavior:** Amount spent on different product categories over the last two years.
- **Promotion Response:** Response to various marketing campaigns.
- **Engagement:** Number of purchases through different channels and web visits.

**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

## Libraries and Tools Used

- **Python:** Programming language used for the analysis.
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical computations.
- **Matplotlib and Seaborn:** For data visualization.
- **TensorFlow and Keras:** For building and training the Autoencoder used in dimensionality reduction.
- **scikit-learn:** For clustering algorithms and evaluation metrics.

### Why These Libraries Were Used

- **TensorFlow and Keras:** We used TensorFlow and Keras to build an Autoencoder for dimensionality reduction. Autoencoders are neural networks capable of learning efficient representations of data (encoding), capturing nonlinear relationships that linear methods like PCA might miss. This helps in reducing the dimensionality of the dataset while preserving important information, improving the performance of clustering algorithms.

- **scikit-learn:** Provides efficient implementations of clustering algorithms (K-Means, Agglomerative Clustering, Gaussian Mixture Models) and evaluation metrics (Silhouette Score, Calinski-Harabasz Index) essential for analyzing the clusters formed.


## Data Understanding and Preprocessing

- **Data Loading:** The dataset is loaded and examined for structure and content.
- **Handling Missing Values:** Missing values in the `Income` column are handled by imputing with the median income.
- **Data Type Conversion:** The `Dt_Customer` column is converted to datetime format for proper date manipulations.
- **Exploratory Data Analysis (EDA):** Statistical summaries and visualizations are performed to understand the distributions and relationships in the data.

## Feature Engineering

- **Creating New Features:**
  - **Age:** Calculated from the `Year_Birth` column.
  - **Total Children:** Sum of `Kidhome` and `Teenhome`.
  - **Total Spending:** Total amount spent across different product categories.
  - **Total Purchases:** Total number of purchases across different channels.
  - **Customer Tenure:** Number of days since the customer enrolled with the company.

- **Simplifying Categories:**
  - **Education Levels:** Simplified into 'Undergraduate', 'Graduate', and 'Postgraduate'.
  - **Marital Status:** Simplified into 'Married' and 'Single'.

- **Encoding Categorical Variables:** One-hot encoding is applied to the simplified categorical variables to convert them into numerical format suitable for modeling.

- **Removing Outliers:** Outliers in `Income` and `Age` are removed using the Interquartile Range (IQR) method to improve the quality of the clustering.

## Dimensionality Reduction with Autoencoder

To capture nonlinear relationships and reduce dimensionality, an Autoencoder neural network was used.

- **Normalization:** Features are normalized using TensorFlow's `Normalization` layer to prepare data for neural network training.

- **Building the Autoencoder:** The Autoencoder consists of an encoder that compresses the data into a lower-dimensional bottleneck layer and a decoder that reconstructs the original data from the compressed representation.

- **Training the Autoencoder:** The model is trained to minimize the reconstruction loss (mean squared error), effectively learning a compressed representation of the data.

- **Extracting Bottleneck Features:** The encoder part of the Autoencoder is used to transform the data into a lower-dimensional space, which is then used for clustering.

## Clustering Analysis

Multiple clustering algorithms were applied to the compressed data:

- **Determining Optimal Clusters:** The Elbow Method and Silhouette Analysis were used to determine that an optimal number of clusters is around 4 or 5.

- **K-Means Clustering:** Applied with 4 clusters based on the analysis.

- **Agglomerative Clustering:** A hierarchical clustering method was also applied for comparison.

- **Gaussian Mixture Models (GMM):** Used to capture the probabilistic distribution of the data.

## Evaluation of Clustering Performance

Clustering performance was evaluated using:

- **Silhouette Score:** Measures how similar an object is to its own cluster compared to other clusters. Higher scores indicate better-defined clusters.

- **Calinski-Harabasz Index:** Evaluates cluster dispersion; higher values indicate well-separated clusters.

**Findings:**

- **Silhouette Scores:**
  - K-Means Silhouette Score: *[insert score]*
  - Hierarchical Clustering Silhouette Score: *[insert score]*
  - GMM Silhouette Score: *[insert score]*

- **Calinski-Harabasz Index:**
  - K-Means Calinski-Harabasz Score: *[insert score]*
  - Hierarchical Clustering Calinski-Harabasz Score: *[insert score]*
  - GMM Calinski-Harabasz Score: *[insert score]*

- K-Means and GMM showed similar performance based on the evaluation metrics.

## Visualization of Customer Segments

- **Cluster Visualization:** Clusters were visualized using scatter plots of the compressed features, showing how customers are grouped in the lower-dimensional space.

- **Cluster Profiling:** Each cluster was profiled by calculating the mean values of features within the cluster, helping in understanding the characteristics of each customer segment.

### Business Recommendations

- **Personalized Marketing:** Tailor campaigns based on customer segments to increase relevance and effectiveness.

- **Channel Optimization:** Focus on preferred channels for each segment to improve engagement.

- **Product Development:** Develop or promote products that meet the specific needs and preferences of each cluster.

## Conclusion

By applying clustering algorithms enhanced with dimensionality reduction using an Autoencoder, we successfully identified distinct customer segments. The use of TensorFlow and Keras allowed us to capture nonlinear relationships in the data, leading to more meaningful clusters. Understanding these segments enables businesses to create targeted marketing strategies, improve customer satisfaction, and increase profitability.
