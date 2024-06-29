import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree

def association_rule_mining(dataset):
    # Load your dataset
    # Replace 'your_dataset.csv' with your actual dataset path
    df = pd.read_csv(dataset)
    # One-hot encode the features and class label
    df = pd.get_dummies(df, columns=['ID','IIR','IR','label'], prefix=['ID', 'IIR', 'IR', 'CL'])

    print(df.head())
    df = df.astype(bool)

    # Apply the Apriori algorithm
    frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

    # Filter rules to focus on those involving the class label
    rules_class_label = rules[rules['consequents'].apply(lambda x: any(['CL_0' in item for item in x]))]

    # Display the rules involving the class label
    print(rules_class_label)



def decision_tree(dataset):
    df = pd.read_csv(dataset)

    # Split the data into features (X) and target variable (y)
    X = df[['ID', 'IIR', 'IR']]
    X = pd.get_dummies(X, columns=['ID', 'IIR', 'IR'])

    y = df['label']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

    # Create a Decision Tree classifier
    dt_model = DecisionTreeClassifier(random_state=16)

    # Train the model
    dt_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = dt_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Function to get the probability of each leaf node
    def get_leaf_probabilities(tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        print("feature_names:", feature_name)

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                print(f"{indent}if {name} <= {threshold:.2f}:")
                recurse(tree_.children_left[node], depth + 1)
                print(f"{indent}else:  # if {name} > {threshold:.2f}")
                recurse(tree_.children_right[node], depth + 1)
            else:
                value = tree_.value[node]
                total = value.sum()
                probabilities = value / total
                print(f"{indent}leaf node, class distribution: {value}, probabilities: {probabilities}")

        recurse(0, 1)

    # Get feature names
    feature_names = X.columns

    # Print leaf probabilities
    get_leaf_probabilities(dt_model, feature_names)

    tree_rules = export_text(dt_model, feature_names=list(X.columns))
    print(tree_rules)
