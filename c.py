import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# -------------------------------------------------------
# Association Rule Mining using Apriori Algorithm
# -------------------------------------------------------

# Step 1: Transaction Dataset
transactions = [
['Milk', 'Bread', 'Butter'],
['Bread', 'Butter'],
['Milk', 'Bread'],
['Milk', 'Butter'],
['Milk', 'Bread', 'Butter']

]

print("=" * 60)
print("ASSOCIATION RULE MINING USING APRIORI")
print("=" * 60)

# Step 2: Transaction Encoding
encoder = TransactionEncoder()
encoded_data = encoder.fit(transactions).transform(transactions)

df = pd.DataFrame(
encoded_data,
columns=encoder.columns_
)

print("\nTransaction Dataset:")
print(df)

# Step 3: Generate Frequent Itemsets
min_support = 0.4

frequent_itemsets = apriori(
df,
min_support=min_support,
use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Step 4: Generate Association Rules
min_confidence = 0.6

rules = association_rules(
frequent_itemsets,
metric="confidence",
min_threshold=min_confidence
)

# Step 5: Display Association Rules
print("\nAssociation Rules:")

if not rules.empty:

    result = rules[
        [
        'antecedents',
        'consequents',
        'support',
        'confidence',
        'lift'
        ]
    ].copy()

    print(result.to_string(index=False))

else:
    print("No association rules found.")

# Step 6: Display Rules in Readable Form
print("\nReadable Association Rules:")

for _, rule in rules.iterrows():

    antecedent = ", ".join(rule['antecedents'])
    consequent = ", ".join(rule['consequents'])

    print(
    f"{antecedent} --> {consequent} | "
    f"Support: {rule['support']:.2f} | "
    f"Confidence: {rule['confidence']:.2f} | "
    f"Lift: {rule['lift']:.2f}"
    )

    print("\n" + "=" * 60)
print("APRΙORI ASSOCIATION RULE MINING COMPLETED")
print("=" * 60)
