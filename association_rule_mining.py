# %% import dataframe from pickle file
import pandas as pd

df = pd.read_pickle("UK.pkl")

df.head()


# %% convert dataframe to invoice-based transactional format
df_transactions = df.groupby(df["InvoiceNo"]).apply(lambda r: list(r["Description"]))


# %% apply apriori algorithm to find frequent items and association rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

te = TransactionEncoder()
te_ary = te.fit_transform(df_transactions)

df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

# %% 
from mlxtend.frequent_patterns import association_rules
rules = association_rules(frequent_itemsets, min_threshold=0.1)
rules

# %% count of frequent itemsets that have more than 1/2/3 items,
# and the frequent itemsets that has most items
length = frequent_itemsets["itemsets"].apply(len)
frequent_itemsets["length"] = length
print(f"more thant 1: {(length > 1).sum()}")
print(f"more thant 2: {(length > 2).sum()}")
print(f"more thant 3: {(length > 3).sum()}")

print(
    frequent_itemsets[length==length.max()]
)



# %% top 10 lift association rules
rules.sort_values("lift", ascending=False).head(10)


# %% scatterplot support vs confidence
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=rules["support"], y=rules["confidence"], alpha=0.5)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence")


# %% scatterplot plot vs lift
sns.scatterplot(x="support", y="lift", data=rules, alpha=0.5)
plt.xlabel("Support")
plt.ylabel("lift")
plt.title("support vs lift")

# %%
