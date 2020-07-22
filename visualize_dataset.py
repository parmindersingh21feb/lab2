# %% Import excel to dataframe
import pandas as pd

df = pd.read_excel("Online Retail.xlsx")


# %%  Show the first 10 rows
df.head(n=10)


# %% Generate descriptive statistics regardless the datatypes
df.describe(include='all')


# %% Remove all the rows with null value and generate stats again
df=df.dropna()
df.describe(include='all')


# %% Remove rows with invalid Quantity (Quantity being less than 0)
df = df[df['Quantity'] >=0]

# %% Remove rows with invalid UnitPrice (UnitPrice being less than 0)
df = df[df['UnitPrice'] >=0]


# %% Only Retain rows with 5-digit StockCode

df = df[df["StockCode"].astype("str").str.isnumeric()]         


# %% strip all description
df["description"] = df["Description"].str.strip()


# %% Generate stats again and check the number of rows
df.describe(include="all")


# %% Plot top 5 selling countries
import matplotlib.pyplot as plt
import seaborn as sns

top5_selling_countries = df["Country"].value_counts()[:5]
sns.barplot(x=top5_selling_countries.index, y=top5_selling_countries.values)
plt.xlabel("Country")
plt.ylabel("Amount")
plt.title("Top 5 Selling Countries")


# %% Plot top 20 selling products, drawing the bars vertically to save room for product description
top20_selling_products = df["Description"].value_counts()[:20]
sns.barplot(y=top20_selling_products.index, x=top20_selling_products.values)
plt.xlabel("Product")
plt.ylabel("Amount")
plt.title("Top 20 Selling Products")


# %% Focus on sales in UK
df = df[df["Country"] == "United Kingdom"]


#%% Show gross revenue by year-month
from datetime import datetime

df["YearMonth"] = df["InvoiceDate"].apply(
    lambda dt: datetime(year=dt.year, month=dt.month, day=1)
)


# %%
df["Total"] = df["UnitPrice"] = df["Quantity"]
df_year_month = df.groupby(["YearMonth"]).sum()["Total"].reset_index()

sns.lineplot(data=df_year_month, x="YearMonth", y="Total")





# %% save df in pickle format with name "uk.pk1" for next lab activity
# we are only interested in InvoiceNo, StockCode, Description coloumns
df.to_pickle("UK.pkl")

# %%
new_df = pd.read_pickle("UK.pkl")

# %%
