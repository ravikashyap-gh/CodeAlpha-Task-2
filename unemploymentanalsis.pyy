import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# extracting dataset
with zipfile.ZipFile("task2unemploment.zip", "r") as zip_ref:
    zip_ref.extractall("data")

# loading data
df = pd.read_csv("data/Unemployment in India.csv")
covid_df = pd.read_csv("data/Unemployment_Rate_upto_11_2020.csv")

# cleaning data
df.columns = df.columns.str.strip()
covid_df.columns = covid_df.columns.str.strip()

df.dropna(inplace=True)
covid_df.dropna(inplace=True)

# converting date column
df["Date"] = pd.to_datetime(df["Date"])
covid_df["Date"] = pd.to_datetime(covid_df["Date"])

print("Shape of Dataset:", df.shape)

# basic statistics
print("\nAverage Unemployment Rate:",
      round(df["Estimated Unemployment Rate (%)"].mean(), 2))

print("Highest Unemployment Rate:",
      df["Estimated Unemployment Rate (%)"].max())

print("Lowest Unemployment Rate:",
      df["Estimated Unemployment Rate (%)"].min())

# state wise unemployment
state_data = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(14,6))
state_data.sort_values().plot(kind="bar")
plt.title("State Wise Average Unemployment Rate")
plt.xlabel("States")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# unemployment trend over time
trend = df.groupby("Date")["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(12,6))
trend.plot()
plt.title("Unemployment Trend")
plt.ylabel("Rate (%)")
plt.show()

# rural and urban comparison
plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="Area",
    y="Estimated Unemployment Rate (%)"
)
plt.title("Rural vs Urban Unemployment")
plt.show()

# covid impact
plt.figure(figsize=(12,6))
sns.lineplot(
    data=covid_df,
    x="Date",
    y="Estimated Unemployment Rate (%)"
)
plt.title("Impact of COVID-19 on Unemployment")
plt.show()

# monthly trend
df["Month"] = df["Date"].dt.month

monthly = df.groupby("Month")[
    "Estimated Unemployment Rate (%)"
].mean()

plt.figure(figsize=(10,5))
monthly.plot(marker="o")
plt.title("Monthly Unemployment Trend")
plt.show()

# correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True
)
plt.title("Correlation Matrix")
plt.show()

# top 10 states
top_states = state_data.sort_values(
    ascending=False
).head(10)

plt.figure(figsize=(10,5))
top_states.plot(kind="bar")
plt.title("Top 10 States with Highest Unemployment")
plt.ylabel("Rate (%)")
plt.show()

print("\nState with Highest Unemployment:",
      state_data.idxmax())

print("State with Lowest Unemployment:",
      state_data.idxmin())
