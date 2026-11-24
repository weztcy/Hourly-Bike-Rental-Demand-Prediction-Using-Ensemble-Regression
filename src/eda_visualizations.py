import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_eda_visualizations():
    os.makedirs("visualizations", exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = sns.color_palette("muted")
    
    df = pd.read_csv("Bike_Sharing.csv")
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["timestamp"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    
    # Season mapping
    season_map = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
    df["season_name"] = df["season"].map(season_map)
    
    # Weather situation mapping
    weather_map = {
        1: "Clear / Few Clouds",
        2: "Mist / Cloudy",
        3: "Light Rain / Snow",
        4: "Heavy Rain / Snow"
    }
    df["weather_name"] = df["weathersit"].map(weather_map)
    
    # Weekday mapping (0: Sunday, 1: Monday, ..., 6: Saturday as per UCI Bike dataset standard)
    day_map = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
    df["day_name"] = df["weekday"].map(day_map)
    
    # Month mapping
    month_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    df["month_name"] = df["mnth"].map(month_map)

    # 1. Target Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Raw cnt
    sns.histplot(df["cnt"], kde=True, ax=axes[0], color="#2b5c8f", bins=40)
    mean_val = df["cnt"].mean()
    median_val = df["cnt"].median()
    skew_val = df["cnt"].skew()
    axes[0].axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
    axes[0].axvline(median_val, color="green", linestyle="-.", linewidth=1.5, label=f"Median: {median_val:.1f}")
    axes[0].set_title(f"Raw Target Distribution ('cnt')\nSkewness: {skew_val:.2f}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Hourly Rental Count (bikes/hour)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].legend(frameon=True)
    
    # Log1p cnt
    sns.histplot(np.log1p(df["cnt"]), kde=True, ax=axes[1], color="#17a2b8", bins=40)
    log_skew = np.log1p(df["cnt"]).skew()
    axes[1].set_title(f"Log-Transformed Target Distribution (log1p('cnt'))\nSkewness: {log_skew:.2f} (More Normal)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("log(1 + cnt)", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    
    plt.suptitle("Distribution of Hourly Bike Rental Demand ('cnt')", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("visualizations/target_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved visualizations/target_distribution.png")

    # 2. Hourly Demand Pattern
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=300)
    
    # Overall Hourly Average & Percentiles
    hourly_stats = df.groupby("hr")["cnt"].agg(["mean", "median", lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)])
    hourly_stats.columns = ["mean", "median", "p25", "p75"]
    
    axes[0].plot(hourly_stats.index, hourly_stats["mean"], color="#1f77b4", marker="o", linewidth=2.5, label="Mean Rental")
    axes[0].plot(hourly_stats.index, hourly_stats["median"], color="#2ca02c", linestyle="--", marker="s", label="Median Rental")
    axes[0].fill_between(hourly_stats.index, hourly_stats["p25"], hourly_stats["p75"], alpha=0.2, color="#1f77b4", label="IQR (25th - 75th percentile)")
    axes[0].set_title("Overall Average Hourly Bike Rental Demand", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Hour of Day (0–23)", fontsize=11)
    axes[0].set_ylabel("Rental Count (bikes/hour)", fontsize=11)
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend(frameon=True)
    axes[0].annotate("Morning Rush (8 AM)\n~359 bikes/hr", xy=(8, 359), xytext=(3, 450),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                    fontsize=9, fontweight="bold")
    axes[0].annotate("Evening Rush (5–6 PM)\n~461 bikes/hr", xy=(17, 461), xytext=(18, 550),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                    fontsize=9, fontweight="bold")
    
    # Hourly Demand by Working Day vs Non-Working Day
    sns.lineplot(data=df, x="hr", y="cnt", hue="workingday", palette={0: "#e66101", 1: "#5e3c99"},
                 ax=axes[1], errorbar=("ci", 95), marker="o", linewidth=2)
    axes[1].set_title("Hourly Demand: Working Day vs Weekend/Holiday", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Hour of Day (0–23)", fontsize=11)
    axes[1].set_ylabel("Rental Count (bikes/hour)", fontsize=11)
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].legend(["Weekend / Holiday (0)", "Working Day (1)"], title="Day Type", frameon=True)
    axes[1].annotate("Leisure Midday Peak\n(12–4 PM on weekends)", xy=(13, 370), xytext=(9, 450),
                    arrowprops=dict(facecolor='#e66101', shrink=0.05, width=1, headwidth=6),
                    fontsize=9, fontweight="bold", color='#b34800')

    plt.suptitle("Hourly Dynamics of Bike Rental Demand", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("visualizations/hourly_demand.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved visualizations/hourly_demand.png")

    # 3. Demand by Calendar Factors (Day of Week, Month, Season)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    # Day of Week
    day_order = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    sns.barplot(data=df, x="day_name", y="cnt", order=day_order, ax=axes[0], palette="Blues_d", errorbar=("ci", 95))
    axes[0].set_title("Mean Demand by Day of Week", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Day of Week", fontsize=11)
    axes[0].set_ylabel("Mean Rental Count (bikes/hour)", fontsize=11)
    
    # Month
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sns.barplot(data=df, x="month_name", y="cnt", order=month_order, ax=axes[1], palette="coolwarm", errorbar=("ci", 95))
    axes[1].set_title("Mean Demand by Month", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Month", fontsize=11)
    axes[1].set_ylabel("Mean Rental Count (bikes/hour)", fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Season
    season_order = ["Winter", "Spring", "Summer", "Fall"]
    sns.barplot(data=df, x="season_name", y="cnt", order=season_order, ax=axes[2], palette="viridis", errorbar=("ci", 95))
    axes[2].set_title("Mean Demand by Season", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Season", fontsize=11)
    axes[2].set_ylabel("Mean Rental Count (bikes/hour)", fontsize=11)

    plt.suptitle("Demand Patterns Across Calendar and Seasonal Dimensions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("visualizations/demand_by_calendar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved visualizations/demand_by_calendar.png")

    # 4. Demand by Weather
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    
    # Weather situation category
    weather_order = ["Clear / Few Clouds", "Mist / Cloudy", "Light Rain / Snow", "Heavy Rain / Snow"]
    weather_counts = df["weather_name"].value_counts()
    sns.boxplot(data=df, x="weather_name", y="cnt", order=weather_order, ax=axes[0, 0], palette="Set2")
    axes[0, 0].set_title("Demand by Weather Situation Category", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Weather Condition", fontsize=11)
    axes[0, 0].set_ylabel("Hourly Rental Count", fontsize=11)
    axes[0, 0].tick_params(axis='x', rotation=15)
    
    # Temperature vs Demand
    # Unnormalize temp: t_celsius = temp * (39 - (-8)) + (-8) = temp * 47 - 8
    df["temp_celsius"] = df["temp"] * 47 - 8
    sns.regplot(data=df.sample(2500, random_state=42), x="temp_celsius", y="cnt",
                scatter_kws={"alpha": 0.2, "color": "#d95f02", "s": 15},
                line_kws={"color": "darkred", "linewidth": 2}, ax=axes[0, 1])
    axes[0, 1].set_title("Demand vs Ambient Temperature (°C)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Temperature (°C)", fontsize=11)
    axes[0, 1].set_ylabel("Hourly Rental Count", fontsize=11)
    
    # Humidity vs Demand
    df["humidity_pct"] = df["hum"] * 100
    sns.regplot(data=df.sample(2500, random_state=42), x="humidity_pct", y="cnt",
                scatter_kws={"alpha": 0.2, "color": "#7570b3", "s": 15},
                line_kws={"color": "navy", "linewidth": 2}, ax=axes[1, 0])
    axes[1, 0].set_title("Demand vs Relative Humidity (%)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Humidity (%)", fontsize=11)
    axes[1, 0].set_ylabel("Hourly Rental Count", fontsize=11)
    
    # Windspeed vs Demand
    df["windspeed_kmh"] = df["windspeed"] * 67
    sns.regplot(data=df.sample(2500, random_state=42), x="windspeed_kmh", y="cnt",
                scatter_kws={"alpha": 0.2, "color": "#1b9e77", "s": 15},
                line_kws={"color": "darkgreen", "linewidth": 2}, ax=axes[1, 1])
    axes[1, 1].set_title("Demand vs Wind Speed (km/h)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Wind Speed (km/h)", fontsize=11)
    axes[1, 1].set_ylabel("Hourly Rental Count", fontsize=11)

    plt.suptitle("Impact of Weather Conditions on Hourly Rental Demand", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("visualizations/demand_by_weather.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved visualizations/demand_by_weather.png")

    # 5. Time Trend (Weekly Aggregation Over 2 Years)
    fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
    weekly = df.set_index("timestamp").resample("W-MON")["cnt"].agg(["sum", "mean", "count"])
    # Filter full weeks
    weekly = weekly[weekly["count"] > 50]
    
    ax.plot(weekly.index, weekly["mean"], color="#005b96", linewidth=2.2, marker="o", markersize=4, label="Weekly Mean Demand (bikes/hour)")
    # Mark split lines
    train_end = pd.to_datetime("2012-05-27 01:00:00")
    val_end = pd.to_datetime("2012-09-12 15:00:00")
    
    ax.axvline(train_end, color="#d95f02", linestyle="--", linewidth=2, label="Train / Val Split (May 27, 2012)")
    ax.axvline(val_end, color="#7570b3", linestyle="--", linewidth=2, label="Val / Test Split (Sep 12, 2012)")
    
    ax.axvspan(df["timestamp"].min(), train_end, alpha=0.08, color="blue", label="Training Partition (70%)")
    ax.axvspan(train_end, val_end, alpha=0.1, color="orange", label="Validation Partition (15%)")
    ax.axvspan(val_end, df["timestamp"].max(), alpha=0.1, color="purple", label="Test Partition (15%)")
    
    ax.set_title("Two-Year Demand Trend with Chronological Evaluation Splits (2011–2012)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Average Hourly Rentals per Week", fontsize=11)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.savefig("visualizations/time_trend.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved visualizations/time_trend.png")

    # 6. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    corr_cols = ["temp", "atemp", "hum", "windspeed", "hr", "weekday", "mnth", "yr", "workingday", "holiday", "weathersit", "cnt"]
    corr_matrix = df[corr_cols].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Correlation Matrix of Numeric & Encoded Features\n(Note: Correlation indicates statistical association, not causation)",
                 fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig("visualizations/correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved visualizations/correlation_heatmap.png")

if __name__ == "__main__":
    generate_eda_visualizations()
