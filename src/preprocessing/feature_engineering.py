import numpy as np


# According to the previous detailed EDA, I propose the following feature engineering:

# 1. **Distance_To_Hydrology**: We want to know the actual distance to water, by combining the horizontal and vertical distances.
# We use the Euclidean distance to find it
# 2. **Log_Horiz_Hydrology, Log_Horiz_Roadways, Log_Horiz_Fire**: We log transform these distances because they are extremely
# right skewed.
# 3. **Aspect_sin, Aspect_cos**: We need to convert the Aspect degrees in sin and cos, because it is Circular
# and 0 degrees = 360 degrees, but the model would think they are very far apart.
# 4. **Hillshade_Mean** = The hillshades of 9am, noon and 3pm all measure sun exposure. Let's average them to capture an
# overall sun exposure thoughout the day.
# 5. **Elevation_Wilderness_Area** (4 in total): Since a specific elevation can mean "very high" in one area and
# "normal/very low" in other areas, we create this feature so the models can better understand area-specific elevation patterns.
# 6. **Close_To_Water**: We want to know if it is close to water (1) or not (0). We will consider "close" to be within 100m.
# 7. **High_Altitude**: If elevation is greater than 3200m (1) is considered to be high altitude, otherwise not (0).


def augment_features(df):
    df = df.copy()

    df["Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    df["Log_Horiz_Hydrology"] = np.log1p(df["Horizontal_Distance_To_Hydrology"])
    df["Log_Horiz_Roadways"] = np.log1p(df["Horizontal_Distance_To_Roadways"])
    df["Log_Horiz_Fire"] = np.log1p(df["Horizontal_Distance_To_Fire_Points"])

    df["Aspect_sin"] = np.sin(df["Aspect"] * np.pi / 180)
    df["Aspect_cos"] = np.cos(df["Aspect"] * np.pi / 180)

    df["Hillshade_Mean"] = (
        df["Hillshade_9am"] + df["Hillshade_Noon"] + df["Hillshade_3pm"]
    ) / 3

    df[f"Elevation_Wilderness_Area1"] = df["Elevation"] * df[f"Wilderness_Area1"]
    df[f"Elevation_Wilderness_Area2"] = df["Elevation"] * df[f"Wilderness_Area2"]
    df[f"Elevation_Wilderness_Area3"] = df["Elevation"] * df[f"Wilderness_Area3"]
    df[f"Elevation_Wilderness_Area4"] = df["Elevation"] * df[f"Wilderness_Area4"]

    df["Close_To_Water"] = (df["Horizontal_Distance_To_Hydrology"] < 100).astype(int)
    df["High_Altitude"] = (df["Elevation"] > 3200).astype(int)

    return df


def remove_base_features(df):
    df = df.copy()
    features_to_remove = [
        "Aspect",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
    ]
    df.drop(columns=features_to_remove, inplace=True)
    return df


def get_final_features(df):
    df = df.copy()
    df = augment_features(df)
    df = remove_base_features(df)
    return df
