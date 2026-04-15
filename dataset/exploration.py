import pandas as pd
import os

# Go up from dataset/ to webadas/, then into kuairand/KuaiRand-Pure/data/
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # dataset/
    '..', 'kuairand', 'KuaiRand-Pure', 'data'   # ../kuairand/KuaiRand-Pure/data
)

print(f"Loading data from: {os.path.normpath(BASE_DIR)}")

def path(filename):
    return os.path.join(BASE_DIR, filename)

df_rand                   = pd.read_csv(path("log_random_4_22_to_5_08_pure.csv"), nrows=1)
df1                       = pd.read_csv(path("log_standard_4_08_to_4_21_pure.csv"), nrows=1)
df2                       = pd.read_csv(path("log_standard_4_22_to_5_08_pure.csv"), nrows=1)
user_features             = pd.read_csv(path("user_features_pure.csv"))
video_features_basic      = pd.read_csv(path("video_features_basic_pure.csv"), nrows=1)
video_features_statistics = pd.read_csv(path("video_features_statistic_pure.csv"), nrows=1)


print("=== user_features_basic colums: ",       user_features.columns.tolist())
print("=== video_features_basic columns:",      video_features_basic.columns.tolist())
print("=== video_features_statistics columns:", video_features_statistics.columns.tolist())


