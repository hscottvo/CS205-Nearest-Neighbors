from ucimlrepo import fetch_ucirepo

# fetch dataset
rice_cammeo_and_osmancik = fetch_ucirepo(id=545)

# data (as pandas dataframes)
X = rice_cammeo_and_osmancik.data.features
y = rice_cammeo_and_osmancik.data.targets

y["label"] = y["Class"].apply(lambda row: 1 if row == "Cammeo" else 0)

X["label"] = y["label"].copy()
cols = "Area   Perimeter  Major_Axis_Length  Minor_Axis_Length  Eccentricity  Convex_Area    Extent  label".split()
print(cols)
cols = [cols[-1]] + cols[:-1]
for i in cols[1:]:
    minimum = X[i].min()
    maximum = X[i].max()
    val_range = maximum - minimum
    X[i] = X[i].apply(lambda row: (row - minimum) / (val_range))
X = X[cols]
X.to_csv("data/rice.txt", header=False, index=False, sep="\t")
