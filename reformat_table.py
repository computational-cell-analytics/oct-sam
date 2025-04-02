import pandas as pd

x = pd.read_excel("segmented_images.xlsx")
x = x.set_index("Layer")

xt = x.T
# xt.columns = x.index
xt.to_excel("segmented_images_reformatted.xlsx")
