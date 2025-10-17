import pandas as pd
import matplotlib.pyplot as plt
#create a samll data frame
df=pd.DataFrame({
    'x':[1,2,3,4,5],
    'y':[10,15,13,17,20]
})
df.to_csv("reports/figures/test.csv",index=False)
print("CSV saved at reprots/figures/test.csv")
#plot
plt.plot(df['x'],df['y'],marker='o')
plt.title("Test Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.savefig("reports/figures/test_plot.png")
print("Plot saved at reports/figures/test_plot.png")
print("Hello from dev branch!")