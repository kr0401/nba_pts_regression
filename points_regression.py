from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# scrape total stats
url = 'https://www.basketball-reference.com/leagues/NBA_2020_totals.html#totals_stats::pts'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

stats = soup.find_all('table')

# convert table to dataframe
data = pd.read_html(str(stats))
points_df = pd.DataFrame(data[0])
points_df['PTS'] = pd.to_numeric(points_df['PTS'], errors='coerce')
points_df['FGA'] = pd.to_numeric(points_df['FGA'], errors='coerce')
points_df['FG%'] = pd.to_numeric(points_df['FG%'], errors='coerce')
points_df['MP'] = pd.to_numeric(points_df['MP'], errors='coerce')
points_df['eFG%'] = pd.to_numeric(points_df['eFG%'], errors='coerce')
points_df['3PA'] = pd.to_numeric(points_df['3PA'], errors='coerce')
points_df['2PA'] = pd.to_numeric(points_df['2PA'], errors='coerce')
points_df['FTA'] = pd.to_numeric(points_df['FTA'], errors='coerce')

points_df = points_df.sort_values(by=['PTS'], ascending=False)

# data needed from dataframe
points_df = points_df.head(50).reset_index()
points = points_df['PTS']
fga = points_df['FGA']
fgper = points_df['3PA']
fta = points_df['FTA']

points = np.array(points)
fga = np.array(fga)
fgper = np.array(fgper)
fta = np.array(fta)
points = points.reshape(-1,1)
fga = fga.reshape(-1,1)
fgper = fgper.reshape(-1,1)
fta = fta.reshape(-1,1)

lsfit = LinearRegression()
lsfit.fit(fga, points)
y_pred = lsfit.predict(fga)
r2_fga = r2_score(points, y_pred)

lsfit2 = LinearRegression()
lsfit2.fit(fgper, points)
y_pred2 = lsfit2.predict(fgper)
r2_fgper = r2_score(points, y_pred2)

lsfit3 = LinearRegression()
lsfit3.fit(fta, points)
y_pred3 = lsfit3.predict(fta)
r2_fta = r2_score(points, y_pred3)

fig, ax = plt.subplots(figsize=(15, 6))

ax1 = plt.subplot(1, 3, 1)
ax1.set_title("FGA as a predictor of scoring")
ax1.scatter(fga, points, facecolor='None', edgecolor='black')
ax1.plot(fga, y_pred, color='red')
ax1.text(0.1, 0.9, "R-squared = " + str('%.4f' % r2_fga), va='center', ha='left', transform=ax1.transAxes)
ax1.set_xlabel("FGA")
ax1.set_ylabel("Points")

ax2 = plt.subplot(1, 3, 2)
ax2.set_title("FG% as a predictor of scoring")
ax2.scatter(fgper, points, facecolor='None', edgecolor='black')
ax2.plot(fgper, y_pred2, color='red')
ax2.text(0.1, 0.9, "R-squared = " + str('%.4f' % r2_fgper), va='center', ha='left', transform=ax2.transAxes)
ax2.set_xlabel("FG%")
ax2.set_ylabel("Points")

ax3 = plt.subplot(1, 3, 3)
ax3.set_title("FTA as a predictor of scoring")
ax3.scatter(fta, points, facecolor='None', edgecolor='black')
ax3.plot(fta, y_pred3, color='red')
ax3.text(0.1, 0.9, "R-squared = " + str('%.4f' % r2_fta), va='center', ha='left', transform=ax3.transAxes)
ax3.set_xlabel("FTA")
ax3.set_ylabel("Points")

fig.tight_layout(w_pad=3.0)
plt.show()

