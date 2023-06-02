import pandas as pd
import matplotlib.pyplot as plt
import config
import seaborn as sns

"""
The purpose of this dataset was to perform text classification. 

To achieve this, scraped reviews from Amazon products were used. 

The dataset is composed of three classes;
    - Negative Reviews
    - Neutral Reviews
    - Positive Reviews. 

The dataset contains several columns of data;  
    - Sentiments, Cleaned Review, Cleaned Review Length, and Review Score.

Resource: https://www.kaggle.com/datasets/danielihenacho/amazon-reviews-dataset 
"""

data = pd.read_csv(config.DATA_FILE)
overview = data.head()
print(f"Exploring the first five rows of the data {overview}")

# get the number of rows and columns, column names, number of non-null values, and data types of each column
stats = data.info()
print(f"Getting a summary of the data {stats}") 

# check targets distribution
sentiment_distribution = data['sentiments'].value_counts()
print("check the sentiments distribution balance")
print(sentiment_distribution)


# plot sentiments distribution 
sentiment_custom_palette = ["#00204a", "#005792", "#00bbf0"]  # define a custom color palette
ax = sns.countplot(x='sentiments', data=data, palette=sentiment_custom_palette, width=0.5)

plt.xlabel('Review')  # adding the x-axis label
plt.ylabel('Frequency')  # adding the y-axis label
plt.title('Distribution of Sentiments')  # adding the plot title

plt.savefig('../outputs/sentiments_distribution.png', dpi=300, bbox_inches='tight') # save the plot as a PNG file
plt.show()


# plot review_score distribution 
score_custom_palette = ["#a5bdfd", "#66bfbf", "#00bbf0", "#005792", "#00204a"] # define a custom color palette
ax = sns.countplot(x='review_score', data=data,  width=0.8, palette=score_custom_palette)

plt.xlabel('Review Score')  # adding the x-axis label
plt.ylabel('Frequency')  # adding the y-axis label
plt.title('Distribution of Review Scores')  # adding the plot title

plt.savefig('../outputs/review_score_distribution.png', dpi=300, bbox_inches='tight') # define a custom color palette
plt.show()


# plot the distribution of the review length 
sns.kdeplot(data['cleaned_review_length'], fill=True, color='#005792')

plt.xlim(-50, 250)  # set the x-axis limits
plt.xlabel('Review Length')   # adding the x-axis label
plt.title('Density Plot of Text Length')  # adding the plot title

plt.savefig('../outputs/density_plot_text_length.png', dpi=300, bbox_inches='tight')  # save the plot as a PNG file
plt.show()
