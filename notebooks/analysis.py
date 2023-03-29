# -*- coding: utf-8 -*-
"""analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ojQWZuQ4gjuKa6-U-F6YrWlYIUm3hLw-
"""

import pandas as pd
import seaborn as sns

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('data/train.csv')
df=df[:20000]
print(df.keys())
cols = ['HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'ProductId', 'Time']

# Compute the correlation matrix
corr = df.corr()

# Plot the correlation matrix using seaborn
sns.heatmap(corr, annot=True, cmap='coolwarm')

import matplotlib.pyplot as plt
# Calculate the percentage of reviews with each score
score_counts = df['Score'].value_counts(normalize=True) * 100

# Plot the results
plt.bar(score_counts.index, score_counts.values)
plt.xlabel('Score')
plt.ylabel('Percentage of Reviews')
plt.title('Distribution of Review Scores')
plt.show()

# Calculate the helpfulness ratio for each review
df['HelpfulnessRatio'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']

# Group the data by score and calculate the mean helpfulness ratio for each group
grouped_data = df.groupby('Score')['HelpfulnessRatio'].mean()

# Plot the mean helpfulness ratio by score
plt.plot(grouped_data.index, grouped_data.values)
plt.xlabel('Score')
plt.ylabel('Mean Helpfulness Ratio')
plt.show()

# Calculate the length of each summary
df['SummaryLength'] = df['Summary'].str.len()

# Group the data by score and calculate the mean summary length for each group
grouped = df.groupby('Score')['SummaryLength'].mean()

# Plot the summary length by score
plt.bar(grouped.index, grouped.values)
plt.xlabel('Score')
plt.ylabel('Mean Summary Length')
plt.show()

# Calculate the length of each summary
df['TextLength'] = df['Text'].str.len()

# Group the data by score and calculate the mean summary length for each group
grouped = df.groupby('Score')['TextLength'].mean()

# Plot the summary length by score
plt.bar(grouped.index, grouped.values)
plt.xlabel('Score')
plt.ylabel('Mean Text Length')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
# Plot the average score over time for a specific product
product_id = 'B003EYVXV4'
df_product = df[df['ProductId'] == product_id]
# Convert the "ReviewId" column to a datetime format
df_product['Time'] = pd.to_datetime(df_product['Time'])

# Set the "ReviewId" column as the index of the dataframe
df_product.set_index('Time', inplace=True)

# Group the data by month and calculate the mean "Score" for each month
df_monthly = df_product.resample('M').mean()

# Plot the mean score for each month
plt.plot(df_monthly['Score'])
plt.title('Change in Review Score over Time for Product ' + product_id)
plt.xlabel('Month')
plt.ylabel('Mean Score')
plt.show()

from nltk.corpus import stopwords
import csv 
# Load stop words
stop_words = set(stopwords.words('english'))

# Open CSV file
with open('data/train.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        
        # Get text and summary columns
        text = row['Text']
        summary = row['Summary']
        
        # Split text and summary into words
        text_words = text.split()
        summary_words = summary.split()
        
        # Print stop words in text and summary
        text_stop_words = [word for word in text_words if word.lower() in stop_words]
        summary_stop_words = [word for word in summary_words if word.lower() in stop_words]
        print('Text stop words:', text_stop_words)
        print('Summary stop words:', summary_stop_words)

import pandas as pd
import matplotlib.pyplot as plt

# Select the relevant features
features = ['HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time']
data = df[features]

# Create the scatter plot
plt.scatter(data['HelpfulnessNumerator'], data['Score'], label='HelpfulnessNumerator')
plt.scatter(data['HelpfulnessDenominator'], data['Score'], label='HelpfulnessDenominator')
plt.xlabel('Helpfulness')
plt.ylabel('Score')
plt.legend()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

data = df[:1000]
# Group the data by userid and calculate the average score for each group
grouped_data = data.groupby('UserId')['Score'].mean()

# Plot the average score for each userid
plt.scatter(grouped_data.index, grouped_data.values)
plt.xlabel('User ID')
plt.ylabel('Average Score')
plt.show()

import pandas as pd
import nltk
from nltk.corpus import stopwords

data=df[:10000]
# Define the stop words
stop_words = set(stopwords.words('english'))

# Create a dictionary to store the frequency of stop words by score
freq_by_score = {}

# Loop through each row in the dataset
for index, row in data.iterrows():
    # Check if the value in the 'Text' column is a string
    if isinstance(row['Text'], str):
        # Split the text into words
        words = row['Text'].lower().split()
        # Remove stop words
        words = [word for word in words if word not in stop_words]
        # Count the frequency of each stop word
        for word in words:
            if word in freq_by_score:
                freq_by_score[word][row['Score']] += 1
            else:
                freq_by_score[word] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                freq_by_score[word][row['Score']] += 1

# Sort the dictionary by the total frequency of each stop word
sorted_freq = sorted(freq_by_score.items(), key=lambda x: sum(x[1].values()), reverse=True)

# Print the top 10 stop words by score
for word, freq in sorted_freq[:20]:
    print(word)
    print('Score 1:', freq[1])
    print('Score 2:', freq[2])
    print('Score 3:', freq[3])
    print('Score 4:', freq[4])
    print('Score 5:', freq[5])
    print('Total:', sum(freq.values()))
    print()

import pandas as pd
import matplotlib.pyplot as plt

# Group the data by productid and userid, and calculate the average score
grouped_data = df.groupby(['ProductId', 'UserId']).mean()['Score'].reset_index()

# Plot the data
plt.scatter(grouped_data['UserId'], grouped_data['ProductId'], c=grouped_data['Score'], cmap='coolwarm')
plt.xlabel('User ID')
plt.ylabel('Product ID')
plt.colorbar(label='Average Score')
plt.show()

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Concatenate all the summaries into a single string
summary_text = ' '.join(df['Summary'].astype(str).tolist())

# Create the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(summary_text)

# Display the word cloud
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('data/train.csv')
df=df[:20000]

# Convert the epoch time to a datetime object
df['Time'] = pd.to_datetime(df['Time'], unit='s')

# Group the data by time and calculate the mean score
grouped = df.groupby(pd.Grouper(key='Time', freq='M')).mean()

# Create a time series plot of the scores
plt.plot(grouped.index, grouped['Score'])
plt.xlabel('Time')
plt.ylabel('Score')
plt.title('Average Score Over Time')
plt.show()