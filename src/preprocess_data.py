
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Read the train and test data
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

# handle categorical variables: convert sentiment to numerical values 
sentiment_mapping = {"negative": -1, "neutral": 0, "positive": 1}  # define the mapping dictionary

# map the sentiment values to numerical values
train_data['sentiments'] = train_data['sentiments'].replace(sentiment_mapping) 
test_data['sentiments'] = test_data['sentiments'].replace(sentiment_mapping)


def preprocess_text(text):
    text = text.lower()  # convert the text to lowercase
    text = text.strip()  # remove leading ot trailing whitespaces
    
    return text

# preprocess the text of cleaned_reviews
train_data['cleaned_review'] = train_data['cleaned_review'].apply(preprocess_text)
test_data['cleaned_review'] = test_data['cleaned_review'].apply(preprocess_text)

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english')  # create TF-IDF vectorizer

# Fit and transform the training data
train_vectorized = vectorizer.fit_transform(train_data['cleaned_review'])
train_df = pd.DataFrame(train_vectorized.toarray())
train_df['sentiments'] = train_data['sentiments']
train_df.to_csv('../input/train_preprocessed.csv', index=False)
print("train data has been processed and saved")

# Transform the test data using the fitted vectorizer
test_vectorized = vectorizer.transform(test_data['cleaned_review'])
test_df = pd.DataFrame(test_vectorized.toarray())
test_df['sentiments'] = test_data['sentiments']
test_df.to_csv('../input/test_preprocessed.csv', index=False)
print("test data has been processed and saved")


# exploring the frequency of each word in the train set
words_count = vectorizer.vocabulary_
words_count_df = pd.DataFrame(list(words_count.items()), columns=['Word', 'Count'])  # convert dictionary to DataFrame
words_count_df.to_csv('../outputs/words_count.csv', index=False)  # save DataFrame to CSV file
print("Words count has been saved.")


