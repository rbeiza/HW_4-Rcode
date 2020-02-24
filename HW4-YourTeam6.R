# Homework-4

# Out date: February 15, 2020
# Due date: February 23, 2020 at 11:59PM

# Team 6 
# Team Member-1: Melvin Zaldivar  Member's Contribution (in %) 33.3%
# Team Member-2: Raul Beiza       Member's Contribution (in %) 33.3%
# Team Member-3: Rahim Abdul      Member's Contribution (in %) 33.3%


# Intall packages and set functions that we will need for this HW
install.packages("tm")
library("tm")
install.packages ("SnowballC") #added
library(SnowballC) #added
install.packages("e1071")
library("e1071")
install.packages("gmodels")
library(gmodels)

Dictionary <- function(x){
  if( is.character(x) ){
    return(x)
  }
  stop('x is not a character vector')
}
convert_counts <- function(x){
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return(x)
}

# PART 1
# Read the raw training data and convert the "Sentiment" column to factors
tweeter_traning <- read.csv("tweeter_traning.csv", stringsAsFactors = FALSE)
tweeter_traning$Sentiment <- factor(tweeter_traning$Sentiment)


# The follwoing functions allow the user to convert all text to lowercase, remove numbers,
# stop words, punctuations, and strip blank spaces from the tweets in the "Sentiment" column
tweeter_corpus <- VCorpus(VectorSource(tweeter_traning$SentimentText)) #added V infront of Corpus
corpus_clean <- tm_map(tweeter_corpus, content.transformer(tolower)) #added content.transformer()
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stemDocuments) #added this line of code from the book
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

# Convert the data into a sparse matrix
tweeter_dtm <- DocumentTermMatrix(corpus_clean)

# From here we are going to split the data into training and testing data
tweet_train <- tweeter_traning[1:52494, ]
tweet_test <- tweeter_traning[52495:69992, ] 

tweet_dtm_train <- tweeter_dtm[1:52494, ]
tweet_dtm_test <- tweeter_dtm[52495:69992, ]

tweet_corpus_train <- corpus_clean[1:52494]
tweet_corpus_test <- corpus_clean[52495:69992]

# Check the proportions of training and testing data
prop.table(table(tweet_train$Sentiment))
prop.table(table(tweet_test$Sentiment))

# The following shows how to find the words that will help idenfity what is a negative or positive tweet
# and then convert the sparse matrix from counts to factors
tweet_dict <- Dictionary(findFreqTerms(tweet_dtm_train, 20))

tweet_dict_train <- DocumentTermMatrix(tweet_corpus_train, list(dictionary = tweet_dict))
tweet_dict_test <- DocumentTermMatrix(tweet_corpus_test, list(dictionary = tweet_dict))


tweet_dict_train <- apply(tweet_dict_train, MARGIN = 2, convert_counts)
tweet_dict_test <- apply(tweet_dict_test, MARGIN = 2, convert_counts)

# PART 2
tweet_classifier <- naiveBayes(tweet_dict_train, tweet_train$Sentiment, laplace = 1)
tweet_pred <- predict(tweet_classifier, tweet_dict_test)


CrossTable(tweet_pred, tweet_test$Sentiment, prop.chisq = FALSE, prop.t = FALSE, 
           prop.r = FALSE, dnn = c('predicted', 'actual'))

# PART 3
# Read test data
tweeter_test <- read.csv("tweeter_test.csv", stringsAsFactors = FALSE)

# Clean text data
test_corpus <- Corpus(VectorSource(tweeter_test$SentimentText))
test_corpus_clean <- tm_map(test_corpus, tolower)
test_corpus_clean <- tm_map(test_corpus_clean, removeNumbers)
test_corpus_clean <- tm_map(test_corpus_clean, removeWords, stopwords())
test_corpus_clean <- tm_map(test_corpus_clean, removePunctuation)
test_corpus_clean <- tm_map(test_corpus_clean, stripWhitespace)

# Turn data into sparse matrix
test_dtm <- DocumentTermMatrix(test_corpus_clean)

test_dict <- DocumentTermMatrix(test_corpus_clean, list(dictionary = tweet_dict))
test_dict <- apply(test_dict, MARGIN = 2, convert_counts)

# PART 4
# Prediction on test data
test_pred <- predict(tweet_classifier, test_dict)
tweeter_test_sub <- read.csv("tweeter_test_submissionFormat.csv")
tweeter_test_sub[, 2] <- test_pred
write.csv(tweeter_test_sub, "test_test_predictions.csv", row.names = FALSE)
