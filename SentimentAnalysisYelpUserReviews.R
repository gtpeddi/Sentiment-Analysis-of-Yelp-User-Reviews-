#Sentiment Analaysis of Yelp User Reviews

library(ggplot2)
library(stringr)
library(tidyverse)
library(textstem)
library(SentimentAnalysis)
library(tidytext)
library(e1071)
library(glmnet)
library(dplyr)
library(janeaustenr)
library(tokenizers)



yelp <- read.csv("yelpRestaurantReviews_sample.csv", sep = ";", header = TRUE, stringsAsFactors = FALSE)



#Distribution Plot
ggplot(data = stars, aes(x=Var1, y=Freq))+geom_bar(stat = "identity")+ggtitle("Frequency of Stars")+labs(y="Frequency", x="Stars")+geom_text(aes(label=percentage), vjust=0)

yelp$label <- if_else(yelp$stars>=4, 1, -1)
yelp$ID <- c(1:nrow(yelp))

chisq.test(yelp$cool, yelp$stars)
chisq.test(yelp$useful, yelp$stars)
chisq.test(yelp$funny, yelp$stars)

#Data Cleaning
yelp$text <- gsub(yelp$name, "", yelp$text)
yelp$text <- gsub("nn", " ", yelp$text)
yelp$text <- gsub("'s", "", yelp$text)

data("stop_words")
texts <- data_frame(ID = yelp$ID, texting = yelp$text)
#tokenizing and filtering out stopwords
t_texts <- texts %>% group_by(ID) %>% unnest_tokens(word, texting) %>%  anti_join(stop_words)
#lemmatizing words
t_texts$word <- lemmatize_words(t_texts$word)
t_texts <- t_texts %>% anti_join(stop_words)
#filtering out numbers and word length <3
t_texts <- t_texts %>% subset(!str_detect(t_texts$word, regex("\\d")))
t_texts <- subset(t_texts, nchar(t_texts$word)>2)

head(t_texts)

#Making a dictionary based on review
wordfq <- count(data.frame(word = t_texts$word, stringsAsFactors = FALSE), word)
dict <- inner_join(t_texts, data.frame(ID = yelp$ID, rating = yelp$stars))
dict <- dict %>% group_by(word) %>% summarise(avg_stars = mean(rating)) 
dict <- inner_join(dict, wordfq) %>% filter(n>5)
dict <- dict %>% arrange(-avg_stars, -n)
dict$label <- ifelse(dict$avg_stars>=3, 1, -1)

#meaningless words to the analysis
rem <- c("food", "service", "time", "restaurant", "eat", "chicken", "price", "menu", "fry", "pizza", 
             "sauce", "table", "meal", "lunch", "drink", "staff", "bite", "salad", "cheese", "serve", "people",
             "night", "bar", "sandwich", "dish", "burger", "egg", "pork", "beef", "fish", "steak", "potato", 
             "sushi") 

#Top 20 positive and negative sentiment words
dict %>% mutate(label = factor(dict$label, label = c("neg", "pos"))) %>% filter(!word %in% rem) %>% select(word, n, label) %>% filter(label == "pos") %>% arrange(-n) %>% top_n(n = 20, wt = n ) %>% mutate(word = reorder(word, n)) %>% ggplot(aes(x = word, y = n, fill = label))+scale_fill_manual(values='green')+geom_col(show.legend = FALSE)+ggtitle("Word Frequency for Positive Sentiment Words")+coord_flip()

dict %>% mutate(label = factor(dict$label, label = c("neg", "pos"))) %>% filter(!word %in% rem) %>% select(word, n, label) %>% filter(label == "neg") %>% arrange(-n) %>% top_n(n = 20, wt = n ) %>% mutate(word = reorder(word, n)) %>% ggplot(aes(x = word, y = n, fill = label))+scale_fill_manual(values='red')+geom_col(show.legend = FALSE)+ggtitle("Word Frequency for Negative Sentiment Words")+coord_flip()


##Harvard IV Dictionary Sentiment Prediction

data(DictionaryGI)
harv <- data.frame(word = c(DictionaryGI$negative, DictionaryGI$positive), stringsAsFactors = FALSE)
harv$sentiment <- ifelse(harv$word %in% DictionaryGI$negative, "negative", "positive")

cat("Matching Terms with Harvard: ", length(intersect(harv$word, dict$word)))

harvtexts <- t_texts %>% inner_join(harv) %>% count(word, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive-negative)

hvd <- aggregate(harvtexts[, c(1, 5)], by = list(harvtexts$ID), mean)
hvd$label_match <- ifelse( sign(hvd$sentiment) == sign(yelp$label[yelp$ID %in% hvd$ID]), 1, 0)

cat("Sentiment Prediction Rate")
prop.table(table(hvd$label_match))

#Bing Dictionary Sentiment Prediction
cat("Matching Terms with Bing ", length(intersect(get_sentiments("bing")$word, dict$word)))

bingtexts <- t_texts %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive-negative)

bng <- aggregate(bingtexts[, c(1, 5)], by = list(bingtexts$ID), mean)
bng$label_match <- ifelse( sign(bng$sentiment) == sign(yelp$label[yelp$ID %in% bng$ID]), 1, 0)
cat("Sentiment Prediction Rate")
prop.table(table(bng$label_match))

##AFINN Dictionary Sentiment Prediction
cat("Matching Terms with AFINN: ", length(intersect(get_sentiments("afinn")$word, dict$word)))

afinn <- t_texts %>% inner_join(get_sentiments("afinn")) %>% group_by(ID, word) %>% 
summarise(sentiment = sum(score)) %>% mutate(method = "AFINN")

afn <- aggregate(afinn[, c(1, 3)], by = list(afinn$ID), mean)
afn$label_match <- ifelse( sign(afn$sentiment) == sign(yelp$label[yelp$ID %in% afn$ID]), 1, 0)
cat("Sentiment Prediction Rate:")
prop.table(table(afn$label_match))

##AFINN Dictionary Sentiment Prediction
cat("Matching Terms with AFINN: ", length(intersect(get_sentiments("afinn")$word, dict$word)))

afinn <- t_texts %>% inner_join(get_sentiments("afinn")) %>% group_by(ID, word) %>% 
summarise(sentiment = sum(score)) %>% mutate(method = "AFINN")

afn <- aggregate(afinn[, c(1, 3)], by = list(afinn$ID), mean)
afn$label_match <- ifelse( sign(afn$sentiment) == sign(yelp$label[yelp$ID %in% afn$ID]), 1, 0)
cat("Sentiment Prediction Rate:")
prop.table(table(afn$label_match))

library(lexicon)
#SentiWordNet Dictionary Sentiment Prediction
cat("Matching Terms with SentiWordNet: ", length(intersect(get_sentiments("loughran")$word, dict$word)))

sentiwn <- t_texts %>% inner_join(get_sentiments("loughran")) %>% group_by(ID, word) %>% 
summarise(sentiment = sum(score)) %>% mutate(method = "SentiWordNet")

swn <- aggregate(sentiwn[, c(1, 3)], by = list(sentiwn$ID), mean)
swn$label_match <- ifelse( sign(swn$sentiment) == sign(yelp$label[yelp$ID %in% swn$ID]), 1, 0)
cat("Sentiment Prediction Rate:")
prop.table(table(swn$label_match))


#TFIDF Bing WordCloud
library(wordcloud)
library(RColorBrewer)

pos<- bingtexts%>% filter(positive==1) %>% group_by(ID,word)%>% count(ID,word)%>%bind_tf_idf(ID,word,n)
wordcloud(pos$word,pos$tf_idf,max.words=500, colors=brewer.pal(8,"Dark2"),scale=c(3,0))

neg<- harvtexts%>% filter(negative==1) %>% group_by(ID,word)%>% count(ID,word)%>%bind_tf_idf(ID,word,n)
wordcloud(neg$word,neg$tf_idf,max.words=500, colors=brewer.pal(8,"Dark2"),scale=c(3,0))

dev.off()

#Modeling

###Bingsplit
set.seed(12345)
docs <- sample(bng$ID, size = 20000, replace = FALSE) #Took a random sample of 20000 from reviews with matching terms from Bingard dictionary

dtm <- bingtexts[bingtexts$ID %in% docs, ] %>% group_by(ID, word) %>% count(ID, word) %>% bind_tf_idf(ID, word, n) %>% ungroup() %>% cast_dtm(ID, word, tf_idf)

index <- sample(docs, size = 10000, replace = FALSE) #Obtaining index for training/test sets from already randomized samples

#hold-out set: split ratio = 50:50
t_BingTrn <- dtm[as.numeric(dtm$dimnames$Docs) %in% index, ] %>% as.matrix() %>% as.data.frame() #Training
t_BingTest <- dtm[!as.numeric(dtm$dimnames$Docs) %in% index, ]  %>% as.matrix() %>% as.data.frame() #Test 

yelptrn <- yelp$label[yelp$ID %in% as.numeric(row.names(t_BingTrn))]
yelptest <- yelp$label[yelp$ID %in% as.numeric(row.names(t_BingTest))]

###AFN split
set.seed(12345)
docsa <- sample(afn$ID, size = 20000, replace = FALSE) #Took a random sample of 20000 from reviews with matching terms from Bingard dictionary

dtma <- bingtexts[bingtexts$ID %in% docsa, ] %>% group_by(ID, word) %>% count(ID, word) %>% bind_tf_idf(ID, word, n) %>% ungroup() %>% cast_dtm(ID, word, tf_idf)

indexa <- sample(docsa, size = 10000, replace = FALSE) #Obtaining index for training/test sets from already randomized samples

#hold-out set: split ratio = 50:50
t_AfnTrn <- dtma[as.numeric(dtma$dimnames$Docs) %in% indexa, ] %>% as.matrix() %>% as.data.frame() #Training
t_AfnTest <- dtma[!as.numeric(dtma$dimnames$Docs) %in% indexa, ]  %>% as.matrix() %>% as.data.frame() #Test 

yelptrna <- yelp$label[yelp$ID %in% as.numeric(row.names(t_AfnTrn))]
yelptesta <- yelp$label[yelp$ID %in% as.numeric(row.names(t_AfnTest))]

###HarvardSplit
set.seed(12345)
docsh <- sample(hvd$ID, size = 20000, replace = FALSE) #Took a random sample of 20000 from reviews with matching terms from Bingard dictionary

dtmh <- harvtexts[harvtexts$ID %in% docsh, ] %>% group_by(ID, word) %>% count(ID, word) %>% bind_tf_idf(ID, word, n) %>% ungroup() %>% cast_dtm(ID, word, tf_idf)

indexh <- sample(docsh, size = 10000, replace = FALSE) #Obtaining index for training/test sets from already randomized samples

#hold-out set: split ratio = 50:50
t_hvdTrn <- dtmh[as.numeric(dtmh$dimnames$Docs) %in% indexh, ] %>% as.matrix() %>% as.data.frame() #Training
t_hvdTest <- dtmh[!as.numeric(dtmh$dimnames$Docs) %in% indexh, ]  %>% as.matrix() %>% as.data.frame() #Test 

yelptrnh <- yelp$label[yelp$ID %in% as.numeric(row.names(t_HvdTrn))]
yelptesth <- yelp$label[yelp$ID %in% as.numeric(row.names(t_HvdTest))]

####

#Naive Bayes Bing
library(e1071)
dtm_nb_trn <- apply(t_BingTrn, 2, function(x) ifelse(x>0, 1, 0))
dtm_nb_test <- apply(t_BingTest, 2, function(x) ifelse(x>0, 1, 0))
yelp_nb_trn <- ifelse(yelptrn>0, 1, 0)
yelp_nb_test<- ifelse(yelptest>0, 1, 0)

t_BingTrn_nb<- naiveBayes(x = dtm_nb_trn, y = as.factor(yelp_nb_trn), laplace = 1)
##Training
predNB<- predict(t_BingTrn_nb, dtm_nb_trn, type="class")
cat("Training Accuracy:", mean(predNB==yelp_nb_trn), "\n")
print(table(pred = predNB, actual = yelp_nb_test))

##Test
predNB<- predict(t_BingTrn_nb, dtm_nb_test, type="class")
cat("Test Accuracy:", mean(predNB==yelp_nb_test), "\n")
print(table(pred = predNB, actual = yelp_nb_test))

#Naive Bayes AFINN
library(e1071)
dtm_nb_trna <- apply(t_AfnTrn, 2, function(x) ifelse(x>0, 1, 0))
dtm_nb_testa <- apply(t_AfnTest, 2, function(x) ifelse(x>0, 1, 0))
yelp_nb_trna <- ifelse(yelptrna>0, 1, 0)
yelp_nb_testa<- ifelse(yelptesta>0, 1, 0)

t_AfnTrn_nb<- naiveBayes(x = dtm_nb_trna, y = as.factor(yelp_nb_trna), laplace = 1)
##Training
predNBa<- predict(t_AfnTrn_nb, dtm_nb_trna, type="class")
cat("Training Accuracy:", mean(predNBa==yelp_nb_trna), "\n")
print(table(pred = predNBa, actual = yelp_nb_testa))

##Test
predNBa<- predict(t_AfnTrn_nb, dtm_nb_testa, type="class")
cat("Test Accuracy:", mean(predNBa==yelp_nb_testa), "\n")
print(table(pred = predNBa, actual = yelp_nb_testa))

#Naive Bayes Harvard
library(e1071)
dtm_nb_trnh <- apply(t_hvdTrn, 2, function(x) ifelse(x>0, 1, 0))
dtm_nb_testh <- apply(t_hvdTest, 2, function(x) ifelse(x>0, 1, 0))
yelp_nb_trnh <- ifelse(yelptrnh>0, 1, 0)
yelp_nb_testh<- ifelse(yelptesth>0, 1, 0)

t_hvdTrn_nb<- naiveBayes(x = dtm_nb_trnh, y = as.factor(yelp_nb_trnh), laplace = 1)
##Training
predNBh<- predict(t_hvdTrn_nb, dtm_nb_trnh, type="class")
cat("Training Accuracy:", mean(predNBh==yelp_nb_trnh), "\n")
print(table(pred = predNBh, actual = yelp_nb_testh))

##Test
predNBh<- predict(t_hvdTrn_nb, dtm_nb_testh, type="class")
cat("Test Accuracy:", mean(predNBh==yelp_nb_testh), "\n")
print(table(pred = predNBh, actual = yelp_nb_testh))



#SVM Harvard

##Training
svmh<- svm(x = t_hvdTrn, y = yelptrnh, scale = FALSE, type = "nu-classification", kernel = "linear", nu = 0.33)
predSVMTrnh<- predict(svmh, t_hvdTrn)
cat("Training Accuracy: ", mean(predSVMTrnh==yelptrnh))

##Test
predSVMtesth<- predict(svmh, t_hvdTest)
cat("Test Accuracy: ", mean(predSVMtesth==yelptesth), "\n")
print(table(pred = predSVMtesth,actual = yelptesth))

#SVM AFINN

##Training
svma<- svm(x = t_AfnTrn, y = yelptrna, scale = FALSE, type = "nu-classification", kernel = "linear", nu = 0.33)
predSVMTrna<- predict(svma, t_AfnTrn)
cat("Training Accuracy: ", mean(predSVMTrna==yelptrna))

##Test
predSVMtesta<- predict(svma, t_AfnTest)
cat("Test Accuracy: ", mean(predSVMtesta==yelptesta), "\n")
print(table(pred = predSVMtesta,actual = yelptesta))

#SVM Bing

##Training
svm<- svm(x = t_BingTrn, y = yelptrn, scale = FALSE, type = "nu-classification", kernel = "linear", nu = 0.33)
predSVMTrnb<- predict(svm, t_BingTrn)
cat("Training Accuracy: ", mean(predSVMTrnb==yelptrn))

##Test
predSVMtestb<- predict(svm, t_BingTest)
cat("Test Accuracy: ", mean(predSVMtestb==yelptest), "\n")
print(table(pred = predSVMtestb,actual = yelptest))

#Random Forest

#RF Bing
require(randomForest)
yelptrain_c <- ifelse(yelptrn == -1, 0, 1)
yelptest_c <- ifelse(yelptest == -1, 0, 1)

t_BingTrn_rf<- randomForest(x= t_BingTrn, y = as.factor(yelptrain_c), ntree = 100)
plot(t_BingTrn_rf)

#training
predTr <- predict(t_BingTrn_rf, t_BingTrn, type = "class")
cat("Train Accuracy: ", mean(predTr==yelptrain_c), "\n")
print(table(predTr,yelptrain_c))

#test
predRfTst<- predict(t_BingTrn_rf,t_BingTrn, type = "class")
cat("Test Accuracy: ", mean(predRfTst==yelptest_c), "\n")
print(table(predRfTst,yelptest_c))

#RF AFINN
require(randomForest)
yelptrain_c <- ifelse(yelptrn == -1, 0, 1)
yelptest_c <- ifelse(yelptest == -1, 0, 1)

t_BingTrn_rf<- randomForest(x= t_BingTrn, y = as.factor(yelptrain_c), ntree = 50)
plot(t_BingTrn_rf)

#training
predTr <- predict(t_BingTrn_rf, t_BingTrn, type = "class")
cat("Train Accuracy: ", mean(predTr==yelptrain_c), "\n")
print(table(predTr,yelptrain_c))

#test
predRfTst<- predict(t_BingTrn_rf,t_BingTrn, type = "class")
cat("Test Accuracy: ", mean(predRfTst==yelptest_c), "\n")
print(table(predRfTst,yelptest_c))

#RF Harvard
require(randomForest)
yelptrain_c <- ifelse(yelptrn == -1, 0, 1)
yelptest_c <- ifelse(yelptest == -1, 0, 1)

t_BingTrn_rf<- randomForest(x= t_BingTrn, y = as.factor(yelptrain_c), ntree = 50)
plot(t_BingTrn_rf)

#training
predTr <- predict(t_BingTrn_rf, t_BingTrn, type = "class")
cat("Train Accuracy: ", mean(predTr==yelptrain_c), "\n")
print(table(predTr,yelptrain_c))

#test
predRfTst<- predict(t_BingTrn_rf,t_BingTrn, type = "class")
cat("Test Accuracy: ", mean(predRfTst==yelptest_c), "\n")
print(table(predRfTst,yelptest_c))
