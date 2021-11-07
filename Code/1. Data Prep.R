library(readr)
library(tidyverse)
library(tidymodels)
library("textcat")
library(stringr)
library(splitstackshape)
library("imputeTS")
library(tidytext)
library(stringr)
library(tibble)
library(tidyr)
library(readxl)
library(data.table)
library(gdata)
library(tm)
library(sentimentr)
library(SentimentAnalysis)
library(ldatuning)
library(topicmodels)
library(textdata)
library(pbapply)
library(caTools)
library(parallel)
library(foreach)
library(doParallel)

#install.packages("imputeTS")
#install.packages("textcat")
#install.packages("splitstackshape")

airbnb_train_x_2021 <- read_csv("airbnb_train_x_2021.csv")
airbnb_train_y_2021 <- read_csv("airbnb_train_y_2021.csv")
airbnb_test_x_2021 <- read_csv("airbnb_test_x_2021.csv")

messed_up_row_indices <- c()

for (i in 1:3) {
  messed_up_row_indices <- append(messed_up_row_indices , which(as.factor(airbnb_train_x_2021$cancellation_policy) == levels(as.factor(airbnb_train_x_2021$cancellation_policy))[i]))
}

airbnb_train_y_2021 <- airbnb_train_y_2021[-messed_up_row_indices,]
airbnb_train_x_2021$source <- "train"
airbnb_test_x_2021$source <- "test"

airbnb_train_x_2021 <- subset(airbnb_train_x_2021,!(is.na(airbnb_train_x_2021$accommodates)))

combined_data <- rbind(airbnb_train_x_2021,airbnb_test_x_2021)

##numerical data

remove_dollar_sign <- c("price", "weekly_price", "cleaning_fee", "security_deposit", "extra_people")

cleaned_1 <- combined_data  %>%
  mutate_at(remove_dollar_sign, ~str_replace_all(., pattern="\\$", replacement="")) %>%
  mutate_at(remove_dollar_sign, ~str_replace_all(., pattern=",", replacement="")) %>%
  mutate_at(remove_dollar_sign, ~as.numeric(.))



remove_percent_sign <- c("host_response_rate")

cleaned_2 <- cleaned_1 %>%
  mutate_at(remove_percent_sign, ~str_replace_all(., pattern="\\%", replacement="")) %>%
  mutate_at(remove_percent_sign, ~str_replace_all(., pattern=",", replacement="")) %>%
  mutate_at(remove_percent_sign, ~as.numeric(.)) %>% 
  select(-X1)



length_train <- nrow(airbnb_train_x_2021) # original length of train data
#corrected_length_train <- nrow(airbnb_train_x_2021)- length(messed_up_row_indices)# for later train test split


#cleaned_2 <- cleaned_2[-messed_up_row_indices,] %>% select(-X1) # removed messed up rows #cancellation policy cleaned as well

skimr::skim(cleaned_2)

# miniumn nights and square feet are clean with the exception of having NA

#-city
levels(as.factor(cleaned_2$city))
which(cleaned_2$city == "11220")


cleaned_2$city[311] = cleaned_2$city_name[311] # replaced Numerical value with the city name


#-city_name
levels(as.factor(cleaned_2$city_name))
#cleaned_2$city_language <- textcat(cleaned_2$city)
# better to take city_name as a variable for model generation.


#-country
cleaned_2$country %>% unique()
levels(as.factor(cleaned_2$country))

#-country_code
cleaned_2$country_code %>% unique()
levels(as.factor(cleaned_2$country_code))
#same as using country, better to use country code





cleaned_4 <- cleaned_2  %>% 
  mutate(amenities = gsub(pattern = "\\{", replacement = "" , x = amenities), 
         amenities = gsub(pattern = "\\}", replacement = "" , x = amenities),
         amenities = gsub(pattern = '\\"', replacement = "" , x = amenities),
         amenities = ifelse(amenities == "",'No Amenity Mentioned',amenities))


cleaned_5 <- cSplit(cleaned_4, 'amenities', sep=",", stripWhite=TRUE, type.convert=FALSE) %>% 
  mutate(amenities = cleaned_4$amenities) 


#list of all amenities
amenities_list <- c(t(cleaned_5[,70:147])) %>% unique()
amenities_list <- amenities_list[!is.na(amenities_list)]


##new Method

new_data <- as.data.frame(cleaned_5$amenities)
new_data_1 <- cSplit(new_data, 'cleaned_5$amenities', sep=",", stripWhite=TRUE, type.convert=FALSE) %>% 
  mutate(amenities = cleaned_4$amenities)



dummies <- new_data_1 %>% mutate(ID = row_number()) %>% 
  pivot_longer(cols = 1:78, names_to = 'new_field')%>% 
  filter(value != "") %>% select(-new_field) %>% 
  mutate(flag = 1) %>% 
  pivot_wider(names_from = value, values_from = flag,values_fn = length)




mid_data <- cleaned_4 %>% select(-amenities)
cleaned_6 <- cbind(mid_data, dummies) %>% select(-ID)

cleaned_7 <- cleaned_6 %>% select(71:280) %>% replace(is.na(.),0)

cleaned_8 <- cleaned_6 %>% select(-c(71:280))

cleaned_8$experiences_offered %>% unique() # remove experiences offered, country (as country code is there )

cleaned_9 <- cbind(cleaned_8,cleaned_7) # final dataset Harsh
cleaned_9 <- cleaned_9 %>% select(-experiences_offered)



cleaned_10 <- cleaned_9 %>%
  mutate(
    host_response_rate = ifelse(is.na(host_response_rate), median(host_response_rate, na.rm = TRUE), 
                                host_response_rate),
    
    host_response_time = ifelse(is.na(host_response_time), "other", host_response_time),
    host_response_time = as.factor(host_response_time),
    host_listings_count = ifelse(is.na(host_listings_count), median(host_listings_count, na.rm=TRUE),
                                 host_listings_count),
    instant_bookable = as.factor(instant_bookable),
    jurisdiction_names = ifelse(is.na(jurisdiction_names), "Other", jurisdiction_names),
    jurisdiction_names = str_trim(gsub("[^[:alnum:]]", " ", jurisdiction_names)),
    jurisdiction_names = str_squish(jurisdiction_names),          #Removing repeated whitespaces
    jurisdiction_names = as.factor(jurisdiction_names),
    is_business_travel_ready = ifelse(is.na(is_business_travel_ready), "Not Provided", is_business_travel_ready),
    is_business_travel_ready = as.factor(is_business_travel_ready),
    is_location_exact = as.factor(is_location_exact)
    
  )


  
colSums(is.na(cleaned_10))
# No NA's in bed_type cancellation_policy, city, city_name, country, country_code, extra_people
# monthly_price has lots of NA (89050)
# weekly_price has lots of NA (85126) , using only price in model
# security_deposit not being used as has lots of NA and is correlated with other price(45766)
# square_feet not being used as has lots of NA(110412)

cleaned_11 <- cleaned_10 %>% group_by(city_name, property_type, accommodates) %>% 
  mutate(cleaning_fee = ifelse(is.na(cleaning_fee), mean(cleaning_fee, na.rm = TRUE), 
                               cleaning_fee),
         city = ifelse(is.na(city), city_name,city),
         minimum_nights = ifelse(is.na(minimum_nights), mean(minimum_nights, na.rm = TRUE), 
                                 minimum_nights),
         price = ifelse(is.na(price), mean(price, na.rm = TRUE), 
                        price)) # Harsh


cleaned_12 <- cleaned_11 %>%
  mutate(license_possession  = case_when(is.na(license) ~ "f",
                                         grepl("pending", license, fixed = TRUE) | grepl("Pending", license, fixed = TRUE) | 
                                           grepl("process", license, fixed = TRUE)   ~ "Pending",
                                         TRUE ~ "t"),
         license_possession = as.factor(license_possession)) %>% 
  mutate(market = ifelse(is.na(market), city_name, market)) %>%
  group_by(market) %>%
  mutate(count_market = n(),
         market = ifelse(count_market < 100, "Other", market),
         market = as.factor(market)) %>% 
  ungroup %>% 
  group_by(property_type) %>%
  mutate(count_property_type = n(),
         property_type = ifelse(count_property_type < 100 & !is.na(property_type), "Other", property_type)) %>% 
  ungroup() %>% 
  mutate(require_guest_phone_verification = as.character(require_guest_phone_verification),
         require_guest_phone_verification = ifelse(is.na(require_guest_phone_verification), "FALSE", require_guest_phone_verification),
         require_guest_phone_verification = as.factor(require_guest_phone_verification)) %>% 
  group_by(city_name, property_type, accommodates) %>%
  mutate(bathrooms = ifelse(is.na(bathrooms), median(bathrooms, na.rm = TRUE), bathrooms)) %>% 
  ungroup() %>% 
  group_by(city_name, property_type, accommodates) %>%
  mutate(bedrooms = ifelse(is.na(bedrooms), median(bedrooms, na.rm = TRUE), bedrooms)) %>% 
  ungroup() %>% 
  group_by(city_name, property_type, bed_type, accommodates) %>%
  mutate(beds = ifelse(is.na(beds), median(beds, na.rm = TRUE), beds)) 


to_replace <- c("Бруклин","南艾尔蒙地","哈仙达岗","圣地亚哥", "天普市","布鲁克林","沃尔纳特",
                "法拉盛", "波士顿", "波摩纳", "洛杉矶","纽约", "纽约市", "纽约法拉盛",
                "罗兰高地", "艾尔蒙地", "西雅图","聖地亞哥","西科维纳","马里布")

after_replace <- c("Brooklyn","South El Monte","Hacienda Height","San Diego", "Temple City",
                   "Brooklyn", "Walnut", "Flushing", "Boston", "Pomona", "Los Angeles", "New York",
                   "New York", "Flushing", "Rowland Heights", "El Monte", "Seattle", "San Diego","West Covina", "Malibu" )

for(i in 1:length(to_replace)){
  cleaned_12$city <- gsub(to_replace[i],after_replace[i],cleaned_12$city,perl = TRUE)
}


cleaned_13 <- cleaned_12

cleaned_14 <- cleaned_13  %>% 
  mutate(host_verifications = gsub(pattern = "\\[", replacement = "" , x = host_verifications ), 
         host_verifications  = gsub(pattern = "\\]", replacement = "" , x = host_verifications ),
         host_verifications  = gsub(pattern = "\\'", replacement = "" , x = host_verifications ),
         host_verifications  = ifelse(host_verifications == "",'No Host Verification Mentioned',host_verifications ))


cleaned_15 <- cSplit(cleaned_14, 'host_verifications', sep=",", stripWhite=TRUE, type.convert=FALSE) %>% 
  mutate(host_verifications = cleaned_14$host_verifications)


#list of all host Verifications
host_verifications_list <- c(t(cleaned_15[,282:295])) %>% unique()
host_verifications_list <- host_verifications_list[!is.na(host_verifications_list)]

##new Method

test_ver <- cleaned_15[,c(282:295)]


cleaned_16 <- test_ver %>% mutate(ID=row_number()) %>% 
  pivot_longer(cols = host_verifications_01:host_verifications_14, names_to = 'new_field')%>% 
  #filter(value != "") %>% 
  select(-new_field) %>% 
  mutate(flag = 1) %>% 
  pivot_wider(names_from = value, values_from = flag, values_fn = length) %>% # 11 error rows getting removed
  select(-"NA") # removing the extra NA column that was created


cleaned_17 <- cleaned_16 %>% replace(is.na(.),0)

cleaned_18 <- cleaned_15 %>% select(-c(282:295))

cleaned_19 <- cbind(cleaned_18,cleaned_17) # warning msg cause of those 11 rows(treated so no warning)
cleaned_19 <- cleaned_19 %>% select(-ID)

cleaned_20 <- cleaned_19 %>%
  mutate(host_is_superhost = ifelse(is.na(host_is_superhost), "FALSE", host_is_superhost)) %>%
  mutate(host_has_profile_pic = ifelse(is.na(host_has_profile_pic), "FALSE", host_has_profile_pic)) %>%
  mutate(host_identity_verified = ifelse(is.na(host_identity_verified), "FALSE", host_identity_verified))
# we won't use host_neighbourhood


cleaned_21<- cleaned_20 %>% 
  group_by(city) %>% 
  mutate(city = trimws(city),
         latitude = if_else(is.na(latitude),mean(latitude,na.rm = T),latitude),
         longitude = if_else(is.na(longitude),mean(longitude,na.rm = T),longitude)) %>% ungroup() %>% 
  mutate(host_total_listings_count = if_else(is.na(host_total_listings_count),
                                             mean(host_total_listings_count,na.rm = T),
                                             host_total_listings_count),
         maximum_nights = if_else(is.na(maximum_nights),median(maximum_nights,na.rm = T),maximum_nights),
         require_guest_profile_picture = if_else(is.na(require_guest_profile_picture),FALSE,require_guest_profile_picture),
         requires_license = if_else(is.na(requires_license),FALSE,requires_license),
         room_type = if_else(is.na(room_type),"Unknown",room_type),
         host_total_listings_count = as.factor(ntile(host_total_listings_count,5)),
         maximum_nights = cut(maximum_nights, breaks = c(0,14,30,1124,Inf),
                              labels = c("0-2 Weeks","2 weeks to a month","1 month to 3 years","More than 3 years"))) %>% select(-state)

state_clean <- read.csv("state.csv",stringsAsFactors = F)

airbnb_clean_1 <- left_join(cleaned_21,state_clean, by = "city_name")

airbnb_clean <- airbnb_clean_1 %>% 
  mutate(smart_location = if_else(is.na(smart_location),paste0(city,", ",state),smart_location))
#cleaned till text analysis (text analysis left)


#verification for cleaned data
na_count <-sapply(airbnb_clean, function(y) sum(length(which(is.na(y)))))
na_count <- as.data.frame(na_count)
na_count$var <- row.names(na_count)
na_count <- na_count %>% mutate(total = nrow(airbnb_clean), per_na = na_count/total) %>% 
  select(-total)

#Text Analysis

##access

library(foreach)
library(doParallel)

cl <- makeCluster(6)
registerDoParallel(cl)

bing_score <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
  print(i)
  tokens <- data_frame(text = airbnb_clean$access[i]) %>% unnest_tokens(word, text)
  sentiment <- tokens %>%
    inner_join(get_sentiments("bing")) %>% # pull out only sentiment words
    count(sentiment) %>% # count the # of positive & negative words
    spread(sentiment, n, fill = 0)
  if("positive" %in% colnames(sentiment) & "negative" %in% colnames(sentiment)){
    bing_val <- sentiment$positive - sentiment$negative
  }else if("positive" %in% colnames(sentiment)){
    bing_val <- sentiment$positive
  }else if("negative" %in% colnames(sentiment)){
    bing_val <- -sentiment$negative
  }else{
    bing_val <- 0
  }
  bing_val
}
airbnb_clean$Bing_Score_access <- bing_score

Polarity <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
    print(i/nrow(airbnb_clean)*100)
    sentence_vec <- get_sentences(airbnb_clean$access[i])
    polarity_val <- sum(sentiment(sentence_vec)$sentiment)
    polarity_val
  }
airbnb_clean$Polarity_access <- Polarity

Loughran_Score <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
    print(i/nrow(airbnb_clean)*100)
    tokens <- data_frame(text = airbnb_clean$access[i]) %>% unnest_tokens(word, text)
    sentiment <- tokens %>%
      inner_join(get_sentiments("loughran")) %>% # pull out only sentiment words
      count(sentiment) %>% # count the # of positive & negative words
      spread(sentiment, n, fill = 0)
    if("positive" %in% colnames(sentiment) & "negative" %in% colnames(sentiment)){
      loughran_val <- sentiment$positive - sentiment$negative
    }else if("positive" %in% colnames(sentiment)){
      loughran_val <- sentiment$positive
    }else if("negative" %in% colnames(sentiment)){
      loughran_val <- -sentiment$negative
    }else{
      loughran_val <- 0
    }
    loughran_val
  }
airbnb_clean$Loughran_Score_access <- Loughran_Score

Afinn_Score <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
  print(i/nrow(airbnb_clean)*100)
  tokens <- data_frame(text = airbnb_clean$access[i]) %>% unnest_tokens(word, text)
  sentiment <- tokens %>%
    inner_join(get_sentiments("afinn")) # pull out only sentiment words
  afinn_val <- sum(sentiment$value)
  return(afinn_val)
}
airbnb_clean$Afinn_Score_access <- Afinn_Score

##summary

bing_score <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
  print(i)
  tokens <- data_frame(text = airbnb_clean$summary[i]) %>% unnest_tokens(word, text)
  sentiment <- tokens %>%
    inner_join(get_sentiments("bing")) %>% # pull out only sentiment words
    count(sentiment) %>% # count the # of positive & negative words
    spread(sentiment, n, fill = 0)
  if("positive" %in% colnames(sentiment) & "negative" %in% colnames(sentiment)){
    bing_val <- sentiment$positive - sentiment$negative
  }else if("positive" %in% colnames(sentiment)){
    bing_val <- sentiment$positive
  }else if("negative" %in% colnames(sentiment)){
    bing_val <- -sentiment$negative
  }else{
    bing_val <- 0
  }
  bing_val
}
airbnb_clean$Bing_Score_summary <- bing_score

Polarity <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
    print(i/nrow(airbnb_clean)*100)
    sentence_vec <- get_sentences(airbnb_clean$summary[i])
    polarity_val <- sum(sentiment(sentence_vec)$sentiment)
    polarity_val
  }
airbnb_clean$Polarity_summary <- Polarity

Loughran_Score <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
    print(i/nrow(airbnb_clean)*100)
    tokens <- data_frame(text = airbnb_clean$summary[i]) %>% unnest_tokens(word, text)
    sentiment <- tokens %>%
      inner_join(get_sentiments("loughran")) %>% # pull out only sentiment words
      count(sentiment) %>% # count the # of positive & negative words
      spread(sentiment, n, fill = 0)
    if("positive" %in% colnames(sentiment) & "negative" %in% colnames(sentiment)){
      loughran_val <- sentiment$positive - sentiment$negative
    }else if("positive" %in% colnames(sentiment)){
      loughran_val <- sentiment$positive
    }else if("negative" %in% colnames(sentiment)){
      loughran_val <- -sentiment$negative
    }else{
      loughran_val <- 0
    }
    loughran_val
  }
airbnb_clean$Loughran_Score_summary <- Loughran_Score

Afinn_Score <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
  print(i/nrow(airbnb_clean)*100)
  tokens <- data_frame(text = airbnb_clean$summary[i]) %>% unnest_tokens(word, text)
  sentiment <- tokens %>%
    inner_join(get_sentiments("afinn")) # pull out only sentiment words
  afinn_val <- sum(sentiment$value)
  return(afinn_val)
}
airbnb_clean$Afinn_Score_summary <- Afinn_Score

##space

bing_score <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
  print(i)
  tokens <- data_frame(text = airbnb_clean$space[i]) %>% unnest_tokens(word, text)
  sentiment <- tokens %>%
    inner_join(get_sentiments("bing")) %>% # pull out only sentiment words
    count(sentiment) %>% # count the # of positive & negative words
    spread(sentiment, n, fill = 0)
  if("positive" %in% colnames(sentiment) & "negative" %in% colnames(sentiment)){
    bing_val <- sentiment$positive - sentiment$negative
  }else if("positive" %in% colnames(sentiment)){
    bing_val <- sentiment$positive
  }else if("negative" %in% colnames(sentiment)){
    bing_val <- -sentiment$negative
  }else{
    bing_val <- 0
  }
  bing_val
}
airbnb_clean$Bing_Score_space <- bing_score

Polarity <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
    print(i/nrow(airbnb_clean)*100)
    sentence_vec <- get_sentences(airbnb_clean$space[i])
    polarity_val <- sum(sentiment(sentence_vec)$sentiment)
    polarity_val
  }
airbnb_clean$Polarity_space <- Polarity

Loughran_Score <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
    print(i/nrow(airbnb_clean)*100)
    tokens <- data_frame(text = airbnb_clean$space[i]) %>% unnest_tokens(word, text)
    sentiment <- tokens %>%
      inner_join(get_sentiments("loughran")) %>% # pull out only sentiment words
      count(sentiment) %>% # count the # of positive & negative words
      spread(sentiment, n, fill = 0)
    if("positive" %in% colnames(sentiment) & "negative" %in% colnames(sentiment)){
      loughran_val <- sentiment$positive - sentiment$negative
    }else if("positive" %in% colnames(sentiment)){
      loughran_val <- sentiment$positive
    }else if("negative" %in% colnames(sentiment)){
      loughran_val <- -sentiment$negative
    }else{
      loughran_val <- 0
    }
    loughran_val
  }
airbnb_clean$Loughran_Score_space <- Loughran_Score

Afinn_Score <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
  print(i/nrow(airbnb_clean)*100)
  tokens <- data_frame(text = airbnb_clean$space[i]) %>% unnest_tokens(word, text)
  sentiment <- tokens %>%
    inner_join(get_sentiments("afinn")) # pull out only sentiment words
  afinn_val <- sum(sentiment$value)
  return(afinn_val)
}
airbnb_clean$Afinn_Score_space <- Afinn_Score


##neighborhood_overview

bing_score <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
  print(i)
  tokens <- data_frame(text = airbnb_clean$neighborhood_overview[i]) %>% unnest_tokens(word, text)
  sentiment <- tokens %>%
    inner_join(get_sentiments("bing")) %>% # pull out only sentiment words
    count(sentiment) %>% # count the # of positive & negative words
    spread(sentiment, n, fill = 0)
  if("positive" %in% colnames(sentiment) & "negative" %in% colnames(sentiment)){
    bing_val <- sentiment$positive - sentiment$negative
  }else if("positive" %in% colnames(sentiment)){
    bing_val <- sentiment$positive
  }else if("negative" %in% colnames(sentiment)){
    bing_val <- -sentiment$negative
  }else{
    bing_val <- 0
  }
  bing_val
}
airbnb_clean$Bing_Score_neighborhood_overview <- bing_score

Polarity <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
    print(i/nrow(airbnb_clean)*100)
    sentence_vec <- get_sentences(airbnb_clean$neighborhood_overview[i])
    polarity_val <- sum(sentiment(sentence_vec)$sentiment)
    polarity_val
  }
airbnb_clean$Polarity_neighborhood_overview <- Polarity

Loughran_Score <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
    print(i/nrow(airbnb_clean)*100)
    tokens <- data_frame(text = airbnb_clean$neighborhood_overview[i]) %>% unnest_tokens(word, text)
    sentiment <- tokens %>%
      inner_join(get_sentiments("loughran")) %>% # pull out only sentiment words
      count(sentiment) %>% # count the # of positive & negative words
      spread(sentiment, n, fill = 0)
    if("positive" %in% colnames(sentiment) & "negative" %in% colnames(sentiment)){
      loughran_val <- sentiment$positive - sentiment$negative
    }else if("positive" %in% colnames(sentiment)){
      loughran_val <- sentiment$positive
    }else if("negative" %in% colnames(sentiment)){
      loughran_val <- -sentiment$negative
    }else{
      loughran_val <- 0
    }
    loughran_val
  }
airbnb_clean$Loughran_Score_neighborhood_overview <- Loughran_Score

Afinn_Score <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
  print(i/nrow(airbnb_clean)*100)
  tokens <- data_frame(text = airbnb_clean$neighborhood_overview[i]) %>% unnest_tokens(word, text)
  sentiment <- tokens %>%
    inner_join(get_sentiments("afinn")) # pull out only sentiment words
  afinn_val <- sum(sentiment$value)
  return(afinn_val)
}
airbnb_clean$Afinn_Score_neighborhood_overview <- Afinn_Score


## transit

bing_score <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
  print(i)
  tokens <- data_frame(text = airbnb_clean$transit[i]) %>% unnest_tokens(word, text)
  sentiment <- tokens %>%
    inner_join(get_sentiments("bing")) %>% # pull out only sentiment words
    count(sentiment) %>% # count the # of positive & negative words
    spread(sentiment, n, fill = 0)
  if("positive" %in% colnames(sentiment) & "negative" %in% colnames(sentiment)){
    bing_val <- sentiment$positive - sentiment$negative
  }else if("positive" %in% colnames(sentiment)){
    bing_val <- sentiment$positive
  }else if("negative" %in% colnames(sentiment)){
    bing_val <- -sentiment$negative
  }else{
    bing_val <- 0
  }
  bing_val
}
airbnb_clean$Bing_Score_transit <- bing_score

Polarity <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
    print(i/nrow(airbnb_clean)*100)
    sentence_vec <- get_sentences(airbnb_clean$transit[i])
    polarity_val <- sum(sentiment(sentence_vec)$sentiment)
    polarity_val
  }
airbnb_clean$Polarity_transit <- Polarity

Loughran_Score <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
    print(i/nrow(airbnb_clean)*100)
    tokens <- data_frame(text = airbnb_clean$transit[i]) %>% unnest_tokens(word, text)
    sentiment <- tokens %>%
      inner_join(get_sentiments("loughran")) %>% # pull out only sentiment words
      count(sentiment) %>% # count the # of positive & negative words
      spread(sentiment, n, fill = 0)
    if("positive" %in% colnames(sentiment) & "negative" %in% colnames(sentiment)){
      loughran_val <- sentiment$positive - sentiment$negative
    }else if("positive" %in% colnames(sentiment)){
      loughran_val <- sentiment$positive
    }else if("negative" %in% colnames(sentiment)){
      loughran_val <- -sentiment$negative
    }else{
      loughran_val <- 0
    }
    loughran_val
  }
airbnb_clean$Loughran_Score_transit <- Loughran_Score

Afinn_Score <- foreach(i = 1:nrow(airbnb_clean), .combine=c) %dopar% {
  print(i/nrow(airbnb_clean)*100)
  tokens <- data_frame(text = airbnb_clean$transit[i]) %>% unnest_tokens(word, text)
  sentiment <- tokens %>%
    inner_join(get_sentiments("afinn")) # pull out only sentiment words
  afinn_val <- sum(sentiment$value)
  return(afinn_val)
}
airbnb_clean$Afinn_Score_transit <- Afinn_Score
stopCluster(cl)

detach("package:MASS", unload=TRUE)
# needed if we want to use tidyverse package

scored <- read_csv("Scored_data.csv")

airbnb_clean <- cbind(airbnb_clean, scored[,307:326])

airbnb_final <- airbnb_clean %>% 
  mutate(minimum_nights = ifelse(is.na(minimum_nights), median(minimum_nights,na.rm = TRUE), minimum_nights),
         minimum_nights = ifelse(minimum_nights > 3, "More than 3", minimum_nights),
         minimum_nights = as.factor(minimum_nights),
         property_type = if_else(is.na(property_type),"unknown",property_type),
         cleaning_fee = if_else(is.na(cleaning_fee),mean(cleaning_fee,na.rm = T),cleaning_fee),
         host_since = if_else(is.na(host_since),mean(host_since,na.rm = T),host_since),
         instant_bookable = as.logical(if_else(is.na(as.logical(instant_bookable)),FALSE,as.logical(instant_bookable))),
         is_location_exact = as.logical(if_else(is.na(as.logical(is_location_exact)),TRUE,as.logical(is_location_exact))),
         price = as.numeric(price),
         price = if_else(is.na(price),mean(price,na.rm = T),price))

airbnb_final <-  airbnb_final %>% 
  select(-c(square_feet,host_acceptance_rate,license, monthly_price,weekly_price,
            space,security_deposit, interaction,access,description,host_about,
            host_name,house_rules,interaction,name,neighborhood_overview,notes,
            street,summary,transit,amenities,city, country,`Beach view`,`Stand alone steam shower`,
            `Shared pool`,host_verifications))

train_data <- airbnb_final %>% filter(source == "train")
test_data <- airbnb_final %>% filter(source == "test")

airbnb_train_final <- bind_cols(train_data, airbnb_train_y_2021)

write.csv(airbnb_train_final,"Final Train Data.csv",row.names = F)
write.csv(test_data,"Final Test Data.csv",row.names = F)
