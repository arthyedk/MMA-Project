#Associationrules

retail <- read.csv(file.choose())


#install and load package arules
install.packages ("arules")
library(arules)
#install and load arulesViz
install.packages("arulesViz")
library(arulesViz)
#install and load tidyverse
install.packages("tidyverse")
library(tidyverse)
#install and load readxml
install.packages("readxml")
library(readxl)
#install and load knitr
install.packages("knitr")
library(knitr)
#load ggplot2 as it comes in tidyverse
library(ggplot2)
#install and load lubridate
install.packages("lubridate")
library(lubridate)
#install and load plyr
install.packages("plyr")
library(plyr)
library(dplyr)

#complete.cases(data) will return a logical vector indicating which rows have no missing values. Then use the vector to get only rows that are complete using retail[,].
retail <- online_retail[complete.cases(online_retail), ]


#mutate function is from dplyr package. It is used to edit or add new columns to dataframe. Here Description column is being converted to factor column. as.factor converts column to factor column. %>% is an operator with which you may pipe values to another function or expression
retail %>% mutate(Description = as.factor(Description))

retail %>% mutate(Country = as.factor(Country))

#Converts character data to date. Store InvoiceDate as date in new variable
retail$Date <- as.Date(retail$InvoiceDate)

#Extract time from InvoiceDate and store in another variable
TransTime<- format(retail$InvoiceDate,format="%H:%M:%S")

#Convert and edit InvoiceNo into numeric
InvoiceNo <- as.numeric(as.character(retail$InvoiceNo))


#Bind new columns TransTime and InvoiceNo into dataframe retail
cbind(retail,TransTime)

cbind(retail,InvoiceNo)

#get a glimpse of your data
glimpse(retail)


#Before applying MBA/Association Rule mining, we need to convert dataframe into transaction data so that all items that are bought together in one invoice are in one row. You can see in glimpse output that each transaction is in atomic form, that is all products belonging to one invoice are atomic as in relational databases. This format is also called as the singles format

library(plyr)
#ddply(dataframe, variables_to_be_used_to_split_data_frame, function_to_be_applied)
transactionData <- ddply(retail,c("InvoiceNo","Date"),
                         function(df1)paste(df1$Description,
                                            collapse = ","))
#The R function paste() concatenates vectors to character and separated results using collapse=[any optional charcater string ]. Here ',' is used

transactionData

write.csv(transactionData,"/Users/arthurkyazze/Desktop/Docs/market_basket_transactions.csv", quote = FALSE, row.names = FALSE)
#transactionData: Data to be written
#"D:/Documents/market_basket.csv": location of file with file name to be written to
#quote: If TRUE it will surround character or factor column with double quotes. If FALSE nothing will be quoted
#row.names: either a logical value indicating whether the row names of x are to be written along with x, or a character vector of row names to be written.


tr = read.transactions('/Users/arthurkyazze/Desktop/Docs/market_basket_transactions.csv', format = 'basket', sep=',')
#sep tell how items are separated. In this case you have separated using ','

'trObj<-as(dataframe.dat,"transactions")'

tr


summary(tr)

# Create an item frequency plot for the top 20 items
if (!require("RColorBrewer")) {
  # install color package of R
  install.packages("RColorBrewer")
  #include library RColorBrewer
  library(RColorBrewer)
}
itemFrequencyPlot(tr,topN=20,type="absolute",col=brewer.pal(8,'Pastel2'), main="Absolute Item Frequency Plot")

itemFrequencyPlot(tr,topN=20,type="absolute")

# Min Support as 0.001, confidence as 0.8.
association.rules <- apriori(tr, parameter = list(supp=0.001, conf=0.8,maxlen=10))

summary(association.rules)

#Mining stopped (maxlen reached). Only patterns up to a length of 10 returned!

inspect(association.rules[1:20])

#Using the above output, you can make analysis such as:
#100% of the customers who bought 'WOBBLY CHICKEN' also bought 'METAL'.
#100% of the customers who bought 'BLACK TEA' also bought SUGAR 'JARS'.

#Limiting the number and size of rules
shorter.association.rules <- apriori(tr, parameter = list(supp=0.001, conf=0.8,maxlen=3))

#Removing redundant rules
subset.rules <- which(colSums(is.subset(association.rules, association.rules)) > 1) # get subset rules in vector
length(subset.rules)  #> 3913

metal.association.rules <- apriori(tr, parameter = list(supp=0.001, conf=0.8),appearance = list(default="lhs",rhs="METAL"))
# Here lhs=METAL because you want to find out the probability of that in how many customers buy METAL along with other items
inspect(head(metal.association.rules))

metal.association.rules <- apriori(tr, parameter = list(supp=0.001, conf=0.8),appearance = list(lhs="METAL",default="rhs"))

# Here lhs=METAL because you want to find out the probability of that in how many customers buy METAL along with other items
inspect(head(metal.association.rules))
