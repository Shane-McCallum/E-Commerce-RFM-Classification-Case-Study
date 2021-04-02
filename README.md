![cover image](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/RFM%20Header.png)

# E-Commerce RFM Segmentation with KMeans

*Every business has consumers; without the consumers there are no sales, and the business would die. So, it is of no surprise that many businesses have a vested interest in understanding who their consumers are, what they want, when do they want it, and how much are they spending for it. The process of finding these answers is known as [customer segmentation](https://smallbusiness.chron.com/basis-segmenting-consumer-markets-1417.html). There are a lot of methods used for customer segmentation today. One of the most universally useful solutions to segmenting a consumer base is using an [Recency, Frequency, and Monetary Value (RFM) analysis](https://www.barilliance.com/rfm-analysis/#:~:text=RFM%20analysis%20is%20a%20data,much%20they've%20spent%20overall.). An RFM analysis allows businesses to segment their consumers into categories or tiers based on the consumer's scores across the three dimensions. Businesses can then see who their highest, average, and lowest performing consumers are. This enables a business to direct [marketing campaigns and strategies to the proper consumer audience more accurately](https://www.optimove.com/resources/learning-center/rfm-segmentation); which in turn will lead to more sales of higher volume. Below, I have assembled a case study intended to focus on demonstrating how to develop a useful RFM Analysis and Classification model from the UCI E-Commerce store data.*

## 1. Problem Identification

[Problem Statement PDF](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/Problem%20Statement%20Worksheet.pdf)

As always, in order to develop an applicable solution, there must first be a clear and smart problem. In this case study, the problem has been clearly provided and I know exactly what the solution should look like. It is clear that a RFM analysis will be needed to segment the consumer data into the proper tiers for precise and specialized marketing.  This often looks like email notifications about discount codes for items left in a checkout cart, or an invitation to a loyalty program or subscription option for the most valued consumers.

## 2. Data Wrangling

E-commerce datasets are often private, proprietary information for most companies. This makes them quite hard to find among publicly available data. However, the data used for this case study was made free to the public through UCI Machine Learning Repository and is well known by most in the data science and analytics community. It is available for download [here](https://www.kaggle.com/carrie1/ecommerce-data).

In regards to cleaning the data, there was some minor touches needed before the data was appropriate for segmentation. 

First, the data had about 135,000 entries where the customer ID was blank. This could easily be explained as customers purchasing under guest accounts instead of making an account with an assigned customed ID number. As there is no way of tracking a specific guest's recency, frequency and total monetary value, I removed them from the data. Additionally, I noted that returns counted as negative values for quantity of the product in the transaction. This makes sense, as the product is returned and the original sale is null. However, negative quantity is assigned a Unit Price of GBP0.00, which only throws off the data’s minimum and median values for UnitPrice as well as standard deviations. Therefore, all return transactions were removed from the data as well. Finally, the last thing of note, is that there were some outliers of considerable magnitude within the data for Quantity and Revenue. These outliers will skew the segmentation of the consumer base into tiers, so I removed the top 0.99% of them. 

## 3. Exploratory Data Analysis (EDA)

In order to prepare for a proper RFM analysis I wanted to be sure that I had cleaned the data enough to get a reliable representation of the client's customer base. In addition, the exploratory analysis would reveal if any further cleaning was needed. What I was checking for here were heavily skewed bar plots that would indicate a strong imbalance among the features. First up was to examine the customer base itself. I am not as concerned about which customers purchased the most products as I am that only few of the several thousand customers make up most of the purchases. To check for this, I used SciKit Learns bar plot feature.

![Top 100 Customers by # of Purchases](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/Top%20100%20Customers%20by%20%23%20of%20Purchases.png)

Wow, alright; CustomerID 17841 has made nearly 8,000 purchases in the last year alone. Following that, though, there is a gradual drop off in the number of purchases. As long as the data does not "flatten out" in comparison to the maximum, I am not too concerned.

Up next, I wanted to make sure there wasn’t a bias in the data on which products were purchased, as that would also signal a single product sustaining the business, and therefore an RFM analysis would be of little use.

![Top 50 Most Often Purchased Items](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/Top%2050%20Most%20Often%20Purchased%20Items.png?raw=true)

Great, nothing of concern here. Next up, I wanted to take a look at which countries comprised the data to be sure I could give a confident representation of them.

![Countries by # of Transactions from Them](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/Countries%20by%20%23%20of%20Transactions%20from%20Them.png)

Here, it is clear that the data is lacking enough transactional data, and therefore, likely lacking the consumer base in any country outside of the UK. So, in order to preserve the authenticity of the segmentation as much as I could, I removed the transactions made from other countries.

The last little bit of EDA I wanted to do is to create a cohort analysis. A cohort analysis will allow me to see how many customers returned each month after the month which they completed their first transaction. Customer cohorts are incredibly valuable as they create mutually exclusive customer segments. This allows a marketing team to clearly measure the metrics of a products lifecycle among the customer base as well as measuring the standard customer lifecycle; such as yearly purchase cycles. This cohort analysis tells the client that each row represents a cohort, the month which that cohort was first active as a consumer on the store, and that each column represents a new month and the retention percentage of consumers from the cohort's first month.

![Cohort Analysis](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/CohortAnalysis.png)

## 4. Modeling with KMeans Clustering

To begin, I grouped the data by Customer ID and created the features for Recency, Frequency, and Monetary Value. Again, Recency is the amount of days from the customer's last purchase; so, a smaller number here means a higher Recency Score. Frequency and Monetary Value are exactly what they sound like, and therefore a higher value here now means a higher score in the segmentation.

![RFM_seg initial table](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/RFM%20seg%20Table.png)

Now, the client wants the customers segmented into three tiers; the Best Customers, Average Customers, and Weak Customers. In order to do this, I am going to make the range for each segment to be scored between 1 and 4. This will divide the customers up into quartiles across three features, allowing for clean and even segmentation. A customer segmented as Best Customer would have to have a score totaling up to 9 or more (such as Recency-3, Frequency-3, and Monetary Value-3).

![RFM seg w/ RFM Level](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/RFM%20seg%20Tablew%20RFM%20Level.png)

To properly implement [KMeans Clustering](https://matteucci.faculty.polimi.it/Clustering/tutorial_html/kmeans.html). This will make all of the features of the RFM segmentation easily comparable and prevent KMeans from outputting really skewed and stretched clusters. It's clear by the distribution graphs below that normalizing the data makes it a lot easier to compare.

![Not-Normalized Distributions](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/Not-Normalized%20Distributions.png)

![Normalized Distributions](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/Normalized%20Distributions.png)

Next, I iterated through Kmeans Clusters to see which clusters had the "best" [sum-of-square-errors score (SSE)](https://datascienceplus.com/k-means-clustering/#:~:text=SSE%20is%20defined%20as%20the,which%20the%20graph%20decrease%20abruptly.). The best value is usually the one where the "crook of the elbow" is visible. However, I know from the client that they want 3 tiers, which means I need to at least have three clusters.

![KMeans Elbow Check](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/KMeans%20Elbow%20Check.png)

Well, the elbow isn't very pronounced. So, to check and se which cluster would be the best I will use another method; the [Silhouette Score](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam). The Silhouette Score provides a great visual representation of the KMeans clustering algorithm at work. What I want to see here is a set of even "Silhouettes," indicating that the clusters are evenly sized and not overlapping into each other. 

![Silhouette Score 2](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/Silhouette%20Scores%202.png)
![Silhouette Score 3](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/Silhouette%20Scores%203.png)
![Silhouette Score 4](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/Silhouette%20Scores%204.png)
![Silhouette Score 5](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/Silhouette%20Scores%205.png)
![Silhouette Score 6](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/Silhouette%20Scores%206.png)

Finally, I will check the [Average Silhouette Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html). Generally, the K value with the closest average score to 1 is the best, with 0 meaning there’s some overlap, and -1 meaning there exist no clusters.

![Average Silhouette Score Graph](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/Average%20Silhouette%20Score.png)

Seems to be that K=3 provides the best score for the segmentation; since 2 is not really an option for what the client wants. With the cluster value chosen, I run the KMeans algorithm and fit it to the normalized RFM data. To visualize and compare, I have plotted the both the customer segmentations and the KMeans clusters on snake plots. Snake plots are great for comparing segments across the various features to identify where they differ.

![RFM Snake Plot](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/RFM%20Snake%20Plot.png)
![KMeans Clusters Snake Plot](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/KMeans%20Clusters%20Snake%20Plot.png)

It is clear that the Best Customers are in Cluster 1. The Average Customer is probably overlapped a little bit between Cluster 0 and Cluster 2, but are mostly found in Cluster 0. Finally, The Weak Customers, are found in Cluster 2.

To better understand how customers where assigned into their segments, I made a heatmap of the relative importance of the customers value in each importance and how that determined their segment. Ideally, this and the relative importance of attributes for the clusters should be similar. The heatmaps show that the further away from 0 an attribute score is, the more important it is in determining what falls into that segment or cluster.

![Relative Importance of Attributes for Segments](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/Relative%20Importance%20of%20Attributes%20for%20Segments.png)
![Relative Importance of Attributes for Clusters](https://github.com/Shane-McCallum/E-Commerce-RFM-Classification-Case-Study/blob/main/2.%20README%20files/Relative%20Importance%20of%20Attributes%20for%20Clusters.png)

## 5. Conclusion and Future Tests

The client now has a clear segmentation of their consumer base and can see who their best, average and weakest customers are based on their recency, frequency and monetary value. Additionally, the client can continually feed future customer information into the model and segment future customers rather easily. However, there are some important notes. First and foremost is that this model is designed for the client's consumers located only within the UK, and not elsewhere. Additionally, the data has been cleaned of most of the client's outlier consumers, and therefore, cannot be applied to those consumers. This stated, with continual data of our client's consumer transactions being fed into the model, the outliers may shrink into the standard quartiles of the data. For future test, I would encourage the client to copy this model and apply it to the data they have for their consumers from other countries. Once there is a healthy enough population of customers from these other countries, the client could have several different models tracking their consumer base across various continents experiencing different trends.
