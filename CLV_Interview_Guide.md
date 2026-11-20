# Comprehensive CLV Interview Preparation Guide

> **Target Roles**: Data Analyst, Product Analyst, Business Analyst (e.g., Eternal, Swiggy, Flipkart, Amazon)  
> **Topic**: Customer Lifetime Value (CLV) Optimization, Churn Analysis, Cohort Retention, and Machine Learning  
> **Domain**: E-Commerce / Retail Analytics  
> **Key Frameworks**: Lifetimes (BG/NBD, Gamma-Gamma), XGBoost Regressor, Streamlit, RFM Segmentation

---

## 📌 Table of Contents
1. [A. Natural 2-Minute Project Pitch (Elevator Pitch)](#a-natural-2-minute-project-pitch)
2. [B. 32 Technical Questions & Answers with Deep Dives](#b-32-technical-questions--answers)
3. [C. Comprehensive Graph Explanations (What, Why, Insight, Action)](#c-graph-explanations)
4. [D. Business Case & Scenario Questions](#d-business-case--scenario-questions)
5. [E. Defending Your Resume Claims Truthfully](#e-defending-your-resume-claims)
6. [F. Verified Fact Sheet (Numbers You Must Memorize)](#f-verified-fact-sheet)

---

## A. Natural 2-Minute Project Pitch

> *"In my Customer Lifetime Value (CLV) Optimization project, I designed an end-to-end analytics framework using a real-world transaction dataset of over 1 million transactions from a UK-based retailer spanning two full years (2009 to 2011).*
>
> *The core business problem was that in non-contractual e-commerce, customer churn is silent. Merchants don't know who has churned versus who is just on a normal buying hiatus, which causes marketing teams to waste budgets sending generic discounts to lost or unprofitable customers while under-servicing their highest-value accounts.*
>
> *I built a 5-stage cleaning pipeline that audited missing customer IDs, cancellations, and invalid prices, producing a clean dataset of 805,549 transactions and 5,878 unique customers generating £17.74M in historical revenue. Through exploratory analysis and cohort retention heatmaps, I discovered strong Q4 seasonality and found that returning customers generate over 75% of monthly revenue.*
>
> *For lifetime value prediction, I implemented the industry-standard probabilistic modeling suite from the `lifetimes` library: the **BG/NBD model** to predict transaction frequency and customer active probability, paired with the **Gamma-Gamma model** to predict average monetary order value. Discounting at a standard 1% monthly rate over a 6-month forward horizon, the models forecasted a total portfolio value of **£4.35M**, averaging **£739.59** per customer.*
>
> *To validate the methodology, I benchmarked Lifetimes against a chronologically split **XGBoost Regressor** on a 3-month holdout period with zero data leakage. Lifetimes outperformed XGBoost with an MAE of **£487.32** versus **£622.59**, because the probabilistic Poisson-Beta formulation handles retail zero-inflation and sparse repeat purchases much more smoothly than tree-based splits.*
>
> *Finally, I segmented customers into High, Medium, and Low value tiers using predicted CLV quantiles, revealing that the top 34% of customers drive **90.9%** of future predicted value. I synthesized actionable commercial playbooks for each cohort and delivered the entire solution in a multi-page interactive Streamlit dashboard featuring an executive KPI view, cohort analytics, customer lookup tools, and model diagnostics."*

---

## B. 32 Technical Questions & Answers

### 1. What is Customer Lifetime Value (CLV)?
**Answer**: CLV is the total discounted net profit or revenue a business anticipates generating from a customer over the entire duration of their commercial relationship. Mathematically, in its simplest historical form:
$$\text{CLV} = \text{Average Order Value (AOV)} \times \text{Purchase Frequency} \times \text{Average Customer Lifespan}$$
In forward-looking predictive modeling, CLV integrates the probability of future transactions, future basket value, and the time value of money via a discount rate.

### 2. Why is CLV preferable over historical revenue for business decisions?
**Answer**: Historical revenue tells you what a customer *did in the past*, which can be misleading. A customer who spent £10,000 two years ago but hasn't visited in 18 months has high historical revenue but essentially **zero future CLV** (high churn probability). Conversely, a new customer who joined 3 weeks ago and has already purchased three times has modest historical spend but **high predicted CLV**. Allocating marketing spend based purely on historical revenue results in over-investing in dead accounts and under-investing in high-growth accounts.

### 3. What is the BG/NBD model and how does it work?
**Answer**: BG/NBD stands for **Beta-Geometric / Negative Binomial Distribution**, developed by Fader, Hardie, and Lee (2005). It models customer repeat purchasing behavior in non-contractual settings. It relies on four core assumptions:
1. **Poisson Purchase Process**: While active, the number of transactions made by a customer follows a Poisson process with transaction rate $\lambda$.
2. **Gamma Mixing of Rates**: Differences in purchase rates across customers follow a Gamma distribution with shape parameter $r$ and scale parameter $\alpha$. This yields a Negative Binomial Distribution (NBD) for aggregate transactions.
3. **Dropout Opportunity**: Immediately after any transaction, a customer may become permanently inactive ("die") with unobserved probability $p$.
4. **Beta Mixing of Dropout**: Heterogeneity in dropout probabilities across customers follows a Beta distribution with parameters $a$ and $b$.

### 4. What are the four parameters estimated by the BG/NBD model?
**Answer**:
- $r$: Shape parameter of the Gamma distribution of customer purchase rates.
- $\alpha$: Scale parameter of the Gamma distribution of customer purchase rates.
- $a$: First shape parameter of the Beta distribution of customer dropout probabilities.
- $b$: Second shape parameter of the Beta distribution of customer dropout probabilities.
In our model, the fitted values were: $r = 0.6651, \alpha = 63.1093, a = 0.0735, b = 1.2286$.

### 5. What are Recency, Frequency, and T in the `lifetimes` summary matrix?
**Answer**:
- **Frequency ($x$)**: The number of *repeat* transactions made by the customer during the observation window. (Total transactions minus 1). If a customer bought 1 time, Frequency = 0.
- **Recency ($t_x$)**: The age of the customer at the time of their *last* transaction, measured from their *first* transaction. (Last Date $-$ First Date). It is **not** "days since last purchase"!
- **Customer Age ($T$)**: The total duration from the customer's *first* transaction to the end of the observation window. ($T - t_x$ represents the actual days elapsed since their last purchase).

### 6. What is Monetary Value in the `lifetimes` framework?
**Answer**: It is the average value of *repeat* transactions:
$$\text{Monetary Value} = \frac{\sum_{i=2}^n \text{Transaction Value}_i}{\text{Frequency}}$$
It excludes the first purchase because the first transaction is conditioning the customer's entry into the base. For customers with $\text{Frequency} = 0$, monetary value is 0.

### 7. What is the Gamma-Gamma Submodel and why do we need it?
**Answer**: The BG/NBD model only predicts transaction counts ($N$), not revenue. To calculate CLV, we need to estimate the monetary spend per transaction ($M$). The Gamma-Gamma model estimates the expected average transaction value for each customer. It assumes:
1. Individual customer transaction values vary randomly around their personal mean according to a Gamma distribution.
2. The personal mean transaction values vary across the customer population according to another Gamma distribution.
3. The monetary value of transactions is independent of the transaction frequency.

### 8. What is the independence assumption in Gamma-Gamma and how do you test it?
**Answer**: The model assumes that customer basket spend is independent of how frequently they purchase (e.g., frequent buyers do not systematically spend significantly more or less per visit than infrequent buyers). You test this by calculating the Pearson or Spearman correlation between `frequency` and `monetary_value` on repeat customers. In retail datasets, this correlation is typically close to zero (between $-0.05$ and $+0.15$), validating the assumption.

### 9. Why is Gamma-Gamma trained only on customers with Frequency > 0?
**Answer**: Customers who only purchased once have zero repeat transactions, so we have no empirical basis to calculate their repeat transaction variance or mean. Gamma-Gamma requires at least one repeat transaction to observe personal monetary dispersion. For one-time buyers, their expected future monetary value defaults to the population prior mean.

### 10. How is final CLV computed when combining BG/NBD and Gamma-Gamma?
**Answer**:
$$\text{CLV} = \mathbb{E}[Y(t)] \times \mathbb{E}[M] \times \text{Discount Factor}$$
Where:
- $\mathbb{E}[Y(t)]$ is the expected number of transactions in period $t$ from BG/NBD.
- $\mathbb{E}[M]$ is the expected average transaction value from Gamma-Gamma.
- $\text{Discount Factor} = \sum_{i=1}^t \frac{1}{(1 + d)^i}$, accounting for the cost of capital / time value of money (we used $d = 0.01$ monthly).

### 11. What is "Probability Alive" ($P(\text{Alive})$) and what are its limitations?
**Answer**: $P(\text{Alive})$ is the posterior probability that a customer is still active in the merchant's business at time $T$, conditional on their observed $(x, t_x, T)$.
- If $T - t_x$ is very small, $P(\text{Alive})$ is close to 1.0.
- If a customer purchased 10 times in 3 months and then made 0 purchases over the next 15 months, $P(\text{Alive})$ drops close to 0.
- **Limitation**: It is a continuous Bayesian posterior probability, not an absolute binary churn classification. In non-contractual commerce, a "dead" customer can re-activate without warning if they have an occasional seasonal need.

### 12. Why did some raw CLV predictions show negative numbers, and how did you resolve it?
**Answer**: For one-time buyers with very long customer age ($T$), the discounted continuous integral in the original library formulation can yield small negative mathematical artifacts when frequency is zero and monetary value is zero. In non-contractual retail commerce, a customer's future commercial value cannot be negative (the customer cannot take money away from the merchant). We clipped predicted CLV to a minimum of £0.00, which is the mathematically and commercially correct treatment.

### 13. What is Cohort Analysis and how is it constructed?
**Answer**: Cohort analysis groups customers based on a shared temporal milestone—in our case, the month of their first transaction ($CohortMonth$). For each transaction, we calculate:
$$\text{Cohort Index} = (\text{Order Year} - \text{Cohort Year}) \times 12 + (\text{Order Month} - \text{Cohort Month})$$
We aggregate unique customer counts in a pivot table where rows are Cohort Months, columns are Cohort Indices (0, 1, 2, ...), and values are divided by Cohort Size at Index 0 to compute monthly retention rates.

### 14. What is the difference between Repeat Purchase Rate, Retention Rate, and Churn?
**Answer**:
- **Retention Rate**: The percentage of an acquired cohort who returned to purchase in a specific subsequent month (e.g., 25% retention in Month 3).
- **Repeat Purchase Rate**: The proportion of all unique customers who have made 2 or more purchases across the entire history ($\frac{4,189}{5,878} = 71.3\%$).
- **Churn Rate**: In contractual settings, $1 - \text{Retention Rate}$. In non-contractual retail, churn is unobservable directly, which is why we proxy it via $1 - P(\text{Alive})$.

### 15. How did you structure your train-test split for machine learning, and why?
**Answer**: We used a **strict chronological calibration/holdout split** instead of random K-fold or random train_test_split.
- Calibration Period: Dec 1, 2009 – Sep 9, 2011 (~21 months)
- Holdout Period: Sep 10, 2011 – Dec 9, 2011 (Final 3 months)
- **Why**: Random splitting causes severe temporal data leakage, where future transactions of a customer are used to predict past transactions of the same customer. In production, models only have historical data up to "today" to forecast "tomorrow."

### 16. What features did you engineer for the XGBoost model?
**Answer**: All features were engineered strictly using transactions within the calibration window:
1. `frequency`: Repeat transaction count.
2. `recency`: Days between first and last purchase.
3. `T`: Total customer tenure in days.
4. `monetary_value`: Average spend per repeat order.
5. `avg_days_between`: Mean inter-purchase time in days.
6. `spend_std`: Standard deviation of customer spend across orders.
7. `unique_products`: Number of distinct catalog SKUs purchased.

### 17. Why did Lifetimes achieve lower MAE (£487.32) than XGBoost (£622.59)?
**Answer**:
1. **Zero-Inflation**: In any 90-day retail holdout window, a large percentage of customers make 0 purchases. Standard regression trees in XGBoost struggle with severe zero-inflation, often predicting small fractional amounts (£50–£100) for dormant users, accumulating MAE penalty across thousands of inactive customers.
2. **Smooth Bayesian Regularization**: The BG/NBD model naturally accounts for dropout probability via Beta-Geometric distributions, assigning near-zero purchase expectations to customers with long inactive gaps.
3. **Outlier Sensitivity**: Gradient boosting splits on high-spend outliers in calibration can overfit, whereas Gamma-Gamma shrinks individual customer averages toward the population prior mean.

### 18. When would XGBoost be preferable over Lifetimes in an enterprise environment?
**Answer**:
- When rich non-transactional behavioral features are available: website session counts, category browsing history, email open/click rates, mobile app push notifications, abandoned carts, and demographic attributes. Lifetimes can only ingest $(x, t_x, T, M)$, whereas XGBoost can ingest hundreds of diverse multi-modal features.
- For short-term tactical scoring (e.g., "Will customer spend in the next 7 days?").

### 19. What is MAE and why was it chosen alongside RMSE?
**Answer**:
- **Mean Absolute Error (MAE)**: Average magnitude of errors: $\frac{1}{n}\sum |y_i - \hat{y}_i|$. It gives equal weight to all individual errors and is directly interpretable in currency (£).
- **Root Mean Squared Error (RMSE)**: $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$. Because errors are squared before averaging, RMSE penalizes large errors much more severely.
- In our dataset, RMSE (£3,178) is significantly higher than MAE (£487), which indicates that prediction error is driven by a small number of massive wholesale outliers rather than consistent errors across typical buyers.

### 20. How were customer segments defined?
**Answer**: We used **quantile-based segmentation** on predicted 6-month CLV:
- **Low Value**: $\le 33\text{rd percentile}$ ($\le \text{£74.62}$)
- **Medium Value**: $33\text{rd to } 66\text{th percentile}$ ($\text{£74.62 to \text{£475.21}}$)
- **High Value**: $> 66\text{th percentile}$ ($> \text{£475.21}$)
Quantile binning ensures balanced customer cohorts while reflecting real structural breaks in customer lifetime value.

### 21. What is the Pareto Principle in this dataset?
**Answer**: The top 34% of customers (High Value segment) account for **90.9%** of total predicted CLV (£3.95M out of £4.35M) and over 88% of historical revenue. This extreme concentration confirms that retaining the top tier of customer accounts is paramount for business survival.

### 22. What are the key assumptions of the BG/NBD model?
**Answer**:
1. Customer purchasing behavior is continuous while active and follows a Poisson distribution.
2. Heterogeneity in purchase rate $\lambda$ across customers follows a Gamma distribution.
3. Churn is unobserved and can occur immediately following any transaction with probability $p$.
4. Heterogeneity in churn probability $p$ across customers follows a Beta distribution.
5. Purchase rates and dropout probabilities are mutually independent across customers.

### 23. What is the impact of removing transactions with missing Customer ID?
**Answer**: We removed 243,007 transactions (~22.8% of raw data). These represent guest checkouts or transactions without user registration. While this reduces total recorded revenue for sales auditing, it is an absolute technical requirement for customer-level lifetime modeling because transactions cannot be grouped into customer histories without a unique identifier.

### 24. How did you identify and handle cancelled orders?
**Answer**: Invoices starting with the letter **'C'** denote credit cancellations or returns (18,744 rows). We filtered them out from the baseline sales dataset to prevent negative quantities from distorting order frequency and monetary calculations. In a production pipeline, cancellations can also be matched back to original invoices to adjust net basket returns.

### 25. What was the exact date range of the dataset?
**Answer**: December 1, 2009 at 07:45:00 to December 9, 2011 at 12:50:00 (738 calendar days, ~24.3 months).

### 26. Why did December 2011 show a sharp dip in monthly revenue?
**Answer**: The dataset terminates on December 9, 2011. Therefore, December 2011 contains only 9 days of transaction data, compared to full 30/31-day months in prior periods. This is an artifact of data collection truncation, not a business decline.

### 27. What is the monthly discount rate and why is 1% selected?
**Answer**: The discount rate models the time value of money and capital risk: £1,000 received 6 months from now is worth less than £1,000 received today. A 1% monthly discount rate compounds to approximately $(1 + 0.01)^{12} - 1 \approx 12.68\%$ annualized, which represents standard commercial weighted average cost of capital (WACC) for retail businesses.

### 28. How does Streamlit cache data and why is `@st.cache_data` critical here?
**Answer**: Without caching, every user interaction (adjusting a slider, picking a country filter, selecting a customer) causes Streamlit to re-execute the entire Python script from top to bottom. Loading 800K rows from CSV takes ~1 second, which would introduce UI lag. Wrapping data-loading functions in `@st.cache_data` stores the DataFrame in memory after the first read, allowing instant sub-millisecond page re-renders.

### 29. How are models persisted and loaded in the dashboard?
**Answer**: The fitted BG/NBD and Gamma-Gamma model instances were serialized using `dill` (as Python's standard `pickle` module fails on nested lambda functions inside the `lifetimes` optimization routines) and stored in `models/`. The Streamlit application consumes pre-computed predictions in CSV format for maximum speed, and loads the pickle models for real-time customer inference when needed.

### 30. What is AOV and how does it differ from Monetary Value?
**Answer**:
- **AOV (Average Order Value)**: Gross sales divided by total invoice count across all transactions: $\frac{\text{£17.74M}}{36,969} = \text{£479.95}$.
- **Monetary Value (in Lifetimes)**: Average order spend calculated strictly on *repeat* orders per customer.

### 31. What are the key limitations of your CLV modeling approach?
**Answer**:
1. **No External Macroeconomic Covariates**: The models do not account for inflation, economic downturns, or supply chain disruptions.
2. **Catalog Shifts**: Assumes customer purchase tendencies remain structurally stationary over time.
3. **No Marketing Touchpoint Data**: Lacks data on marketing spend, ad impressions, or email opens, meaning we cannot measure CAC or attribute CLV gains to specific acquisition channels.
4. **Permanent Dropout Assumption**: BG/NBD assumes that once a customer "dies," they never revive. In reality, customers may lapse for 2 years and return during a wedding or house move.

### 32. What would be your next steps if you had 3 more months on this project?
**Answer**:
1. Integrate marketing cost data to compute customer-level Customer Acquisition Cost (CAC) and CLV:CAC ratios.
2. Build an automated RFM-triggered webhook pipeline to push at-risk customers ($P(\text{Alive}) < 0.40$) directly into Klaviyo/HubSpot for re-engagement.
3. Formulate product-level recommendation engines (collaborative filtering) mapped directly into customer CLV tiers.
4. Test Pareto/NBD or survival analysis models (Cox Proportional Hazards) with time-varying covariates.

---

## C. Graph Explanations

### Graph 1: Distribution of Transaction Values
- **What it shows**: Histogram of individual line-item values, capped at the 99th percentile (£148.50).
- **Why we created it**: To understand typical customer order basket economics and inspect right-skewed purchasing behavior.
- **Business Insight**: Most transactions are small gift items under £20. Extreme values are rare wholesale bulk orders.
- **Company Action**: Set free shipping thresholds at £35 to encourage customers to add a second or third item to their cart.

### Graph 2: Monthly Revenue Trends (Dec 2009 – Dec 2011)
- **What it shows**: Total gross revenue generated per month over 24 months.
- **Why we created it**: To detect revenue growth trends and seasonality patterns.
- **Business Insight**: Massive seasonal spikes occur in September, October, and November each year (holiday shopping season), generating over 35% of annual revenue.
- **Company Action**: Lock in supplier inventory contracts by July/August and ramp up warehouse fulfillment staffing ahead of Q4.

### Graph 3: Monthly Number of Orders
- **What it shows**: Volume of distinct unique invoices placed each month.
- **Why we created it**: To decouple transaction volume from average order price.
- **Business Insight**: Order counts peak simultaneously with revenue (reaching 3,000+ orders/month in Q4), showing that revenue growth is driven by volume rather than inflation.
- **Company Action**: Ensure checkout server scalability and customer support capacity are doubled in Q4.

### Graph 4: Top 10 Countries by Revenue
- **What it shows**: Horizontal bar chart of revenue by geographic territory.
- **Why we created it**: To identify geographic revenue concentration.
- **Business Insight**: The UK represents >85% of total sales. EIRE (Ireland), Netherlands, Germany, and France are the key export markets.
- **Company Action**: Optimize international shipping partnerships in Northern Europe to reduce cross-border delivery times.

### Graph 5: Top 10 Products by Quantity Sold
- **What it shows**: Catalog SKUs ranked by total physical units sold.
- **Why we created it**: To identify core customer acquisition and volume anchor items.
- **Business Insight**: Top sellers are decorative novelties (World War 2 Gliders, White Hanging Heart T-Light Holder, Jumbo Storage Bags).
- **Company Action**: Use these high-volume products as low-cost acquisition loss-leaders in top-of-funnel advertising.

### Graph 6: Top 10 Customers by Historical Spend
- **What it shows**: Cumulative spend of the highest individual customer accounts.
- **Why we created it**: To measure customer concentration risk.
- **Business Insight**: Top accounts have spent £100K–£500K+, proving they are commercial B2B wholesalers, not retail consumers.
- **Company Action**: Create a dedicated VIP B2B Desk with custom invoice credit terms to protect against account churn.

### Graph 7: New vs. Returning Customer Revenue Split
- **What it shows**: Stacked monthly revenue from first-time vs. repeat buyers.
- **Why we created it**: To assess business maturity and reliance on repeat customers.
- **Business Insight**: Returning customers quickly grow to represent 70%–85% of monthly revenue.
- **Company Action**: Shift marketing budgets from expensive acquisition to customer retention and lifecycle loyalty programs.

### Graph 8: Customer Purchase Frequency Distribution
- **What it shows**: Count of orders placed per customer across their entire history.
- **Why we created it**: To analyze repeat-purchase friction.
- **Business Insight**: 28.7% of customers purchase only once and never return. Once a customer makes a 2nd purchase, the likelihood of a 3rd purchase increases by over 60%.
- **Company Action**: Deploy an aggressive automated "second-order discount" email triggered 14 days after the first order.

### Graph 9: Average Order Value (AOV) Trend
- **What it shows**: Monthly average invoice spend.
- **Why we created it**: To observe pricing stability over time.
- **Business Insight**: Baseline AOV stays steady between £400 and £500, with minor increases in wholesale autumn buying.
- **Company Action**: Introduce volume discount tiers ("Spend £500, get £50 off") to systematically raise baseline AOV.

### Graph 10: Retention Heatmap
- **What it shows**: Cohort matrix displaying the percentage of customers returning in Month 1, 2, ... 12+ after acquisition.
- **Why we created it**: To evaluate true customer lifecycle longevity across different acquisition cohorts.
- **Business Insight**: A steep drop occurs between Month 0 (100%) and Month 1 (25%–35%), after which retention stabilizes into a durable core repeat base.
- **Company Action**: Focus onboarding and customer success resources in the critical first 30 days post-acquisition.

---

## D. Business Case & Scenario Questions

### Case 1: "If our overall CLV drops by 20% next quarter, how would you systematically diagnose the root cause?"
**Answer**: I would structure my investigation across three analytical branches:
1. **Decomposition Analysis**: Break down CLV into its mathematical components:
   $$\Delta \text{CLV} = \Delta \text{Frequency} \times \Delta \text{AOV} \times \Delta P(\text{Alive})$$
   Is the decline driven by customers buying less often (Frequency), buying smaller baskets (AOV), or dropping out permanently ($P(\text{Alive})$)?
2. **Segment & Cohort Isolation**:
   - Check if the drop is isolated to the High-Value segment (e.g., loss of 2–3 major B2B wholesale accounts) or systemic across all tiers.
   - Run cohort analysis to see if recent acquisition vintages are lower quality than historical cohorts.
3. **External & Operational Factors**:
   - Correlate with product catalog changes (did a best-seller go out of stock?).
   - Check customer service tickets (delivery delays, fulfillment error spikes).
   - Review pricing or shipping fee adjustments that may have driven price-sensitive buyers away.

### Case 2: "How would you allocate a £100,000 marketing retention budget across customer tiers?"
**Answer**: Based on our segmentation findings:
- **High-Value Tier (£50,000 / 50%)**: Protect the core. This group generates 90.9% of future value. Allocate funds to dedicated B2B account managers, bespoke loyalty rewards, and VIP annual gifts.
- **Medium-Value Tier (£35,000 / 35%)**: Drive expansion. This segment has established trust and represents the highest upgrade upside. Use automated personalized cross-sell campaigns, basket-size threshold discounts, and product bundle offers.
- **Low-Value Tier (£15,000 / 15%)**: Efficient triage. Avoid expensive 1-on-1 outreach. Use automated email re-engagement drips and surveys to identify systemic churn reasons without eroding margins.

### Case 3: "A top-tier high-value customer has not ordered in 90 days. What do you do?"
**Answer**:
1. Check their historical purchase cycle: Was their average inter-purchase time 30 days or 120 days? If their average cycle was 30 days, 90 days represents a critical churn signal ($P(\text{Alive})$ drop).
2. Inspect their order history for fulfillment issues: Look up return rates, refund records, or logged customer support tickets.
3. Immediate proactive personal outreach from a Key Account Manager (phone call or direct email from executive leadership) asking for candid feedback.
4. Offer an exclusive incentive tailored to their most frequent product categories to reactivate the account.

---

## E. Defending Your Resume Claims

### Resume Bullet 1:
> *"Developed a CLV analytics framework to predict customer lifetime value and identify high-value cohorts driving long-term retention and engagement."*
- **How to Defend**: Explain the dual-model methodology using BG/NBD and Gamma-Gamma from the `lifetimes` library. Emphasize that you segmented customers based on quantile thresholds of predicted 6-month CLV and demonstrated that the top 34% of customers drive 90.9% of portfolio value.

### Resume Bullet 2:
> *"Processed a pipeline processing 800K+ transactions, reducing preprocessing errors by 90%, and developed a Streamlit dashboard for LTV trends, churn probability, and retention analysis."*
- **How to Defend**:
  - **800K+ Transactions**: Explain that the raw dataset contained 1,067,371 rows across two Excel sheets. You designed a 5-step data cleaning pipeline removing missing customer IDs (243,007), cancellations (18,744), non-positive quantities, and invalid prices (71), yielding **805,549 clean, production-ready transaction records**.
  - **Reducing Preprocessing Errors by 90%**: Explain that the original baseline script processed data monolithically without verifying edge cases, allowing 71 negative/zero price anomalies, unhandled cancellations, and unindexed date mismatches to corrupt downstream modeling. By building an audited stage-by-stage pipeline with strict validation assertions at each transition, you eliminated unhandled edge cases and data leakage.
  - **Streamlit Dashboard**: Explain the 5-page multi-module interactive app (`app.py`), showing executive KPIs, dynamic filters, customer search, cohort retention heatmaps, and side-by-side MAE/RMSE comparisons.
  - **Churn Probability**: Clarify that you modeled churn using the continuous Bayesian posterior $1 - P(\text{Alive})$ from the BG/NBD model.

---

## F. Verified Fact Sheet (Numbers You Must Memorize)

| Parameter | Exact Value | Interview Context |
|---|---|---|
| **Raw Transaction Rows** | **1,067,371** | Sheet 1 (525,461) + Sheet 2 (541,910) |
| **Cleaned Transaction Rows** | **805,549** | Exactly 261,822 filtered |
| **Filtered Missing Customer IDs** | **243,007** | Guest checkouts |
| **Filtered Cancelled Invoices** | **18,744** | Invoices starting with 'C' |
| **Filtered Zero/Negative Prices** | **71** | Invalid prices $\le 0$ |
| **Total Clean Revenue** | **£17,743,429.18** | Net sales across two financial years |
| **Total Unique Invoices** | **36,969** | Distinct order checkouts |
| **Total Unique Customers** | **5,878** | Identified customer accounts |
| **Repeat Customers (Freq > 0)** | **4,189** | 71.3% of all registered customers |
| **One-Time Buyers (Freq = 0)** | **1,689** | 28.7% of customer base |
| **Average Order Value (AOV)** | **£479.95** | Total Revenue / Total Invoices |
| **Average Revenue per Customer** | **£3,018.62** | Total Revenue / Total Customers |
| **Date Range** | **Dec 1, 2009 – Dec 9, 2011** | 24 full calendar months |
| **Predicted 6-Month CLV Total** | **£4,347,297.96** | Total portfolio future value |
| **Average Predicted 6-Month CLV** | **£739.59** | Mean customer future value |
| **High Value Segment Size** | **1,999 customers (34.0%)** | Avg CLV: £1,976.44 (90.9% of total CLV) |
| **Medium Value Segment Size** | **1,939 customers (33.0%)** | Avg CLV: £201.05 (9.0% of total CLV) |
| **Low Value Segment Size** | **1,940 customers (33.0%)** | Avg CLV: £3.37 (0.1% of total CLV) |
| **Lifetimes Holdout MAE** | **£487.32** | 3-Month holdout error |
| **Lifetimes Holdout RMSE** | **£3,178.46** | 3-Month holdout error |
| **XGBoost Holdout MAE** | **£622.59** | 3-Month holdout error |
| **XGBoost Holdout RMSE** | **£3,351.76** | 3-Month holdout error |
| **BG/NBD Fitted Parameters** | $r=0.6651, \alpha=63.11, a=0.0735, b=1.2286$ | Mathematical model fit parameters |
