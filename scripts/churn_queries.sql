-- ============================================================
-- Customer Churn Analysis — SQL Queries
-- Author  : Shaik Roshan Basha
-- Dataset : Telco Customer Churn (IBM)
-- Tool    : MySQL
-- ============================================================

-- ─────────────────────────────────────────────
-- 1. DATABASE & TABLE SETUP
-- ─────────────────────────────────────────────

CREATE DATABASE IF NOT EXISTS telco_churn_db;
USE telco_churn_db;

DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customerID        VARCHAR(20)   PRIMARY KEY,
    gender            VARCHAR(10),
    SeniorCitizen     TINYINT,
    Partner           VARCHAR(5),
    Dependents        VARCHAR(5),
    tenure            INT,
    PhoneService      VARCHAR(5),
    MultipleLines     VARCHAR(25),
    InternetService   VARCHAR(20),
    OnlineSecurity    VARCHAR(25),
    Contract          VARCHAR(25),
    PaperlessBilling  VARCHAR(5),
    PaymentMethod     VARCHAR(35),
    MonthlyCharges    DECIMAL(8,2),
    TotalCharges      DECIMAL(10,2),
    Churn             VARCHAR(5)
);

-- Load data (update path as needed)
-- LOAD DATA INFILE '/path/to/telco_churn.csv'
-- INTO TABLE customers
-- FIELDS TERMINATED BY ','
-- ENCLOSED BY '"'
-- LINES TERMINATED BY '\n'
-- IGNORE 1 ROWS;


-- ─────────────────────────────────────────────
-- 2. BASIC OVERVIEW
-- ─────────────────────────────────────────────

-- Q1: Overall churn rate
SELECT
    COUNT(*)                                          AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)   AS churned,
    SUM(CASE WHEN Churn = 'No'  THEN 1 ELSE 0 END)   AS retained,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                 AS churn_rate_pct
FROM customers;


-- Q2: Average metrics by churn status
SELECT
    Churn,
    COUNT(*)                        AS customer_count,
    ROUND(AVG(MonthlyCharges), 2)   AS avg_monthly_charge,
    ROUND(AVG(TotalCharges), 2)     AS avg_total_charge,
    ROUND(AVG(tenure), 1)           AS avg_tenure_months
FROM customers
GROUP BY Churn;


-- ─────────────────────────────────────────────
-- 3. CHURN BY CONTRACT TYPE
-- ─────────────────────────────────────────────

-- Q3: Churn rate per contract type (most impactful driver)
SELECT
    Contract,
    COUNT(*)                                                    AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)             AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                           AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)                               AS avg_monthly_charge
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- ─────────────────────────────────────────────
-- 4. CHURN BY TENURE BAND
-- ─────────────────────────────────────────────

-- Q4: Churn rate by tenure segment
SELECT
    CASE
        WHEN tenure BETWEEN 0  AND 6  THEN '0-6 months   (New)'
        WHEN tenure BETWEEN 7  AND 12 THEN '7-12 months  (Early)'
        WHEN tenure BETWEEN 13 AND 24 THEN '13-24 months (Growing)'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48 months (Mature)'
        ELSE                               '49+ months   (Loyal)'
    END                                                         AS tenure_band,
    COUNT(*)                                                    AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)             AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                           AS churn_rate_pct
FROM customers
GROUP BY tenure_band
ORDER BY MIN(tenure);


-- ─────────────────────────────────────────────
-- 5. CHURN BY INTERNET & PAYMENT
-- ─────────────────────────────────────────────

-- Q5: Churn rate by internet service type
SELECT
    InternetService,
    COUNT(*)                                                    AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)             AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                           AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)                               AS avg_monthly_charge
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;


-- Q6: Churn rate by payment method
SELECT
    PaymentMethod,
    COUNT(*)                                                    AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)             AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                           AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;


-- ─────────────────────────────────────────────
-- 6. HIGH-RISK CUSTOMER SEGMENTS
-- ─────────────────────────────────────────────

-- Q7: Highest-risk segment — Month-to-month + Fiber + Electronic check
SELECT
    Contract,
    InternetService,
    PaymentMethod,
    COUNT(*)                                                    AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)             AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                           AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)                               AS avg_monthly_charge
FROM customers
GROUP BY Contract, InternetService, PaymentMethod
HAVING total >= 50
ORDER BY churn_rate_pct DESC
LIMIT 10;


-- Q8: At-risk customers still active (for targeted retention campaigns)
SELECT
    customerID,
    Contract,
    InternetService,
    PaymentMethod,
    tenure,
    MonthlyCharges
FROM customers
WHERE Churn = 'No'
  AND Contract        = 'Month-to-month'
  AND InternetService = 'Fiber optic'
  AND PaymentMethod   = 'Electronic check'
  AND tenure          <= 12
ORDER BY MonthlyCharges DESC
LIMIT 20;


-- ─────────────────────────────────────────────
-- 7. REVENUE IMPACT ANALYSIS
-- ─────────────────────────────────────────────

-- Q9: Monthly revenue lost due to churn
SELECT
    ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue_lost,
    COUNT(*)                      AS churned_customers,
    ROUND(AVG(MonthlyCharges), 2) AS avg_lost_per_customer
FROM customers
WHERE Churn = 'Yes';


-- Q10: Revenue at risk — active Month-to-month customers
SELECT
    ROUND(SUM(MonthlyCharges), 2)   AS revenue_at_risk,
    COUNT(*)                         AS at_risk_customers
FROM customers
WHERE Churn      = 'No'
  AND Contract   = 'Month-to-month';


-- ─────────────────────────────────────────────
-- 8. WINDOW FUNCTION — RUNNING CHURN RATE
-- ─────────────────────────────────────────────

-- Q11: Cumulative churn rate by tenure (using window function)
SELECT
    tenure,
    total_in_month,
    churned_in_month,
    ROUND(
        SUM(churned_in_month) OVER (ORDER BY tenure) * 100.0 /
        SUM(total_in_month)   OVER (ORDER BY tenure), 2
    ) AS cumulative_churn_rate_pct
FROM (
    SELECT
        tenure,
        COUNT(*)                                              AS total_in_month,
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)       AS churned_in_month
    FROM customers
    GROUP BY tenure
) AS monthly_summary
ORDER BY tenure;


-- ─────────────────────────────────────────────
-- 9. SENIOR CITIZEN ANALYSIS
-- ─────────────────────────────────────────────

-- Q12: Churn comparison — seniors vs non-seniors
SELECT
    CASE WHEN SeniorCitizen = 1 THEN 'Senior' ELSE 'Non-Senior' END AS segment,
    COUNT(*)                                                          AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)                  AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                                 AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)                                     AS avg_monthly_charge
FROM customers
GROUP BY SeniorCitizen;
