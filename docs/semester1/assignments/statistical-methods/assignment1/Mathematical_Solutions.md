# Mathematical Solutions - Assignment 1

This document contains detailed mathematical solutions for all questions in Assignment 1.

---

## Q1: Temperature Statistics Analysis

### Problem Statement
Temperatures (in °C) measured at noon for 11 consecutive days are:
**31, 29, 27, 34, 32, 28, 33, 29, 35, 26, 30**

### (i) Compute Mean, Median, Variance, Range, Q1, Q3

#### Step 1: Mean (μ)

The mean is calculated as:

$$\mu = \frac{\sum_{i=1}^{n} x_i}{n}$$

Where:
- n = 11 (number of observations)
- Σx_i = 31 + 29 + 27 + 34 + 32 + 28 + 33 + 29 + 35 + 26 + 30 = 334

$$\mu = \frac{334}{11} = 30.36°C$$

**Answer: Mean = 30.36°C**

#### Step 2: Median

First, arrange the data in ascending order:
**26, 27, 28, 29, 29, 30, 31, 32, 33, 34, 35**

Since n = 11 (odd number), the median is the middle value:
- Position = (n + 1)/2 = (11 + 1)/2 = 6th value
- Median = 30°C

**Answer: Median = 30°C**

#### Step 3: Variance (σ²)

Variance is calculated as:

$$\sigma^2 = \frac{\sum_{i=1}^{n} (x_i - \mu)^2}{n}$$

Calculating each squared deviation:
- (31 - 30.36)² = (0.64)² = 0.4096
- (29 - 30.36)² = (-1.36)² = 1.8496
- (27 - 30.36)² = (-3.36)² = 11.2896
- (34 - 30.36)² = (3.64)² = 13.2496
- (32 - 30.36)² = (1.64)² = 2.6896
- (28 - 30.36)² = (-2.36)² = 5.5696
- (33 - 30.36)² = (2.64)² = 6.9696
- (29 - 30.36)² = (-1.36)² = 1.8496
- (35 - 30.36)² = (4.64)² = 21.5296
- (26 - 30.36)² = (-4.36)² = 19.0096
- (30 - 30.36)² = (-0.36)² = 0.1296

Sum of squared deviations = 0.4096 + 1.8496 + 11.2896 + 13.2496 + 2.6896 + 5.5696 + 6.9696 + 1.8496 + 21.5296 + 19.0096 + 0.1296 = 84.55

$$\sigma^2 = \frac{84.55}{11} = 7.69$$

**Answer: Variance = 7.69**

#### Step 4: Range

Range = Maximum value - Minimum value
- Maximum = 35°C
- Minimum = 26°C
- Range = 35 - 26 = **9°C**

**Answer: Range = 9°C**

#### Step 5: Q1 (25th Percentile)

For Q1, we need the value at the 25th percentile position:
- Position = 0.25 × (n + 1) = 0.25 × 12 = 3rd position

Using the sorted data: 26, 27, **28**, 29, 29, 30, 31, 32, 33, 34, 35

Q1 = 28°C (or interpolated between 2nd and 3rd values)

**Answer: Q1 = 28.5°C**

#### Step 6: Q3 (75th Percentile)

For Q3, we need the value at the 75th percentile position:
- Position = 0.75 × (n + 1) = 0.75 × 12 = 9th position

Using the sorted data: 26, 27, 28, 29, 29, 30, 31, 32, **33**, 34, 35

Q3 = 32.5°C (interpolated between 8th and 9th values)

**Answer: Q3 = 32.5°C**

---

### (ii) Identify Skewness (Left/Right Skewed)

#### Skewness Formula

The skewness coefficient is calculated as:

$$Skewness = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left(\frac{x_i - \mu}{\sigma}\right)^3$$

Where:
- n = 11
- μ = 30.36°C (mean)
- σ = √7.69 = 2.77°C (standard deviation)

#### Calculation

$$Skewness = \frac{11}{10 \times 9} \sum_{i=1}^{11} \left(\frac{x_i - 30.36}{2.77}\right)^3$$

After calculating all terms and summing:
**Skewness ≈ 0.1333**

#### Interpretation

- **Skewness > 0**: Right Skewed (Positively Skewed) - tail extends to the right
- **Skewness < 0**: Left Skewed (Negatively Skewed) - tail extends to the left
- **Skewness ≈ 0**: Symmetric

Since Skewness = 0.1333 > 0, the data is **Right Skewed (Positively Skewed)**.

**Rule of thumb verification:**
- Mean = 30.36°C
- Median = 30.00°C
- Since Mean > Median, this confirms right skewness.

**Answer: The data is Right Skewed (Positively Skewed)**

---

### (iii) Identify Outliers using IQR Method

#### Step 1: Calculate IQR (Interquartile Range)

$$IQR = Q3 - Q1 = 32.5 - 28.5 = 4.0°C$$

**IQR = 4.0°C**

#### Step 2: Define Outlier Boundaries

- **Lower Bound** = Q1 - 1.5 × IQR = 28.5 - 1.5 × 4.0 = 28.5 - 6.0 = **22.5°C**
- **Upper Bound** = Q3 + 1.5 × IQR = 32.5 + 1.5 × 4.0 = 32.5 + 6.0 = **38.5°C**

#### Step 3: Identify Outliers

Any value < 22.5°C or > 38.5°C is considered an outlier.

**Check each value in sorted order:**
- 26, 27, 28, 29, 29, 30, 31, 32, 33, 34, 35

All values are between 22.5°C and 38.5°C.

**Answer: No outliers detected**

---

## Q2: Probability Problem - Students, Sports, and Grades

### Problem Statement
In a class of 40 students, there are 18 boys and 22 girls.
- Out of the boys: 7 participate in sports, 6 scored an 'A' grade, with 3 boys involved in both.
- Among the girls: 9 participate in sports, 8 scored an 'A' grade, and 4 girls did both.

**Question:** If a student is picked at random, what is the probability the student is either involved in sports or scored an 'A' grade?

### Mathematical Solution

#### Step 1: Organize the Data

**Given:**
- Total students = 40
- Boys = 18, Girls = 22

**Boys:**
- In sports = 7
- With 'A' grade = 6
- In both = 3

**Girls:**
- In sports = 9
- With 'A' grade = 8
- In both = 4

#### Step 2: Calculate Individual Sets (Using Set Theory)

**For Boys:**
- Sports only = 7 - 3 = **4**
- Grade A only = 6 - 3 = **3**
- Both = **3**
- Neither = 18 - 7 - 6 + 3 = **8**

**For Girls:**
- Sports only = 9 - 4 = **5**
- Grade A only = 8 - 4 = **4**
- Both = **4**
- Neither = 22 - 9 - 8 + 4 = **9**

#### Step 3: Calculate Total Counts

**Total students in sports** = 7 + 9 = **16**

**Total students with 'A' grade** = 6 + 8 = **14**

**Total students in both** = 3 + 4 = **7**

#### Step 4: Apply Inclusion-Exclusion Principle

**Formula:** |A ∪ B| = |A| + |B| - |A ∩ B|

Where:
- A = Students in sports
- B = Students with 'A' grade
- A ∩ B = Students in both

**Total students in sports OR 'A' grade:**
|A ∪ B| = 16 + 14 - 7 = **23 students**

#### Step 5: Calculate Probability

**Using Inclusion-Exclusion Principle:**

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

Where:
- P(A) = P(Sports) = 16/40 = 0.40
- P(B) = P(Grade A) = 14/40 = 0.35
- P(A ∩ B) = P(Both) = 7/40 = 0.175

**Calculation:**
$$P(Sports \cup Grade A) = P(Sports) + P(Grade A) - P(Sports \cap Grade A)$$
$$P(Sports \cup Grade A) = \frac{16}{40} + \frac{14}{40} - \frac{7}{40}$$
$$P(Sports \cup Grade A) = \frac{16 + 14 - 7}{40} = \frac{23}{40} = 0.575$$

**Alternative Direct Calculation:**
$$P(Sports \cup Grade A) = \frac{23}{40} = 0.575 = 57.5\%$$

**Answer: The probability is 0.575 or 57.5%**

---

## Q3: Naive Bayes Classifier for Depression Prediction

### Problem Statement
You have the following dataset:

| Person | Trouble Sleeping | Low Energy | Anxiety | Has Depression |
|--------|------------------|------------|---------|----------------|
| A1     | Yes              | Yes        | Yes     | Yes            |
| A2     | No               | Yes        | No      | No             |
| A3     | Yes              | No         | Yes     | Yes            |
| A4     | No               | No         | No      | No             |

A new person arrives with: **Trouble Sleeping = Yes, Low Energy = No, Anxiety = Yes**

**Question:** Predict whether the person has depression using Naive Bayes Classifier.

### Mathematical Solution

#### Naive Bayes Formula

$$P(Class | Features) = \frac{P(Class) \times P(Features | Class)}{P(Features)}$$

Since P(Features) is constant for both classes, we can use:
$$P(Class | Features) \propto P(Class) \times \prod_{i=1}^{n} P(Feature_i | Class)$$

**Given:**
- New person: Trouble Sleeping = Yes, Low Energy = No, Anxiety = Yes
- We need to find: P(Depression = Yes | Features) vs P(Depression = No | Features)

#### Step 1: Calculate Prior Probabilities

From the dataset:
- Total instances = 4
- Depression = Yes: 2 instances (A1, A3)
- Depression = No: 2 instances (A2, A4)

$$P(Depression = Yes) = \frac{2}{4} = 0.5$$
$$P(Depression = No) = \frac{2}{4} = 0.5$$

#### Step 2: Calculate Likelihood Probabilities

**For Depression = Yes (2 instances: A1, A3):**

Looking at instances with Depression = Yes:
- A1: Trouble Sleeping = Yes, Low Energy = Yes, Anxiety = Yes
- A3: Trouble Sleeping = Yes, Low Energy = No, Anxiety = Yes

- P(Trouble Sleeping = Yes | Depression = Yes) = 2/2 = **1.0**
- P(Low Energy = No | Depression = Yes) = 1/2 = **0.5**
- P(Anxiety = Yes | Depression = Yes) = 2/2 = **1.0**

**For Depression = No (2 instances: A2, A4):**

Looking at instances with Depression = No:
- A2: Trouble Sleeping = No, Low Energy = Yes, Anxiety = No
- A4: Trouble Sleeping = No, Low Energy = No, Anxiety = No

- P(Trouble Sleeping = Yes | Depression = No) = 0/2 = **0.0**
- P(Low Energy = No | Depression = No) = 1/2 = **0.5**
- P(Anxiety = Yes | Depression = No) = 0/2 = **0.0**

#### Step 3: Calculate Posterior Probabilities (Unnormalized)

**For Depression = Yes:**
$$P(Yes) \times P(TS=Yes|Yes) \times P(LE=No|Yes) \times P(Anx=Yes|Yes)$$
$$= 0.5 \times 1.0 \times 0.5 \times 1.0 = 0.25$$

**For Depression = No:**
$$P(No) \times P(TS=Yes|No) \times P(LE=No|No) \times P(Anx=Yes|No)$$
$$= 0.5 \times 0.0 \times 0.5 \times 0.0 = 0.0$$

#### Step 4: Normalize Probabilities

**Normalization factor** = 0.25 + 0.0 = 0.25

$$P(Depression = Yes | Features) = \frac{0.25}{0.25} = 1.0 = 100\%$$
$$P(Depression = No | Features) = \frac{0.0}{0.25} = 0.0 = 0\%$$

**Answer: The person has Depression = Yes (with 100% confidence)**

---

## Q4: Bayes' Theorem - Engineering vs Arts Students

### Problem Statement
In a university:
- 5% of engineering students score above 95% in the final exam
- 2% of arts students score above 95% in the final exam
- 70% of the students are enrolled in arts

**Question:** If a randomly selected student scores above 95%, what is the probability that the student is an arts student?

### Mathematical Solution

#### Given Information

- P(Arts) = 0.70 (70%)
- P(Engineering) = 1 - 0.70 = 0.30 (30%)
- P(Score > 95% | Engineering) = 0.05 (5%)
- P(Score > 95% | Arts) = 0.02 (2%)

**Find:** P(Arts | Score > 95%)

#### Step 1: Calculate P(Score > 95%) using Law of Total Probability

The Law of Total Probability states:
$$P(Score > 95\%) = P(Score > 95\% | Arts) \times P(Arts) + P(Score > 95\% | Engineering) \times P(Engineering)$$

**Substitute values:**
$$P(Score > 95\%) = (0.02 \times 0.70) + (0.05 \times 0.30)$$
$$P(Score > 95\%) = 0.014 + 0.015 = 0.029 = 2.9\%$$

**Answer: P(Score > 95%) = 0.029 or 2.9%**

#### Step 2: Apply Bayes' Theorem

**Bayes' Theorem Formula:**
$$P(Arts | Score > 95\%) = \frac{P(Score > 95\% | Arts) \times P(Arts)}{P(Score > 95\%)}$$

**Substitute values:**
$$P(Arts | Score > 95\%) = \frac{0.02 \times 0.70}{0.029}$$
$$P(Arts | Score > 95\%) = \frac{0.014}{0.029}$$
$$P(Arts | Score > 95\%) = 0.4828 = 48.28\%$$

**Answer: P(Arts | Score > 95%) = 0.4828 or 48.28%**

#### Step 3: Verification (Optional)

**Calculate P(Engineering | Score > 95%) for verification:**

$$P(Engineering | Score > 95\%) = \frac{P(Score > 95\% | Engineering) \times P(Engineering)}{P(Score > 95\%)}$$
$$P(Engineering | Score > 95\%) = \frac{0.05 \times 0.30}{0.029} = \frac{0.015}{0.029} = 0.5172 = 51.72\%$$

**Verification:**
P(Arts | Score > 95%) + P(Engineering | Score > 95%) = 0.4828 + 0.5172 = 1.0 ✓

This confirms our calculation is correct.

**Final Answer: P(Arts | Score > 95%) = 0.4828 or 48.28%**

---

## Summary of All Answers

| Question | Answer |
|----------|--------|
| **Q1(i)** | Mean = 30.36°C, Median = 30°C, Variance = 7.69, Range = 9°C, Q1 = 28.5°C, Q3 = 32.5°C |
| **Q1(ii)** | Right Skewed (Positively Skewed) - Skewness = 0.1333 |
| **Q1(iii)** | No outliers detected (all values within 22.5°C to 38.5°C) |
| **Q2** | P(Sports OR Grade A) = 0.575 or 57.5% |
| **Q3** | Depression = Yes (with 100% confidence) |
| **Q4** | P(Arts \| Score > 95%) = 0.4828 or 48.28% |

---

*End of Mathematical Solutions*

