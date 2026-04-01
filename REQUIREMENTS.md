# Project Specification: UFC Matchup Predictor

## 1. Objective
Build a full-stack Python application that predicts the outcome of upcoming UFC bouts. The system must fetch real-time "Upcoming Fight" data and compare two specific fighters using a trained Machine Learning model to determine a predicted winner and a confidence percentage. Be able to give reasons why someone will or will not win. You may even want to go to social media and see who fans think will win, but if you make that be sure to have a seperate data value for that, do not pollute the whole model with social media thoughts

## 2. Technical Stack
- **Language:** YOU DECIDE
- **Environment:** YOU DECIDE
- **Data Handling:** YOU DECIDE
- **Machine Learning:** YOU DECIDE
- **Data Sourcing:** YOU DECIDE
- **Visualization:** YOU DECIDE

## 3. Data Requirements & Feature Engineering
The model must account for the following features for both Fighter A and Fighter B, but add more if it helps accuracy:
- **Physical Attributes:** Height (cm), Weight (kg), Reach/Wingspan (cm), Age (Date of Birth delta).
- **Career Metrics:** Win/Loss/Draw record, Win Streak, Strength of Schedule (quality of previous opponents).
- **Combat Style (Categorical):** Strikers, Grapplers, Aggressive, Passive, or All-Rounders (One-Hot Encoded).
- **Technical Stats:** Striking accuracy/defense, Takedown accuracy/defense, Significant strikes landed per minute.
- **Head-to-Head Logic:** Historical performance against similar styles (e.g., How does Fighter A perform specifically against southpaw grapplers?).

## 4. Functional Requirements
1. **Live Data Fetching:** Automatically scrape or API-call for the next scheduled UFC event card.
2. **Preprocessing Pipeline:** Handle missing reach data (imputation) and normalize physical stats.
3. **Matchup Simulation:** Create a "Difference Matrix" ($X_A - X_B$) to feed into the classifier.
4. **Outcome Prediction:** Return a binary result (0 or 1) representing the winner, plus a probability score (e.g., "72% Confidence").

## 5. Visual Explainability (Interpretability)
For every prediction, the application must generate:
- **Feature Importance Graphs:** A bar chart showing which variables (e.g., Reach Advantage vs. Age) most influenced that specific prediction.
- **Tale of the Tape Comparison:** A radar chart or side-by-side bar graph comparing the two fighters' stats visually.
- **Historical Trend Lines:** A plot showing both fighters' performance trajectories over their last 5 fights.

## 6. Prompt to the AI Engineer
"Act as a Senior Machine Learning Engineer. Using the requirements above, generate the Python code for a `UFC_Predictor` class. Include the data cleaning pipeline, a Random Forest Classifier (or a better one if you decide another is better), and a function that takes two fighter names as input and returns a visualized breakdown of the predicted winner. Be sure to also have predictions for upcoming fights as well.  **Note: Please provide graphs and visual data (Matplotlib/Seaborn) to explain the 'Why' behind every prediction made.**"
