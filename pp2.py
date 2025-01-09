import pandas as pd

# Load the dataset
file_path = "course_data.xlsx"  # Update with your file path
courses_df = pd.read_excel(file_path)

# Clean and preprocess the Review Count column
# Replace "No reviews" with 0
courses_df['Review Count'] = courses_df['Review Count'].replace("No reviews", "0")

# Remove parentheses from numeric values in Review Count
courses_df['Review Count'] = courses_df['Review Count'].str.replace(r"[()]", "", regex=True)

# Convert the Review Count column to integers
courses_df['Review Count'] = pd.to_numeric(courses_df['Review Count'], errors='coerce').fillna(0).astype(int)

# Merge fields to prepare for embedding generation
courses_df['Text For Embedding'] = (
    courses_df['Category'] + " | " + 
    courses_df['Title'] + " | " + 
    courses_df['Lesson Count'].astype(str) + " | " +
    courses_df['Price']
)

# Save the cleaned data
cleaned_data_path = "cleaned_courses_data.csv"  # Update with your desired path
courses_df.to_csv(cleaned_data_path, index=False)

print("Data cleaned and saved to:", cleaned_data_path)
