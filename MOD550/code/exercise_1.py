import os
import pandas as pd
import matplotlib.pyplot as plt

# Point 8: Generate the assessment file
def generate_assessment_file():
    assessment_text = """Assigment 1 point 8:
Select 3 GitHub/GitLab repositories and make an assessment on the coding standards they use.

author: Urszula Starowicz
email: 283591@uis.no

Assessment of the coding standards was done with pylint. Each code was scored and analyzed by the module.
Assessment was done considering three coding standards: code readability and documentation, code formatting and style, and naming conventions.

Gitlab 1:

https://github.com/SaiPieGiera/MOD550/blob/c2ae13a98226abd6949ce557f68cc7972cf07e14/code/exercise_1.py

I. Code Readability and Documentation
    - The module lacks a module-level docstring (C0114), which should describe the script’s purpose.
    - Several functions (exercise_1.py:16:4, exercise_1.py:22:4, exercise_1.py:30:4) are missing function/method docstrings.

II. Code Formatting and Style
    - There are multiple instances of trailing whitespace (lines 30, 33, 36, 40, 63, etc.), which reduces readability and should be removed.
    - Two lines exceed the 100-character limit (lines 50 and 126).

III. Naming Conventions
    - Incorrect constant naming.

Gitlab 2:

https://github.com/dladea/MOD550/blob/03e35181e0b8de400ae30b98c7e1e6f707786473/code/exercise_1.py

I. Code Readability and Documentation
    - The script lacks a module-level docstring.

II. Code Formatting and Style
    - Missing final newline.
    - Multiple lines exceed the 100-character limit (lines 65, 129, 154).

III. Naming Conventions
    - Any issue was found.

Gitlab 3: 

https://github.com/svetaandrusenko/MOD550_Andrusenko/blob/220767e34ee333073256e89894c93c75fe54a6f6/MOD550/code/exercise_1.py

I. Code Readability and Documentation
    - TA function (line 22) lacks a docstring.

II. Code Formatting and Style
    - Whitespace is present at lines 35 and 59.
    - Lines 32 (107/100) and 63 (196/100) exceed the 100-character limit.
    - Missing final newline.

III. Naming Conventions
    - "header" (line 54) should be in UPPER_CASE to indicate it is a constant.
"""
    with open("point_8_assessment.txt", "w", encoding="utf-8") as file:
        file.write(assessment_text)
    print("Assessment file generated: point_8_assessment.txt")

# Point 7: Generate a plot with metadata and assumed model
def generate_plot():
    data_path = os.path.join("..", "data", "Dea")
    files = [f for f in os.listdir(data_path) if f.endswith(".csv")]

    if not files:
        print("No CSV files found in the dataset folder.")
        return

    file_path = os.path.join(data_path, files[0])
    DATA_FOLDER = pd.read_csv(file_path)

    if DATA_FOLDER.empty:
        print("The imported dataset is empty.")
        return

    # Assuming columns: 'x' and 'y' exist in the dataset
    if 'x' not in DATA_FOLDER.columns or 'y' not in DATA_FOLDER.columns:
        print("Dataset does not contain expected columns 'x' and 'y'.")
        return

    # Plot data
    plt.figure(figsize=(8, 6))
    plt.scatter(DATA_FOLDER['x'], DATA_FOLDER['y'], label='Imported Data', alpha=0.7)

    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Point 7: Data vs. Assumed Model')
    plt.legend()
    plt.grid(True)
    plt.savefig("point_7_plot.png")
    plt.show()
    print("Plot saved as: point_7_plot.png")

if __name__ == "__main__":
    generate_assessment_file()
    generate_plot()
