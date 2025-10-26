import pandas as pd
import pickle

# Load the updated CSV
df = pd.read_csv("job_roles_with_skills_updated.csv")

# Save it as a pickle
with open("job_roles_dataset.pkl", "wb") as f:
    pickle.dump(df, f)

print(f"✅ Pickle file updated with {df.shape[0]} job roles")
print("\nJob Roles:")
for i, role in enumerate(df['Job Role'], 1):
    print(f"{i:2d}. {role}")