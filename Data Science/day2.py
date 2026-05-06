
# Function to calculate average
def average(a, b, c):
    return (a + b + c) / 3

# Input Student marks
m1 = int(input("Enter marks Student 1: "))
m2 = int(input("Enter marks Student 2: "))
m3 = int(input("Enter marks Student 3: "))

# Calculate average marks of Student
avg = average(m1, m2, m3)

# Check result
if avg >= 75:
    grade = "Distinction"
elif avg >= 35:
    grade = "Pass"
else:
    grade = "Fail"

# Output
print("Average Marks:", avg)
print("Result:", grade)