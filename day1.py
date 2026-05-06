
# Input
name = input("Enter your name: ")
age = int(input("Enter your age: "))
salary = float(input("Enter your monthly salary: "))

# Calculating
yearly_salary = salary * 12

# Store in dictionary
data = {
    "name": name,
    "age": age,
    "monthly_salary": salary,
    "yearly_salary": yearly_salary
}

# Output
print("\nUser Details:")
print(data)