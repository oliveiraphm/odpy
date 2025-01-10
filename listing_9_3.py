import pandas as pd


def num_digits(x):
    return len([c for c in x if c.isdigit()])

df = pd.DataFrame({'Phone': ['123-456-9890', '555-555-5555', '555-555']})
df['Num Characters'] = df['Phone'].str.len()
df['Num Digits'] = df['Phone'].apply(num_digits)
print(df)

def area_code(x):
    digits_only = ''.join([c for c in x if c.isdigit()])
    if len(digits_only) == 10:
        return digits_only[:3]
    if len(digits_only) == 11:
        return digits_only[1:4]
    return ''

def num_unique_digits(x):
    return len(set([c for c in x if c.isdigit()]))