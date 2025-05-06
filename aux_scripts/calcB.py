b = 1
m = 2
visited = []
while b > 0:
    b = round(10000 / (2*(m-1)))
    if b in visited:
        m += 1
        continue
    m += 1    
    visited.append(b)
    print(f"Si M = {m} -> b = {b}")