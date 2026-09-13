import numpy as np

# Transportation cost matrix
cost = np.array([
    [19, 30, 50],
    [70, 30, 40],
    [40,  8, 70]
], dtype=float)

supply = [7, 9, 18]
demand = [5, 8, 21]

supply = np.array(supply, dtype=float)
demand = np.array(demand, dtype=float)

m, n = cost.shape


# -------------------------------
# Vogel's Approximation Method
# -------------------------------

def VAM(cost, supply, demand):

    allocation = np.zeros_like(cost)

    supply = supply.copy()
    demand = demand.copy()

    active_rows = [True] * m
    active_cols = [True] * n

    while np.any(supply > 0) and np.any(demand > 0):

        row_penalty = np.full(m, -1.0)
        col_penalty = np.full(n, -1.0)

        # Row penalties
        for i in range(m):
            if active_rows[i]:

                values = [
                    cost[i][j]
                    for j in range(n)
                    if active_cols[j] and demand[j] > 0
                ]

                if len(values) >= 2:
                    values.sort()
                    row_penalty[i] = values[1] - values[0]
                elif len(values) == 1:
                    row_penalty[i] = values[0]

        # Column penalties
        for j in range(n):
            if active_cols[j]:

                values = [
                    cost[i][j]
                    for i in range(m)
                    if active_rows[i] and supply[i] > 0
                ]

                if len(values) >= 2:
                    values.sort()
                    col_penalty[j] = values[1] - values[0]
                elif len(values) == 1:
                    col_penalty[j] = values[0]

        max_row = np.max(row_penalty)
        max_col = np.max(col_penalty)

        # Select row or column with maximum penalty
        if max_row >= max_col:

            i = np.argmax(row_penalty)

            j = min(
                [j for j in range(n)
                 if active_cols[j] and demand[j] > 0],
                key=lambda j: cost[i][j]
            )

        else:

            j = np.argmax(col_penalty)

            i = min(
                [i for i in range(m)
                 if active_rows[i] and supply[i] > 0],
                key=lambda i: cost[i][j]
            )

        # Allocate
        quantity = min(supply[i], demand[j])

        allocation[i][j] = quantity

        supply[i] -= quantity
        demand[j] -= quantity

        # Cross out satisfied row/column
        if supply[i] == 0:
            active_rows[i] = False

        if demand[j] == 0:
            active_cols[j] = False

    return allocation


# -------------------------------
# MODI Method
# -------------------------------

def MODI(cost, allocation):

    while True:

        basic = allocation > 0

        # Calculate u and v
        u = [None] * m
        v = [None] * n

        u[0] = 0

        changed = True

        while changed:
            changed = False

            for i in range(m):
                for j in range(n):

                    if basic[i][j]:

                        if u[i] is not None and v[j] is None:
                            v[j] = cost[i][j] - u[i]
                            changed = True

                        elif u[i] is None and v[j] is not None:
                            u[i] = cost[i][j] - v[j]
                            changed = True

        # Calculate opportunity costs
        delta = np.zeros((m, n))

        for i in range(m):
            for j in range(n):

                if not basic[i][j]:
                    delta[i][j] = cost[i][j] - (u[i] + v[j])

        print("\nU values:", u)
        print("V values:", v)

        print("\nOpportunity Cost Matrix:")
        print(np.round(delta, 2))

        # For minimization:
        # If all delta >= 0, solution is optimal
        if np.all(delta >= -1e-9):
            break

        # Select most negative opportunity cost
        entering = np.unravel_index(
            np.argmin(delta),
            delta.shape
        )

        print("Entering cell:", entering)

        # Find closed loop
        loop = find_loop(allocation, entering)

        # Alternate + and -
        minus_cells = loop[1::2]

        theta = min(
            allocation[i][j]
            for i, j in minus_cells
        )

        # Update allocation
        for k, (i, j) in enumerate(loop):

            if k % 2 == 0:
                allocation[i][j] += theta
            else:
                allocation[i][j] -= theta

        # Remove zero allocation from basis
        for i, j in minus_cells:

            if allocation[i][j] == 0:
                basic[i][j] = False
                break

        basic[entering] = True

    return allocation


def find_loop(allocation, start):

    # Simple loop search for transportation table
    basic = allocation > 0
    basic[start] = True

    path = [start]

    def search(current):

        if len(path) >= 4 and current == start:
            return True

        i, j = current

        # Move horizontally
        for col in range(n):

            cell = (i, col)

            if col != j and (basic[i][col] or cell == start):

                if cell == start and len(path) >= 4:
                    return True

                if cell not in path:

                    path.append(cell)

                    if search(cell):
                        return True

                    path.pop()

        # Move vertically
        for row in range(m):

            cell = (row, j)

            if row != i and (basic[row][j] or cell == start):

                if cell == start and len(path) >= 4:
                    return True

                if cell not in path:

                    path.append(cell)

                    if search(cell):
                        return True

                    path.pop()

        return False

    search(start)

    return path


# -------------------------------
# Main Program
# -------------------------------

allocation = VAM(cost, supply, demand)

print("Initial Solution using VAM:")
print(allocation)

initial_cost = np.sum(allocation * cost)

print("Initial Transportation Cost =", initial_cost)

allocation = MODI(cost, allocation)

print("\nOptimal Transportation Plan:")
print(allocation)

optimal_cost = np.sum(allocation * cost)

print("Minimum Transportation Cost =", optimal_cost)