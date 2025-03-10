"""
Module: Balanced Course Assignment

This module provides a solution for assigning courses to teachers in a balanced manner while considering teacher preferences and course conflicts.

Problem Statement:
At the beginning of the semester, the head of a computer science department needs to assign courses to teachers in a balanced way. The department has `m` teachers `T={1,2,...,m}` and `n` courses `C={1,2,...,n}`. Each teacher has a preference list containing the courses they can teach based on their specialization. Additionally, some courses conflict, meaning they cannot be assigned to the same teacher due to scheduling constraints.

Objective:
- Assign `n` courses to `m` teachers such that:
  - Each assigned course is in the teacher's preference list.
  - No two conflicting courses are assigned to the same teacher.
  - The maximal teaching load among all teachers is minimized.

Input Format:
- The first line contains two integers, `m` and `n` (1 ≤ m ≤ 10, 1 ≤ n ≤ 30).
- The next `m` lines describe teacher preferences:
  - Each line starts with an integer `k` (number of courses the teacher can teach), followed by `k` integers representing the course IDs.
- The next line contains an integer `k`, the number of conflicting course pairs.
- The next `k` lines each contain two integers `i` and `j`, indicating a conflict between courses `i` and `j`.

Output Format:
- A single integer representing the minimal possible maximal load of the teachers.
- Output `-1` if no valid assignment exists.

Example:

**Input:**
```
4 12
5 1 3 5 10 12
5 9 3 4 8 12
6 1 2 3 4 9 7
7 1 2 3 5 6 10 11
25
1 2
1 3
1 5
2 4
2 5
2 6
3 5
3 7
3 10
4 6
4 9
5 6
5 7
5 8
6 8
6 9
7 8
7 10
7 11
8 9
8 11
8 12
9 12
10 11
11 12
```

**Output:**
```
3
```
"""

from ortools.linear_solver import pywraplp

def solve_course_assignment(m, n, preferences, conflicts):
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return -1, []

    x = {}
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            x[i, j] = solver.IntVar(0, 1, f"x_{i}_{j}")

    max_load = solver.IntVar(0, n, "max_load")

    for j in range(1, n + 1):
        solver.Add(solver.Sum([x[i, j] for i in range(1, m + 1)]) == 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if j not in preferences[i]:
                solver.Add(x[i, j] == 0)

    for (j1, j2) in conflicts:
        for i in range(1, m + 1):
            solver.Add(x[i, j1] + x[i, j2] <= 1)

    for i in range(1, m + 1):
        teacher_load = solver.Sum([x[i, j] for j in range(1, n + 1)])
        solver.Add(teacher_load <= max_load)

    solver.Minimize(max_load)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        solution = []
        for i in range(1, m + 1):
            teacher_courses = []
            for j in range(1, n + 1):
                if x[i, j].solution_value() > 0.5:
                    teacher_courses.append(j)
            solution.append(teacher_courses)

        return int(max_load.solution_value()), solution
    else:
        return -1, []


def main():
    m, n = map(int, input().split())

    preferences = {}
    for i in range(1, m + 1):
        parts = list(map(int, input().split()))
        preferences[i] = parts[1:]

    k = int(input())

    conflicts = []
    for _ in range(k):
        j1, j2 = map(int, input().split())
        conflicts.append((j1, j2))

    max_load, solution = solve_course_assignment(m, n, preferences, conflicts)

    print(max_load)

if __name__ == "__main__":
    main()
