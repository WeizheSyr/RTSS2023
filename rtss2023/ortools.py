from ortools.linear_solver import pywraplp

# 首先，调用CBC求解器
# 整数规划使用pywraplp.Solver.GLOP_LINEAR_PROGRAMMING
solver = pywraplp.Solver('SolveIntegerProblem',
                         pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

# 定义x和y的定义域，这里是从0到正无穷
x = solver.IntVar(0.0, solver.infinity(), 'x')
y = solver.IntVar(0.0, solver.infinity(), 'y')
# 添加约束：x+7y<17.5
constraint1 = solver.Constraint(-solver.infinity(), 17.5)
constraint1.SetCoefficient(x, 1)
constraint1.SetCoefficient(y, 7)
# 添加约束：x <= 3.5
constraint2 = solver.Constraint(-solver.infinity(), 3.5)
constraint2.SetCoefficient(x, 1)
constraint2.SetCoefficient(y, 0)
# 定义目标函数： Maximize x + 10 * y
objective = solver.Objective()
objective.SetCoefficient(x, 1)
objective.SetCoefficient(y, 10)
objective.SetMaximization()
# 获取问题的答案
result_status = solver.Solve()
# 判断结果是否是最优解
assert result_status == pywraplp.Solver.OPTIMAL
# 验证一下结果是否正确，这一步不是必要但是推荐加上
assert solver.VerifySolution(1e-7, True)
# 输出结果
print('Number of variables =', solver.NumVariables())
print('Number of constraints =', solver.NumConstraints())
print('Optimal objective value = %d' % solver.Objective().Value())
variable_list = [x, y]
for variable in variable_list:
    print('%s = %d' % (variable.name(), variable.solution_value()))
