from mindoptpy import *

# Step 1. Create a model and change the parameters.
model = MdoModel()
MDO_INFINITY = MdoModel.get_infinity()

# Step 2. Input model.
# Change to minimization problem.
model.set_int_attr(MDO_INT_ATTR.MIN_SENSE, 1)

# Add variables.
x = []
x.append(model.add_var(0.0,         10.0, 1.0, None, "x0", True))
x.append(model.add_var(0.0, MDO_INFINITY, 1.0, None, "x1", True))
x.append(model.add_var(0.0, MDO_INFINITY, 1.0, None, "x2", True))
x.append(model.add_var(0.0, MDO_INFINITY, 1.0, None, "x3", False))

# Add constraints.
# Note that the nonzero elements are inputted in a row-wise order here.
model.add_cons(1.0, MDO_INFINITY, 1.0 * x[0] + 1.0 * x[1] + 2.0 * x[2] + 3.0 * x[3], "c0")
model.add_cons(1.0,          1.0, 1.0 * x[0]              - 1.0 * x[2] + 6.0 * x[3], "c1")

# Step 3. Solve the problem and populate the result.
model.solve_prob()
# model.display_results()

print("#################")
print(x[0].get_soln_value())
print(x[3].get_real_attr("PrimalSoln"))

# Step 4. Free the model.
model.free_mdl()

