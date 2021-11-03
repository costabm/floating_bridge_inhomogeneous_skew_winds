def Taylor_polynomial_sympy(function_expression, variable_list, evaluation_point, degree):
    """
    Mathematical formulation reference:
    https://math.libretexts.org/Bookshelves/Calculus/Supplemental_Modules_(Calculus)/Multivariable_Calculus/3%3A_Topics_in_Partial_Derivatives/Taylor__Polynomials_of_Functions_of_Two_Variables
    :param function_expression: Sympy expression of the function
    :param variable_list: list. All variables to be approximated (to be "Taylorized")
    :param evaluation_point: list. Coordinates, where the function will be expressed
    :param degree: int. Total degree of the Taylor polynomial
    :return: Returns a Sympy expression of the Taylor series up to a given degree, of a given multivariate expression, approximated as a multivariate polynomial evaluated at the evaluation_point
    """
    from sympy import factorial, Matrix, prod
    import itertools

    n_var = len(variable_list)
    point_coordinates = [(i, j) for i, j in (zip(variable_list, evaluation_point))]  # list of tuples with variables and their evaluation_point coordinates, to later perform substitution

    deriv_orders = list(itertools.product(range(degree + 1), repeat=n_var))  # list with exponentials of the partial derivatives
    deriv_orders = [deriv_orders[i] for i in range(len(deriv_orders)) if sum(deriv_orders[i]) <= degree]  # Discarding some higher-order terms
    n_terms = len(deriv_orders)
    deriv_orders_as_input = [list(sum(list(zip(variable_list, deriv_orders[i])), ())) for i in range(n_terms)]  # Individual degree of each partial derivative, of each term

    polynomial = 0
    for i in range(n_terms):
        partial_derivatives_at_point = function_expression.diff(*deriv_orders_as_input[i]).subs(point_coordinates)  # e.g. df/(dx*dy**2)
        denominator = prod([factorial(j) for j in deriv_orders[i]])  # e.g. (1! * 2!)
        distances_powered = prod([(Matrix(variable_list) - Matrix(evaluation_point))[j] ** deriv_orders[i][j] for j in range(n_var)])  # e.g. (x-x0)*(y-y0)**2
        polynomial += partial_derivatives_at_point / denominator * distances_powered
    return polynomial

# Solving the exercises in section 13.7 of https://math.libretexts.org/Bookshelves/Calculus/Supplemental_Modules_(Calculus)/Multivariable_Calculus/3%3A_Topics_in_Partial_Derivatives/Taylor__Polynomials_of_Functions_of_Two_Variables
from sympy import symbols, sqrt, atan, ln

# Exercise 1
x = symbols('x')
y = symbols('y')
function_expression = x*sqrt(y)
variable_list = [x,y]
evaluation_point = [1,4]
degree=1
print(Taylor_polynomial_sympy(function_expression, variable_list, evaluation_point, degree))
degree=2
print(Taylor_polynomial_sympy(function_expression, variable_list, evaluation_point, degree))

# Exercise 3
x = symbols('x')
y = symbols('y')
function_expression = atan(x+2*y)
variable_list = [x,y]
evaluation_point = [1,0]
degree=1
print(Taylor_polynomial_sympy(function_expression, variable_list, evaluation_point, degree))
degree=2
print(Taylor_polynomial_sympy(function_expression, variable_list, evaluation_point, degree))

# Exercise 5
x = symbols('x')
y = symbols('y')
function_expression = x**2*y + y**2
variable_list = [x,y]
evaluation_point = [1,3]
degree=1
print(Taylor_polynomial_sympy(function_expression, variable_list, evaluation_point, degree))
degree=2
print(Taylor_polynomial_sympy(function_expression, variable_list, evaluation_point, degree))

# Exercise 7
x = symbols('x')
y = symbols('y')
function_expression = ln(x**2+y**2+1)
variable_list = [x,y]
evaluation_point = [0,0]
degree=1
print(Taylor_polynomial_sympy(function_expression, variable_list, evaluation_point, degree))
degree=2
print(Taylor_polynomial_sympy(function_expression, variable_list, evaluation_point, degree))

