"""
created: May 2020
author: Bernardo Costa
email: bernamdc@gmail.com
"""

import numpy as np
import scipy.optimize
import scipy.linalg
from sympy import symbols, solve, linear_eq_to_matrix, Matrix, solve_linear_system, parse_expr

def intertools_product(*args, repeat=1):
    """Copied from intertools.product (not in Anaconda by default)"""
    # product('ABCD', 'xy') --> Ax Ay Bx By C_SOH_x C_SOH_y Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def cons_poly_fit(data_in, data_ind_out, data_ind_bounds, degree, ineq_constraint, other_constraint, degree_type, minimize_method='trust-constr', init_guess='zeros'):
    """
    (Constrained) Polynomial fitting of the data_in given, evaluated at the data_ind_out, giving data_dep_out. Constraints can be included.

    :param degree_type:
    :param data_in: Data input. np.array with shape (n_ind_var+1,n_data) with n_ind_var as the number of independent variables (coordinates)
    + 1 dependent variable (data value at those coordinates), and with n as the number of data_in points.
    :param data_ind_out: Independent data (coordinates) output. Where the new polynomial predictions will be evaluated. Shape (n_ind_var, n_out_data)
    :param data_ind_bounds: np.array with shape (v,2) describing, respectively, the lower and upper bounds of each independent variable.
    data_bounds are necessary to assert a constraint over that bounded domain (e.g. positivity between lower and upper bound of each variable).
    :param degree: Natural number as an integer. Polynomial degree (order). 2 or higher.
    :param ineq_constraint: Inequality constraint. Possibilites:
     False        -> No inequality constraints
     'positivity' -> Force the polynomial to be always positive
     'negativity' -> Force the polynomial to be always negative
     ... (other possible constraints can be implemented)
    :param other_constraint: List of equality constraints. NOTE: Only working for 3D data (2 independent variables + 1 dep. variable). Possibilites:
     False                     -> No equality constraints
     'F_is_0_at_x0_start'      -> Force polynomial to 0, whenever the first independent variable x0 is at the lower bound (which will be re-scaled to 0).
     'F_is_0_at_x0_end'        -> Force polynomial to 0, whenever the first independent variable x0 is at the upper bound (which will be re-scaled to 1).
     'dF/dx0_is_0_at_x0_start' -> Force dF/dx0 to 0, whenever the first independent variable x0 is at the lower bound (which will be re-scaled to 0).
     'dF/dx0_is_0_at_x0_end'   -> Force dF/dx0 to 0, whenever the first independent variable x0 is at the upper bound (which will be re-scaled to 1).
     'dF/dx1_is_0_at_x0_end'   -> Force dF/dx1 to 0, whenever the first independent variable x0 is at the upper bound (which will be re-scaled to 1).
      ... (other possible constraints can be implemented)
    :param degree_type: total degree of a multivariate polynomial is different from the maximum degree. Possibilities:
     'total' -> total degree of polynomial   (e.g.: (lower order terms) + x0**2 * x1**2 has tot. degree 4). Coef. of monomials with higher degrees = 0.
     'max'   -> maximum degree of polynomial (e.g.: (lower order terms) + x0**2 * x1**2 has max. degree 2). All coefficients are kept
    :param minimize_method: Method used in the scipy.optimize.minimize (see https://docs.scipy.org/doc/scipy/reference/optimize.html )
    :param init_guess: Initial guess of the polynomial coefficients: 'ones', 'zeros' or 'random', or array with equal size to n_coef
    :return: polynomial coefficients

    Fitting a polynomial is in other words to minimize ||Ax-b||, i.e. the least sqares method. Where:
    - Ax is the polynomial expression. 2-variable 2nd degree example: C00 + C01*x2 + C02*x2**2 + C10*x1 + C11*x1*x2 + C12*x1*x2**2 + ... + C22*x1**2*x2**2;
    - x is the vector with the polynomial coefficients, e.g. [C00, C01, C02, C10, ... ];
    - A is the necessary vector or matrix to make Ax=B happen. e.g. [[1, x2, x2**2, x1, x1*x2, x1*x2**2,..., x1**2*x2**2], [same, but new data point],...];
    - B is the vector with the dependent data_in to be fitted e.g. [data_in[-1,0], data_in[-1,1], data_in[-1,2], data_in[-1,3]...]
    """

    data_ind = data_in[:-1, :]  # independent variables of the data_in. e.g. coordinates: x0, x1, x2,...
    data_dep = data_in[-1, :]  # dependent variable of the data_in. e.g. data itself: y(x0,x1,x2,...)
    n_ind_var = data_ind.shape[0]  # number of independent variables (both in the input and output)

    if not other_constraint:
        other_constraint = []  # converts a possible False to []

    # Re-scaling all the independent variables from the intervals [data lower bound, data upper bound] to [0,1]
    # Independent data needs to be re-scaled to the [0,1] interval, for this preposition on constraints to work (see https://hal.inria.fr/hal-01073514v)
    data_ind_01 = np.array([(data_ind[n,:] - data_ind_bounds[n, 0]) / (data_ind_bounds[n, -1] - data_ind_bounds[n, 0]) for n in range(n_ind_var)])
    data_ind_out_01 = np.array([(data_ind_out[n, :] - data_ind_bounds[n, 0]) / (data_ind_bounds[n, -1] - data_ind_bounds[n, 0]) for n in range(n_ind_var)])

    # The total number of coefficients in a complete multivariate polynomial is obtained as:
    n_coef = np.prod(([degree + 1] * n_ind_var))

    # Naming each coefficient, according to the respective variables and their exponents. Generic monomial: 'Cijk... * x1**i  * x2**j * x3**k...'
    coef_str_list = []
    for c in range(n_coef):
        unpadded = np.base_repr(c, base=degree + 1, padding=0)
        padding_length = n_ind_var - len(unpadded)  # padding parameter in np.base_repr doesnt work well for '0'
        coef_str_list.append('0' * padding_length + unpadded)

    def A_rows(D):
        """
        :param D: Independent data point or points, e.g. data_in[:-1,0]. Shape (n_ind_var) or (n_ind_var, n_data)
        :return: One row of the A matrix (or several rows if multiple data input and output)
        """
        poly_terms = []  # list to fill with the polynomial terms
        poly_terms_str = []  # string version of poly_terms, necessary for defining the constraints
        for c in range(n_coef):
            term = []
            term_str = []
            for n, i in enumerate(coef_str_list[c]):
                term.append(D[n, :] ** int(i))
                term_str.append('x'+str(n)+'^'+str(i)+' ')
            poly_terms.append(np.product(np.array(term), axis=0))
            poly_terms_str.append(term_str)
        return np.array(poly_terms).T

    A = A_rows(data_ind_01)

    # 'x' below refers to the polynomial fitting coefficients. e.g.: C00, C01, C10... etc
    def func_to_minimize_fitting(x):  # function to be minimized, which is the least squares method
        return np.sum((np.dot(A_rows(data_ind_01), x) - data_dep) ** 2)

    def func_to_minimize_fitting_jacobian(x):  # Jacobian of function to be minimized, which is the least squares method
        # # Slower version, easier to understand:
        # partial_der = np.zeros_like(x)  # list with each partial derivative to be appended
        # for c in range(n_coef):
        #     partial_der[c] =  2*np.dot(A[:,c],np.dot(A,x) - data_dep)
        # Fast version:
        return np.einsum('ix,i->x', 2*A, np.dot(A,x) - data_dep)  # https://math.stackexchange.com/questions/2143052/%e2%88%92-2-compute-the-hessian-of-f-and-show-it-is-positive-defini/3692521#3692521

    def func_to_minimize_fitting_hessian(x):  # Hessian of function to be minimized, which is the least squares method. (Not necessary in some methods)
        return 2 * np.transpose(A) @ A  # https://math.stackexchange.com/questions/2143052/%e2%88%92-2-compute-the-hessian-of-f-and-show-it-is-positive-defini/3692521#3692521

    if init_guess == 'ones':
        C_guess = np.ones(n_coef)  # Initial guess of the polynomial fit coefficients. Change here if desired.
    elif init_guess == 'zeros':
        C_guess = np.zeros(n_coef)
    elif init_guess == 'random':
        C_guess = np.random.uniform(low=-100,high=100,size=n_coef)
    else:
        C_guess = init_guess

    # if not (ineq_constraint or other_constraint):  # if there are no constraints at all.
    #     res = scipy.optimize.minimize(func_to_minimize_fitting, C_guess, jac=func_to_minimize_fitting_jacobian)
    #     if 'Optimization terminated successfully' not in res.message:
    #         print(res.message)
    #     poly_coeff = res.x
    #     # print('Minimized function value, 2D-fit-free:')
    #     # print(res.fun)
    #     data_ind_out = np.einsum('ic,c->i', A_rows(data_ind_out_01), poly_coeff)
    #     return poly_coeff, data_ind_out

    # Expressing the polynomial with Symbolic mathematics (Sympy).
    coefs = symbols(['C' + c for c in coef_str_list], real=True)
    ind_vars = symbols(['x' + str(v) for v in range(n_ind_var)], real=True)
    # Building the polynomial equation, for an arbitrary degree and number of ind_vars
    poly_eq_terms = []
    for c in coefs:
        poly_eq_terms.append(c)
        for v_num, v in enumerate(ind_vars):
            poly_eq_terms[-1] = poly_eq_terms[-1] * v ** int(str(c)[v_num + 1])
    poly_eq = sum(poly_eq_terms)

    # Initializing
    eq_cons = []  # to be list of constraint equations
    eq_cons_sol_dict = []  # to be list of dictionaries with the solutions of eq_cons
    ineq_cons_mat = []  # matrix (n_ineq_cons, n_coef) to represent the inequality constraints as a system of equations.
    list_eq_and_ineq_cons_obj = []  # list of the final constraint objects (one constraint object (scipy.optimize.LinearConstraint) for equality, and one for inequality, if they exist)

    # "Total" vs "Maximum" polynomial degree.
    if n_ind_var != 2: raise Warning('Equality constraints are only implemented for 2 independent variables! The script needs to be upgraded.')
    if degree_type == 'total':
        # Limiting the number of coefficients, from the 'max' degree polynomial (x**2*y**2 allowed in degree 2), to the 'total' degree polynomial (x**2*y**2 not allowed in degree 2)
        mono_degrees = []  # degrees of each monomial.
        mono_idx_discard = []  # indexes of monomials to discard, due to total degree being higher than the desired total degree.
        for c in coef_str_list:  # e.g. c = '00'... = '01' ... = '02'
            mono_degrees.append(np.sum(list(map(int, [i for i in c]))))  # converting string to list of integers and summing to get degree of each monomial
        for i, m in enumerate(mono_degrees):
            if m > degree:
                mono_idx_discard.append(i)  # e.g. in 'total' degree 3: [7,10,11,13,14,15] -> indexes of the coefficients to be forced to 0.
        for i in mono_idx_discard:
            eq_cons_sol_dict.append({coefs[i]: 0})  # dictionary with the coefficients to be set to 0 (those with "total degree" above "degree")

    # Calculate equality constraints (using a SLOW function Solve, or instead the known solution that I copied from Solve):
    assert (degree_type == 'max' or not other_constraint), 'STOP: only degree_type == max is implemented in each eq. constraint using FAST given solutions. Otherwise SLOW sympy.Solve should be used and the code adjuted'

    if 'F_is_0_at_x0_start' in other_constraint:
        if degree == 4:
            solution = {coefs[1]: 0, coefs[2]: 0, coefs[3]: 0, coefs[4]: 0, coefs[0]: 0}  # solution obtained by manually running sympy.Solve (which is slow). This is to speed up the code
        elif degree == 3:
            solution = {coefs[1]: 0, coefs[2]: 0, coefs[3]: 0, coefs[0]: 0}  # solution obtained by manually running sympy.Solve (which is slow). This is to speed up the code
        else:
            eq_cons.append(poly_eq.subs(ind_vars[0], 0) - 0)  # <---- Change here the constraint as desired. Constraint equation (<=> eq_cons = 0)
            solution = solve(eq_cons[-1], coefs, dict=True, simplify=False, rational=False)[0]
        eq_cons_sol_dict.append(solution)  # solve eq_cons (can be slow)

    if 'F_is_0_at_x0_end' in other_constraint:
        # Constraint equation (<=> eq_cons = 0):
        if degree == 4:
            solution = {coefs[1]: -coefs[6] - coefs[11] - coefs[16] - coefs[21], coefs[2]: -coefs[7] - coefs[12] - coefs[17] - coefs[22], coefs[3]: -coefs[8] - coefs[13] - coefs[18] - coefs[23],
                        coefs[4]: -coefs[9] - coefs[14] - coefs[19] - coefs[24], coefs[0]: -coefs[5] - coefs[10] - coefs[15] - coefs[20]}
        else:
            eq_cons.append(poly_eq.subs(ind_vars[0], 1) - 0)  # <---- Change here the constraint as desired. Constraint equation (<=> eq_cons = 0)
            solution = solve(eq_cons[-1], coefs, dict=True, simplify=False, rational=False)[0]
        eq_cons_sol_dict.append(solution)  # solve eq_cons (can be slow)

    if 'F_is_0_at_x1_start' in other_constraint:
        # Constraint equation (<=> eq_cons = 0):
        if degree == 4:
            solution = {coefs[5]: 0, coefs[10]: 0, coefs[15]: 0, coefs[20]: 0, coefs[0]: 0}
        elif degree == 3:
            solution = {coefs[4]: 0, coefs[8]: 0, coefs[12]: 0, coefs[0]: 0}
        else:
            eq_cons.append(poly_eq.subs(ind_vars[1], 0) - 0)  # <---- Change here the constraint as desired. Constraint equation (<=> eq_cons = 0)
            solution = solve(eq_cons[-1], coefs, dict=True, simplify=False, rational=False)[0]
        eq_cons_sol_dict.append(solution)  # solve eq_cons (can be slow)

    if 'F_is_0_at_x1_end' in other_constraint:
        # Constraint equation (<=> eq_cons = 0):
        if degree == 4:
            solution = {coefs[5]: -coefs[6] - coefs[7] - coefs[8] - coefs[9], coefs[10]: -coefs[11] - coefs[12] - coefs[13] - coefs[14], coefs[15]: -coefs[16] - coefs[17] - coefs[18] - coefs[19],
                        coefs[20]: -coefs[21] - coefs[22] - coefs[23] - coefs[24], coefs[0]: -coefs[1] - coefs[2] - coefs[3] - coefs[4]}
        elif degree == 3:
            solution = {coefs[4]: -coefs[5] - coefs[6] - coefs[7], coefs[8]: -coefs[9] - coefs[10] - coefs[11], coefs[12]: -coefs[13] - coefs[14] - coefs[15],coefs[0]: -coefs[1] - coefs[2] - coefs[3]}

        else:
            eq_cons.append(poly_eq.subs(ind_vars[1], 1) - 0)  # <---- Change here the constraint as desired. Constraint equation (<=> eq_cons = 0)
            solution = solve(eq_cons[-1], coefs, dict=True, simplify=False, rational=False)[0]
        eq_cons_sol_dict.append(solution)  # solve eq_cons (can be slow)

    if 'F_is_0_at_x0_end_at_x1_middle' in other_constraint:
        # Constraint equation (<=> eq_cons = 0):
        if degree == 4:
            solution = {coefs[0]: -0.5 * coefs[1] - 0.25 * coefs[2] - 0.125 * coefs[3] - 0.0625 * coefs[4] - coefs[5] - 0.5 * coefs[6] - 0.25 * coefs[7] - 0.125 * coefs[8] - 0.0625 * coefs[9] - coefs[
                10] - 0.5 * coefs[11] - 0.25 * coefs[12] - 0.125 * coefs[13] - 0.0625 * coefs[14] - coefs[15] - 0.5 * coefs[16] - 0.25 * coefs[17] - 0.125 * coefs[18] - 0.0625 * coefs[19] - coefs[
                                      20] - 0.5 * coefs[21] - 0.25 * coefs[22] - 0.125 * coefs[23] - 0.0625 * coefs[24]}
        else:
            eq_cons.append(poly_eq.subs(ind_vars[0], 1).subs(ind_vars[1], 0.5) - 0)  # <---- Change here the constraint as desired. Constraint equation (<=> eq_cons = 0)
            solution = solve(eq_cons[-1], coefs, dict=True, simplify=False, rational=False)[0]
        eq_cons_sol_dict.append(solution)  # solve eq_cons (can be slow)

    if 'F_is_-2_at_x1_start' in other_constraint:
        # Constraint equation (<=> eq_cons = 0):
        if degree == 4:
            solution = {coefs[5]: 0, coefs[10]: 0, coefs[15]: 0, coefs[20]: 0, coefs[0]: -1.90000000000000}
        else:
            eq_cons.append(
                poly_eq.subs(ind_vars[1], 0) + 1.9)  # <---- Change here the constraint as desired (e.g. explanation: ind_vars[0] = beta; ind_vars[1] = theta; 0 = -90deg. f(theta=-90deg)=1.2)
            solution = solve(eq_cons[-1], coefs, dict=True, simplify=False, rational=False)[0]
        eq_cons_sol_dict.append(solution)  # solve eq_cons (can be slow)

    if 'F_is_2_at_x1_end' in other_constraint:
        # Constraint equation (<=> eq_cons = 0):
        if degree == 4:
            solution = {coefs[5]: -1.0 * coefs[6] - 1.0 * coefs[7] - 1.0 * coefs[8] - 1.0 * coefs[9], coefs[10]: -1.0 * coefs[11] - 1.0 * coefs[12] - 1.0 * coefs[13] - 1.0 * coefs[14],
                        coefs[15]: -1.0 * coefs[16] - 1.0 * coefs[17] - 1.0 * coefs[18] - 1.0 * coefs[19], coefs[20]: -1.0 * coefs[21] - 1.0 * coefs[22] - 1.0 * coefs[23] - 1.0 * coefs[24],
                        coefs[0]: -1.0 * coefs[1] - 1.0 * coefs[2] - 1.0 * coefs[3] - 1.0 * coefs[4] + 1.9}
        else:
            eq_cons.append(poly_eq.subs(ind_vars[1], 1) - 1.9)  # <---- Change here the constraint as desired. Constraint equation (<=> eq_cons = 0)
            solution = solve(eq_cons[-1], coefs, dict=True, simplify=False, rational=False)[0]
        eq_cons_sol_dict.append(solution)  # solve eq_cons (can be slow)

    if 'dF/dx0_is_0_at_x0_start' in other_constraint:
        # Constraint equation (<=> eq_cons = 0):
        if degree == 4:
            solution = {coefs[6]: 0, coefs[7]: 0, coefs[8]: 0, coefs[9]: 0, coefs[5]: 0}
        else:
            eq_cons.append(poly_eq.diff(ind_vars[0]).subs(ind_vars[0], 0) - 0)  # <---- Change here the constraint as desired. Constraint equation (<=> eq_cons = 0)
            solution = solve(eq_cons[-1], coefs, dict=True, simplify=False, rational=False)[0]
        eq_cons_sol_dict.append(solution)  # solve eq_cons (can be slow)

    if 'dF/dx0_is_0_at_x0_end' in other_constraint:
        # Constraint equation (<=> eq_cons = 0):
        if degree == 4:
            solution = {coefs[6]: -2 * coefs[11] - 3 * coefs[16] - 4 * coefs[21], coefs[7]: -2 * coefs[12] - 3 * coefs[17] - 4 * coefs[22], coefs[8]: -2 * coefs[13] - 3 * coefs[18] - 4 * coefs[23],
                        coefs[9]: -2 * coefs[14] - 3 * coefs[19] - 4 * coefs[24], coefs[5]: -2 * coefs[10] - 3 * coefs[15] - 4 * coefs[20]}
        elif degree == 3:
            solution = {coefs[5]: -2*coefs[9] - 3*coefs[13], coefs[6]: -2*coefs[10] - 3*coefs[14], coefs[7]: -2*coefs[11] - 3*coefs[15], coefs[4]: -2*coefs[8] - 3*coefs[12]}
        else:
            eq_cons.append(poly_eq.diff(ind_vars[0]).subs(ind_vars[0], 1) - 0)  # <---- Change here the constraint as desired. Constraint equation (<=> eq_cons = 0)
            solution = solve(eq_cons[-1], coefs, dict=True, simplify=False, rational=False)[0]
        eq_cons_sol_dict.append(solution)  # solve eq_cons (can be slow)

    if 'dF/dx0_is_0_at_x0_end_at_x1_middle' in other_constraint:
        # Constraint equation (<=> eq_cons = 0):
        if degree == 4:
            solution = {
                coefs[5]: -0.5 * coefs[6] - 0.25 * coefs[7] - 0.125 * coefs[8] - 0.0625 * coefs[9] - 2 * coefs[10] - 1.0 * coefs[11] - 0.5 * coefs[12] - 0.25 * coefs[13] - 0.125 * coefs[14] - 3 *
                          coefs[15] - 1.5 * coefs[16] - 0.75 * coefs[17] - 0.375 * coefs[18] - 0.1875 * coefs[19] - 4 * coefs[20] - 2.0 * coefs[21] - 1.0 * coefs[22] - 0.5 * coefs[23] - 0.25 * coefs[
                              24]}
        else:
            eq_cons.append(poly_eq.diff(ind_vars[0]).subs(ind_vars[0], 1).subs(ind_vars[1], 0.5) - 0)  # <---- Change here the constraint as desired. Constraint equation (<=> eq_cons = 0)
            solution = solve(eq_cons[-1], coefs, dict=True, simplify=False, rational=False)[0]
        eq_cons_sol_dict.append(solution)  # solve eq_cons (can be slow)

    if eq_cons_sol_dict:
        A_eq_cons = Matrix([])  # empty matrix, to be filled
        b_eq_cons = Matrix([])  # empty vector, to be filled
        # Re-writting the constraint equation(s) into a scipy.optimize.LinearConstraint format (lb<=Ax<=ub). Thus, the original poly_eq can be used together with LinearConstraint when optimizing:
        for i in range(len(eq_cons_sol_dict)):  # for each set of constraints:
            eq_cons_sol_temp = [(key - value) for (key, value) in eq_cons_sol_dict[i].items()]  # coef = f(other_coefs) <=> coef - f(other_coefs) = 0 <=> 'f(coefs) = 0'
            A_temp, b_temp = linear_eq_to_matrix(eq_cons_sol_temp, coefs)  # temporary A and b, for each set of constraints
            # Updating the matrix of constraints and bounds, to feed scipy.optimize.LinearConstraint
            A_eq_cons = A_eq_cons.row_insert(pos=A_eq_cons.rows, other=A_temp)  # adding the constraints by expanding the LinearConstraint A matrix
            b_eq_cons = b_eq_cons.row_insert(pos=A_eq_cons.rows, other=b_temp)  # lb<=Ax<=ub. b=lb=ub -> equality constraint
        # Since scipy.optimize.minimize does not allow for redundant constraints, a Gaussian elimination needs to be performed.
        Ab_eq_cons = A_eq_cons.col_insert(pos=n_coef, other=b_eq_cons)  # A and b in one matrix, to feed the solve_linear_system().
        Ab_eq_cons_reduced = solve_linear_system(Ab_eq_cons, *coefs)  # the simplification and reduction of redundancies occurs here
        # Re-writting again the result from a solve_linear_system format (Ab) back into the same scipy.optimize.LinearConstraint format (A and b)
        eq_cons_sol = [(key - value) for (key, value) in Ab_eq_cons_reduced.items()]  # coef = f(other_coefs) <=> coef - f(other_coefs) = 0 <=> 'f(coefs) = 0'
        A_eq_cons, b_eq_cons = linear_eq_to_matrix(eq_cons_sol, coefs)
        A_eq_cons = np.array(A_eq_cons).astype(np.float)  # converting to array
        b_eq_cons = np.array(b_eq_cons.T)[0].astype(np.float)  # converting column vector to simple array
        # And finally getting the constraint in scipy:
        eq_cons_list_class = scipy.optimize.LinearConstraint(A_eq_cons, lb=b_eq_cons, ub=b_eq_cons)  # Object to use in scipy.optimize.minimize
        list_eq_and_ineq_cons_obj.append(eq_cons_list_class)  # List of scipy.optimize.LinearConstraint objects (maximum two elements in this list, one for eq., one for ineq. constraints).

    # Inequality contraints (Note: it should be working for a generic number of dimensions and degree, but only two independent dimensions were tested):
    # Since Sympy (v1.6.2) doesn't have multivariate polynomial inequality solvers, refer to: "Polynomial regression under shape constraints - Francois Wahl, T Espinasse (2014)"
    if ineq_constraint:
    # Number of necessary constraints (see https://hal.inria.fr/hal-01073514v)
        n_ineq_cons = np.prod([degree+1]*n_ind_var)  # this is equal to the len(coef_str_list)!
        assert n_ineq_cons == len(coef_str_list)

        # Automating the list of coefficients in each ineq_constraint:
        all_iii_list = []  # list of iii_list
        list_coef_in_constrains = [[] for _ in range(n_ineq_cons)]
        for c, j1jv in enumerate(coef_str_list):  # each ineq_constraint
            iii_list = []  # list of i_list, one for each ineq_constraint
            for n in range(n_ind_var):  # each digit
                i_list = list(range(int(j1jv[n])+1))  # list of all i such that i <= j, one for each digit
                iii_list.append(i_list)
            all_iii_list.append(iii_list)
            for l in intertools_product(*iii_list):  # list of all relevant combinations of iii
                list_coef_in_constrains[c].append(''.join(list(map(str, l))))  # just converting to str and joining digits accordingly

        # List of functions to add to the constraints
        for cons in range(n_ineq_cons):
            # Be careful with Python's "late binding closure"! SEE https://docs.python-guide.org/writing/gotchas/#late-binding-closures
            ineq_cons_mat.append(np.zeros(n_coef))
            n_cons_coef = len(list_coef_in_constrains[cons])  # number of constrained coefficients on this ineq_constraint
            for coef in range(n_cons_coef):
                idx = coef_str_list.index(list_coef_in_constrains[cons][coef])
                ineq_cons_mat[-1][idx] = 1

        ineq_cons_mat = np.array(ineq_cons_mat)  # There should be no redundant inequality constraints, with this formulation.

        if ineq_constraint == '-0.1_to_0.1':
            lower_bounds = np.ones(n_ineq_cons) * (-0.1)  # Having a number slightly different from 0 avoids non-convergences when equality and inequality constraints are used together!
            upper_bounds = np.ones(n_ineq_cons) * 0.1
        if ineq_constraint == 'positivity':
            lower_bounds = np.ones(n_ineq_cons) * (-0.001)   # Having a number slightly different from 0 avoids non-convergences when equality and inequality constraints are used together!
            upper_bounds = np.ones(n_ineq_cons) * np.inf
        if ineq_constraint == 'negativity':
            lower_bounds = np.ones(n_ineq_cons) * (-np.inf)
            upper_bounds = np.ones(n_ineq_cons) * 0.001
        if ineq_constraint == '-2_to_2':
            lower_bounds = np.ones(n_ineq_cons) * -1.2
            upper_bounds = np.ones(n_ineq_cons) * 1.2

        ineq_cons_list_class = scipy.optimize.LinearConstraint(ineq_cons_mat, lb=lower_bounds, ub=upper_bounds)
        list_eq_and_ineq_cons_obj.append(ineq_cons_list_class)

    # Minimization
    if minimize_method == 'SLSQP':
        func_to_minimize_fitting_hessian = None
        # print('SLSQP method used (Hessian information discarded)')
    # (Note: method= 'SLSQP' is used by default for a constrained problem, if no method is specified. Other methods might require the Hessian matrix (defined above but unused))
    res = scipy.optimize.minimize(func_to_minimize_fitting, C_guess, jac=func_to_minimize_fitting_jacobian, hess=func_to_minimize_fitting_hessian,
                                  constraints=list_eq_and_ineq_cons_obj, method=minimize_method, options={'maxiter':5000})
    if not (('Optimization terminated successfully' in res.message) or ('is satisfied' in res.message)):
        print(res.message)  # Prints warnings or errors
    poly_coeff = res.x
    # print('Minimized function value, 2D-fit-cons:')
    # print(res.fun)

    # Finally, the dependent data output. (Note: It was obtained from a data_ind_out scaled to [0,1], but when used with the original data_ind interval, gives expected results)
    data_dep_out = np.einsum('ic,c->i', A_rows(data_ind_out_01), poly_coeff)
    # print('Degree: '+str(degree))
    # print(poly_coeff)
    return poly_coeff, data_dep_out

