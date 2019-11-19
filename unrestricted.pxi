'''
Implementation of unrestricted algorithms.
'''

import cython

from random import randint, choice

from sage.misc.misc_c import prod
from sage.rings.real_double import RDF
from sage.rings.infinity import Infinity
from sage.matrix.constructor import matrix
from sage.functions.other import floor, ceil
from sage.matrix.special import identity_matrix
from sage.modules.free_module_element import vector
from sage.geometry.polyhedron.constructor import Polyhedron
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing

cpdef GLPKBackend make_solver(int n):
  '''
  Creates an empty model in a GLPK backend solver.

  OUTPUTS:
  - a GLPK backend
  '''

  from sage.numerical.backends.generic_backend import get_solver
  import sage.numerical.backends.glpk_backend as backend
  lp = get_solver(solver="GLPK")
  lp.solver_parameter(backend.glp_simplex_or_intopt, backend.glp_simplex_only)
  lp.set_sense(1) #Maximization
  if n > 0:
    lp.add_variables(n)

  return lp

@cython.profile(True)
cpdef void init_linear_program(GLPKBackend lp, int n):
  lp.add_variables(n) #Add variables representing the negative orthant summand
  return

@cython.profile(True)
cpdef void update_linear_program(GLPKBackend lp, MPolynomial_libsingular p, list vertices):

  cdef int n = p.parent().ngens()
  NP = p.newton_polytope() + Polyhedron(rays=(-identity_matrix(n)).rows())
  Vs = NP.vertices()
  vertices.append((lp.ncols(), Vs))
  cdef int l = len(Vs)
  lp.add_variables(l)

  #Remove previous constraints determining the point of the Minkowski sum
  cdef list rows = []
  first = False
  if lp.nrows() >= n:
    for i in xrange(lp.nrows() - n, lp.nrows()):
      rows.append(lp.row(i))
  if lp.nrows() > n:
    lp.remove_constraints(xrange(lp.nrows() - n, lp.nrows()))
  else:
    first = True

  #Insert auxiliary constraint
  lp.add_linear_constraint(list(zip(xrange(lp.ncols() - l, lp.ncols()), [1]*l)), 1.0, 1.0)

  #Insert constraint determining the point of the Minkowski sum
  cdef list var_indices, var_coefs
  for i in xrange(n):
    if first:
      var_indices = [i, i + n] + list(range(lp.ncols() - l, lp.ncols()))
      var_coefs = [-1.0, -1.0] + [ Vs[j][i] for j in xrange(l) ]
    else:
      var_indices = rows[i][0] + list(range(lp.ncols() - l, lp.ncols()))
      var_coefs = rows[i][1] + [ Vs[j][i] for j in xrange(l) ]
    lp.add_linear_constraint(list(zip(var_indices, var_coefs)), 0.0, 0.0)

  return

@cython.profile(True)
cpdef list weight_vector(GLPKBackend lp, int n):
  '''
  Returns the weight vector currently used as objective function in the linear
  programming model lp.

  INPUTS:

  - `lp` -- a GLPK linear programming model
  - `n` -- number of variables in the polynomial system

  OUTPUTS:

  - a list representing a weight vector
  '''
  cdef list w = []
  cdef int i
  #for i in xrange(lp.ncols() - n, lp.ncols()):
  #  w.append(lp.objective_coefficient(i))
  for i in xrange(0, n):
    w.append(lp.objective_coefficient(i))
  return w

@cython.profile(True)
cpdef list apply_sensitivity_range(float lower, float upper, GLPKBackend lp, int change_idx, int n):

  '''
  Returns a list of weight vectors corresponding to neighboring orders to the optimum of lp.
  '''

  cdef list vectors = []
  cdef list w

  from math import isinf

  cdef float epsilon = 0.001
  if (not isinf(lower)) and abs(lower - round(lower)) < epsilon:
    lower = float(round(lower))
  if (not isinf(upper)) and abs(upper - round(upper)) < epsilon:
    upper = float(round(upper))

  cdef float increment
  if lower == float("-inf") and upper == float("inf"):
    return vectors
  if upper == float("inf"):
    increment = ceil(lower) - 1
    if lp.objective_coefficient(change_idx) + increment <= 0:
      return vectors
  elif lower == float("-inf"):
    #Append only negative increment to list
    increment = floor(upper) + 1
    w = weight_vector(lp, n)
    w[change_idx] += increment
    vectors.append(w)
  else:
    increment = ceil(lower) - 1
    if lp.objective_coefficient(change_idx) + increment <= 0:
      #Append only positive increment to list
      increment = floor(upper) + 1
      w = weight_vector(lp, n)
      w[change_idx] += increment
      vectors.append(w)
    else:
      #Append negative increment to list
      w = weight_vector(lp, n)
      w[change_idx] += increment
      vectors.append(w)
      #Append positive increment to list
      increment = floor(upper) + 1
      w = weight_vector(lp, n)
      w[change_idx] += increment
      vectors.append(w)

  return vectors

  #assert abs(increment) > 0
  #cdef float old_value = lp.objective_coefficient(change_idx)
  #lp.objective_coefficient(change_idx, old_value + increment)

  #return weight_vector(lp, n)

cpdef void sensitivity_warm_start(GLPKBackend lp, GLPKBackend aux):

  cdef int i
  for i in xrange(aux.ncols() - 1):
    if lp.get_row_stat(i) != 1:
      aux.set_col_stat(i, 1)
    else:
      aux.set_col_stat(i, 2)
  for i in xrange(aux.nrows() - 1):
    if lp.get_col_stat(i) != 1:
      aux.set_row_stat(i, 1)
    else:
      aux.set_row_stat(i, 2)
  aux.set_row_stat(aux.nrows() - 1, 1)
  aux.set_col_stat(aux.ncols() - 1, 2)
  aux.warm_up()

cpdef list lp_column(GLPKBackend lp, int col_idx):

  cdef int i, j, idx
  cdef list column = [], row_indices, row_coeffs
  for i in xrange(lp.nrows()):
    row_indices, row_coeffs = lp.row(i)
    for idx, j in enumerate(row_indices):
      if j == col_idx:
        column.append((i, row_coeffs[idx]))
        break
  return column

cpdef list lp_bounds(GLPKBackend lp):

  cdef int i
  cdef list b = []
  for i in xrange(lp.nrows()):
    lb, ub = lp.row_bounds(i)
    if ub is not None:
      b.append(ub)
    elif lb is not None:
      b.append(lb)

  assert len(b) == lp.nrows()

  return b

@cython.profile(True)
cpdef list wide_sensitivity(GLPKBackend lp, int n, int coef = 0):
  '''
  Implements the sensitivity analysis idea from Jensen et al, 1997.
  '''

  #Make model here
  cdef int coef_change_idx, idx
  cdef int i
  if coef == 0:
    #coef_change_idx = randint(lp.ncols() - n, lp.ncols() - 1)
    #coef_change_idx = randint(0, n-1)
    m = 100000.0
    for i in xrange(n):
      if lp.objective_coefficient(i) < m:
        m = lp.objective_coefficient(i)
        idx = i
    coef_change_idx = idx
  else:
    coef_change_idx = coef
  cdef GLPKBackend glpk = make_solver(0)
  cdef int num_vars = lp.nrows()
  glpk.add_variables(num_vars, lower_bound=None) #Variables of the dual problem
  glpk.add_variable(lower_bound=None, upper_bound=None, binary=False, continuous=True,integer=False, obj=1.0) #The gamma variable
  cdef int gamma = glpk.ncols() - 1

  for i in xrange(lp.ncols()):
    if i != coef_change_idx:
      glpk.add_linear_constraint(lp_column(lp, i), lp.objective_coefficient(i), None)
    else:
      glpk.add_linear_constraint(lp_column(lp, i) + [(gamma, -1.0)], lp.objective_coefficient(i), None)

  cdef float epsilon = 0.0001
  cdef float zeta = lp.get_objective_value()
  #assert abs(zeta - c*x) < epsilon, "zeta is weird " + str(zeta) + " " + str(c*x)
  glpk.add_linear_constraint(list(zip(xrange(num_vars), lp_bounds(lp))) + [(gamma, -lp.get_variable_value(coef_change_idx))], zeta, zeta)

  #Solve for maximization
  cdef float upper
  glpk.set_sense(+1)
  sensitivity_warm_start(lp, glpk)
  try:
    glpk.solve()
    upper = glpk.get_variable_value(gamma)
  except Exception as e:
    s = str(e)
    if "no feasible" in s:
      upper = 0.0
    elif "unbounded" in s:
      upper = float("inf")
    else:
      raise e

  #Solve for minimization
  cdef float lower
  glpk.set_sense(-1)
  sensitivity_warm_start(lp, glpk)
  try:
    glpk.solve()
    lower = glpk.get_variable_value(gamma)
  except Exception as e:
    s = str(e)
    if "no feasible" in s:
      lower = 0.0
    elif "unbounded" in s:
      lower = float("-inf")
    else:
      raise e

  #print "wide sensitivity:", lower, upper
  assert lower < epsilon and upper > -epsilon, "Inconsistent sensitivity range"

  return apply_sensitivity_range(lower, upper, lp, coef_change_idx, n)

@cython.profile(True)
cpdef list find_monomials(GLPKBackend lp, MPolynomialRing_libsingular R, list vertices, int k):
  #vertices is a list with tuples (idx, tuple) where tuple is a tuple with vertices, and idx is the index of the
  #first variable in lp referring to these vertices
  cdef int n = R.ngens()
  cdef int l, i, j
  cdef float val, epsilon = 0.0001
  cdef list LTs = []
  cdef MPolynomial_libsingular LM
  for i in xrange(k):
    idx = vertices[i][0]
    l = len(vertices[i][1])
    for j in xrange(l):
      val = lp.get_variable_value(idx + j)
      if abs(val - 1.0) < epsilon:
        break
    vertex = vertices[i][1][j]
    LM = prod([ R.gens()[j]**vertex[j] for j in xrange(n)])
    LTs.append(LM)

  return LTs

@cython.profile(True)
cpdef tuple choose_simplex_ordering\
    (list G, list current_ordering, GLPKBackend lp, list vertices, str heuristic, \
     int prev_betti, int prev_hilb):

  '''

  INPUTS:

  - `G` -- the current system of generators
  - `current_ordering` -- the current weight ordering
  - `lp` -- previous GLPK linear programming model
  - `vertices` -- TODO
  - `iterations` -- number of neighbors to visit

  OUTPUTS:

  - a list of weights representing a monomial order
  '''
  global first
  cdef MPolynomialRing_libsingular R = G[0].value().parent()
  cdef MPolynomialRing_libsingular newR
  cdef int k = len(G)
  cdef int n = R.ngens()
  cdef int i, j, it = 0
  cdef list CLTs, LTs, oldLTs, w, best_w

  best_w = current_ordering

  #Transform last element of G to linear program, set objective function given by w and solve
  update_linear_program(lp, G[k-1].value(), vertices)
  lp.set_objective(best_w)
  lp.solve()

  #Get current LTs to compare with Hilbert heuristic
  newR = PolynomialRing(R.base_ring(), R.gens(), order=create_order(best_w))
  LTs = find_monomials(lp, newR, vertices, k)
  #CLTs = [ (newR.ideal(LTs).hilbert_polynomial(), newR.ideal(LTs).hilbert_series(), w ) ]
  CLTs = [ (LTs, best_w) ]

  #Try sensitivity in each variable
  for i in xrange(n):
    wlist = wide_sensitivity(lp, n, i)

    for w in wlist:
      lp.set_objective(w)
      lp.solve()
      newR = PolynomialRing(R.base_ring(), R.gens(), order=create_order(w))
      LTs = find_monomials(lp, newR, vertices, k)
      CLTs.append( (LTs, w) )

    CLTs = sort_CLTs_by_heuristic(CLTs, heuristic, True, prev_betti, prev_hilb)
    if heuristic == 'hilbert' or heuristic == 'mixed':
      if CLTs[0][0] != ():
        prev_hilb = CLTs[0][0].degree() #New Hilbert degree, IF IT IS USED by the current heuristic. Else, harmless.
    best_w = CLTs[0][2][1] #Take first improvement
    CLTs = [ CLTs[0][2] ]
    lp.set_objective(best_w)
    lp.solve()

  return best_w, vertices, prev_hilb

cdef class PopulationAlgorithm:
  cdef list population
  cdef list current_best
  cdef int population_size
  cdef int _max_value

  def __init__(self, int n, bool random=False, int max_value=10000):
    self.population_size = n
    self._max_value = max_value

    if random:
      self.population = self._random_ordering(n)
    else:
      self.population = self._spread_ordering(n)

  cdef list _random_ordering(self, int n):
    cdef list orderings = []
    for i from 0 <= i < self.population_size:
      orderings.append([ randint(1, self._max_value) for i from 0 <= i < n])
    return orderings

  cdef list _spread_ordering(self, int n):
    '''
    Creates a list of orderings that are "spread out" in the search space.
    '''
    cdef list grevlex = [ 1 ] * n
    cdef list orderings = [ grevlex ]
    cdef list new_ordering
    for i from 0 <= i < n:
      new_ordering = copy(grevlex)
      new_ordering[i] = self._max_value
      orderings.append(new_ordering)

    return orderings

  cdef list midpoint(self, list order1, list order2):
    '''
    The order that is geometrically the midpoint of the line segment passing
    through order1 and order2.
    '''
    cdef list mid = []
    for i from 0 <= i < len(order1):
      mid.append(int(order1[i] + order2[i] / 2))
    return mid

  cdef list next_ordering(self, list G):
    '''
    Chooses best ordering for current basis G.
    '''
    cdef list candidates = self.population
    cdef int i, j

    #TODO this is probably pretty heavy (too many orderings)
    #I could try sampling from this if necessary
    for i from 0 <= i < self.population_size:
      for j from 0 <= j < i:
        candidates.append(self.midpoint(self.population[i], self.population[j]))
    cdef list ordered_candidates = best_orderings_hilbert(self.population, G)

    self.population = ordered_candidates[:self.population_size]

    return ordered_candidates[0] #Ordering of best candidate

@cython.profile(True)
cpdef tuple choose_random_ordering \
    (list G, list current_ordering, str heuristic,\
     int prev_betti, int prev_hilb, int iterations=10):
  '''
  Chooses a weight vector for a term ordering for the basis ``G`` that is optimal
  with respect to the Hilbert tentative function on G among randomly generated orders.

  INPUTS:

  - `G` -- a basis of a polynomial ideal
  - `current_ordering` -- the current ordering of G, as a list of weights
  - `heuristic` -- the heuristic function to use (hilbert, betti or mixed)

  OUTPUTS:

  - a weighted ordering, as a list of weights
  '''

  cdef int n = G[0].value().parent().ngens()
  cdef list rand_weights = [ current_ordering ]
  cdef list w, CLTs, LTs, best_order
  cdef MPolynomialRing_libsingular R = G[0].value().parent()
  cdef MPolynomialRing_libsingular newR

  cdef int i
  for i in xrange(iterations):
    #Choose random vector
    w = [ randint(1, 10000) for i in xrange(n) ]
    rand_weights.append(w)

  #Compute CLTs
  CLTs = []
  for w in rand_weights:
    newR = PolynomialRing(R.base_ring(), R.gens(), order=create_order(w))
    LTs = [ newR(G[i].value()).lm() for i in xrange(len(G)) ]
    CLTs.append((LTs, w))

  #Evaluate CLTs with Hilbert function
  CLTs = sort_CLTs_by_heuristic(CLTs, heuristic, True, prev_betti, prev_hilb)
  if heuristic == 'hilbert' or heuristic == 'mixed':
    if CLTs[0][0] != ():
      prev_hilb = CLTs[0][0].degree() #New Hilbert degree, IF IT IS USED by the current heuristic. Else, harmless.
  #best_order = min_weights_by_Hilbert_heuristic(R, CLTs)
  best_order = CLTs[0][2][1]

  return best_order, prev_hilb

@cython.profile(True)
cpdef tuple choose_perturbation_ordering \
    (list G, list current_ordering, str heuristic, int prev_betti, int prev_hilb):
  '''
  Chooses a weight vector for polynomial system `G` randomly and then optimizes it
  locally for a few iterations using small perturbations.

  INPUTS:

  - `G` -- a basis of a polynomial ideal
  - `current_ordering` -- the current ordering of G, as a list of weights

  OUTPUTS:

  - a weighted ordering, as a list of weights
  '''

  cdef int n = G[0].value().parent().ngens()
  cdef list curr_w, w, LTs, CLTs
  cdef MPolynomialRing_libsingular R = G[0].value().parent()
  cdef MPolynomialRing_libsingular newR
  cdef clothed_polynomial g

  curr_w = current_ordering
  LTs = [ g.value().lm() for g in G ]
  CLTs = [ (LTs, curr_w) ]

  #Compute perturbations
  cdef int i, perturbation
  cdef int max_deg = max([ g.value().total_degree(True) for g in G])
  for i in xrange(n):

    w = curr_w[:]
    perturbation_plus = randint(1, max_deg)
    perturbation_minus = -min(randint(1, max_deg), w[i] - 1)
    for perturbation in [perturbation_plus, perturbation_minus]:

      #Find LTs w.r.t current order w
      w[i] += perturbation
      newR = PolynomialRing(R.base_ring(), R.gens(), order=create_order(w))
      LTs = [ newR(g.value()).lm() for g in G ]
      CLTs.append( (LTs, w) )

    #Choose best one among current and perturbed orders
    CLTs = sort_CLTs_by_heuristic(CLTs, heuristic, True, prev_betti, prev_hilb)
    curr_w = CLTs[0][2][1] #Work in a first improvement basis
    if heuristic == 'hilbert' or heuristic == 'mixed':
      if CLTs[0][0] != ():
        prev_hilb = CLTs[0][0].degree() #New Hilbert degree, IF IT IS USED by the current heuristic. Else, harmless.
    CLTs = [ CLTs[0][2] ]

  return curr_w, prev_hilb

cpdef list normal(v, P):
  '''
  Computes a vector in the normal cone N(v, P)
  '''

  cdef tuple inequalities = P.inequalities()
  cdef list indices = [ ieq.index() for ieq in v.incident() ]
  cdef list rays = [ -inequalities[i].A() for i in indices ]
  return list(sum(rays))

cpdef list gs_initial_ordering(list F, int sugar_type):

  cdef MPolynomialRing_libsingular R = F[0].parent()
  cdef MPolynomial_libsingular g
  cdef int n = R.ngens()

  cdef list G = [ clothed_polynomial(g, sugar_type) for g in F ]

  #Consider only half of the input polynomials for computing the polyhedron
  polyhedron = Polyhedron(rays=(-identity_matrix(n)).rows())
  for g in F:
    polyhedron += g.newton_polytope()

  #Find minimal vertex according to the Hilbert heuristic
  w1 = [ 1 ] * n
  for v in polyhedron.vertex_generator():
    w2 = normal(v, polyhedron)
    w1 = hilbert_smaller(w1, w2, G)

  return w1

cpdef list caboara_initial_ordering(list F, int sugar_type):
  '''
  Currently a simplified version of the initial ordering algorithm given in
  Caboara (1993).

  - F is a list of polynomials
  '''

  cdef MPolynomialRing_libsingular R = F[0].parent()
  cdef MPolynomial_libsingular g
  cdef clothed_polynomial cg
  cdef int n = R.ngens()
  cdef int m = len(F)
  cdef int p = int(m / 2)

  cdef list G = [ clothed_polynomial(g, sugar_type) for g in F ]

  #Consider only half of the input polynomials for computing the polyhedron
  polyhedron = Polyhedron(rays=(-identity_matrix(n)).rows())
  for cg in G[:p]:
    polyhedron += cg.value().newton_polytope()

  #Find minimal vertex according to the Hilbert heuristic
  w1 = [ 1 ] * n
  for v in polyhedron.vertex_generator():
    w2 = normal(v, polyhedron)
    w1 = hilbert_smaller(w1, w2, G)

  R = PolynomialRing(R.base_ring(), R.gens(), order=create_order(w1))
  current_Ts = [ R(cg.value()).lm() for cg in G[:p] ]

  #Now, for the remaining polynomials, use the Caboara algorithm
  #This is based on the function choose_ordering_restricted
  lp = new_linear_program(n = n)
  cdef int j = 0
  for cg in G[p:]:
    #Find which monomials of g can lead
    CLTs = possible_lts(cg.value(), set(), False)

    #Order possible lts by hilbert heuristic
    CLTs = sort_CLTs_by_heuristic_restricted(R, current_Ts, CLTs, 'hilbert')
    CLTs = [tup[len(tup)-1] for tup in CLTs]
    LTups = [tup[0] for tup in CLTs]

    #Pick the first element from LTups that is actually compatible with w1
    j = 0
    can_work = []
    found = False
    while not found:
      can_work = feasible(j, LTups, lp, set(), G, current_Ts, False)
      if len(can_work) > 0:
        found = True
      else:
        j += 1

    #TODO debug this, apparently can_work is not set
    t = <MPolynomial_libsingular>CLTs[j][1]
    current_Ts.append(t)
    lp = can_work[2]
    w1 = can_work[1]

  return w1

cpdef negative_orthant(n):
  '''
  The negative orthant of R^n
  '''
  return Polyhedron(rays=(-identity_matrix(n)).rows())

cpdef tuple choose_ordering_unrestricted(list G, old_polyhedron, str heuristic,\
                                         int m, int prev_betti, int prev_hilb):

  cdef MPolynomialRing_libsingular R = G[0].value().parent() # current ring
  cdef MPolynomialRing_libsingular newR
  cdef MPolynomial_libsingular p = G[len(G)-1].value()
  cdef clothed_polynomial g
  cdef int n = R.ngens()

  new_polyhedron = negative_orthant(n)
  for g in G[m:]:
    p = g.value()
    new_polyhedron += p.newton_polytope()
  if old_polyhedron is not None:
    new_polyhedron += old_polyhedron

  cdef list CLTs = []
  cdef list LTs = []
  for v in new_polyhedron.vertex_generator():
    w = normal(v, new_polyhedron)
    newR = PolynomialRing(R.base_ring(), R.gens(), order=create_order(w))
    LTs = [ newR(g.value()).lm() for g in G ]
    CLTs.append( (LTs, w) )
    CLTs = sort_CLTs_by_heuristic(CLTs, heuristic, True, prev_betti, prev_hilb)
    if heuristic == 'hilbert' or heuristic == 'mixed':
      if CLTs[0][0] != ():
        prev_hilb = CLTs[0][0].degree() #New Hilbert degree, IF IT IS USED by the current heuristic. Else, harmless.
    CLTs = [ CLTs[0][2] ]

  cdef list best_order = CLTs[0][1]

  return best_order, new_polyhedron, prev_hilb

@cython.profile(True)
cpdef tuple choose_regrets_ordering\
    (list G, list current_ordering, list constraints, \
     MixedIntegerLinearProgram lp, str heuristic):
  '''
  Choose an ordering in an unrestricted way but based on Perry's algorithm.
  Idea: reinserting previously computed polynomials, in order to choose new LMs for them.

  I think Perry's simplifications of constraints make so that this doesn't work
  evidence: too many cases where no constraints are added - i.e., only one LM was possible
  '''

  #STEP 1: apply Perry's algorithm on the new polynomial
  cdef int i, j, k = len(G)
  cdef MPolynomialRing_libsingular R = G[0].value().parent()
  cdef int new_beginning = lp.number_of_constraints()
  cdef tuple result = choose_ordering_restricted(G, [ G[i].value().lm() for i in xrange(k-1)], k-1, current_ordering, lp, set(), set(), False, False, False, heuristic, True)
  current_ordering = result[0]
  lp = result[1]
  bvs = result[2]
  cdef int new_end = lp.number_of_constraints()
  constraints.append((new_beginning, new_end))
  cdef MPolynomialRing_libsingular PR = PolynomialRing(R.base_ring(), R.gens(), order=create_order(current_ordering))
  for i in xrange(len(G)):
    G[i].set_value(PR(G[i].value()))

  #STEP 2: build the lists of useful and useless LTs

  ##We need to implement a criterion to decide which previous LTs are useless
  ##TODO this is not very efficient. In practice, this could be computed once and updated on changes!
  #cdef list useful_LTs = []
  cdef list useless_LTs = []
  for i in xrange(k):
    if any([ monomial_divides(G[j].value().lm(), G[i].value().lm()) for j in xrange(k) if j != i ]):
      useless_LTs.append(i) #lm of G[i] is useless
  #  else: #lm of G[i] is useful, add to list
  #    useful_LTs.append(G[i].value().lm())

  ##print useless_LTs

  ##STEP 3: Choose an old polynomial and reinsert it using Caboara and Perry's restricted criterion
  cdef int reinsert
  if len(useless_LTs) == 0:
    return result + (constraints, False)
  else:
    reinsert = choice(useless_LTs)

  #We need to keep a list of which constraints in lp correspond to which polynomials.
  #Then we remove relevant constraints from lp when a polynomial is removed and reinserted.

  #constraints is a list of tuples, each tuple with the beginning and end indices of constraints
  beginning, end = constraints[reinsert]
  if beginning != end:
    lp.remove_constraints(xrange(beginning, end))

    #Reindex stuff in constraints because some were removed
    for i in xrange(reinsert+1, len(constraints)):
      beg_i, end_i = constraints[i]
      constraints[i] = (beg_i - (end - beginning), end_i - (end - beginning))
  #Remove element from G and constraints list
  cdef clothed_polynomial g = G.pop(reinsert)
  constraints.pop(reinsert)

  #print "showing g", g.value().lm()
  G.append(g)
  new_beginning = lp.number_of_constraints()
  result = choose_ordering_restricted(G, [G[i].value().lm() for i in xrange(k-1)], k-1, current_ordering, lp, set(), set(), False, False, False, heuristic, True)
  new_end = lp.number_of_constraints()
  constraints.append((new_beginning, new_end))

  PR = PolynomialRing(R.base_ring(), R.gens(), order=create_order(result[0]))

  return result + (constraints, PR(G[k-1].value()).lm() != G[k-1].value().lm())


cpdef affine_newton_polyhedron(clothed_polynomial g):
  cdef int n = g.value().parent().ngens()
  return g.value().newton_polytope() + negative_orthant(n)

cpdef MPolynomial_libsingular poly_from_exponents (exponents, MPolynomialRing_libsingular R):
  cdef int exponent
  cdef int i = 0
  cdef MPolynomial_libsingular f = R(1)
  for exponent in exponents:
    f *= R.gens()[i] ** exponent
    i += 1
  return f

cdef class LocalSearchState:
  '''
  Class to store state for local search based algorithm.
  '''
  cdef str heuristic

  #Structures updated at each call
  cdef list newton_polyhedra
  cdef list constraints
  cdef list current_ordering
  cdef MPolynomialRing_libsingular ring
  cdef MixedIntegerLinearProgram lp

  def __init__(self, int n, list initial_ordering, str heuristic,
                MPolynomialRing_libsingular R):
    self.heuristic = heuristic
    self.newton_polyhedra = []
    self.constraints = []
    self.current_ordering = initial_ordering
    self.lp = new_linear_program(n = n)
    self.ring = R

  cdef newton_polyhedron(self, int i):
    return self.newton_polyhedra[i]

  cdef list candidates(self, int i, list LTs):
    '''
    Returns the list of candidate LTs with better heuristic value than
    the current one.
    '''
    cdef tuple all_candidates = self.newton_polyhedron(i).vertices()
    cdef list CLTs = []
    cdef int j
    for j in range(len(all_candidates)):
      CLTs.append(LTs.copy())
      CLTs[j][i] = poly_from_exponents(all_candidates[j], self.ring)

    CLTs = sort_CLTs_by_heuristic(CLTs, self.heuristic, False)

    return CLTs

  cdef void add_constraint(self, beginning, end):
    self.constraints.append((beginning, end))

  cdef void add_polynomial(self, list G):

    #Pick the best LM for new polynomial according to Caboara's algorithm
    cdef int i, k = len(G)
    cdef int new_beginning = self.lp.number_of_constraints()
    cdef tuple result = choose_ordering_restricted(G,[ G[i].value().lm() for i in xrange(k-1)], k-1, self.current_ordering, self.lp, set(), set(), False, False,False, self.heuristic, True)
    cdef int new_end = self.lp.number_of_constraints()

    #Update structures
    self.add_constraint(new_beginning, new_end)
    self.current_ordering = result[0]
    self.lp = result[1]
    self.newton_polyhedra.append(affine_newton_polyhedron(G[len(G)-1]))
    self.ring = PolynomialRing(self.ring.base_ring(), self.ring.gens(),
                               order=create_order(self.current_ordering))

    for i in xrange(len(G)):
      G[i].set_value(self.ring(G[i].value()))

  cdef void readd_constraints(self, list constraints, int i):
    '''
    Add constraints that were removed before
    '''
    cdef tuple constraint
    cdef list indices, coefs
    cdef int new_beginning = self.lp.number_of_constraints()
    for constraint in constraints:
      lb = constraint[0]
      up = constraint[2]
      indices, coefs = constraint[1]
      s = self.lp.sum(coefs[i] * self.lp[indices[i]]
                      for i in range(len(indices)))
      self.lp.add_constraint(s, max=up, min=lb)

    cdef int new_end = self.lp.number_of_constraints()
    self.constraints[i] = (new_beginning, new_end)

  cdef list remove_constraints(self, int i):
    '''
    Remove from lp the constraints of polynomial i
    Return these constraints
    '''
    cdef int start, end
    start, end = self.constraints[i]
    if start == end:
      return []
    cdef list constraints = self.lp.constraints(list(range(start, end)))

    self.lp.remove_constraints(list(range(start, end)))

    #Reindex constraints that appear after
    cdef int j
    for j in range(len(self.constraints)):
      beg_j, end_j = self.constraints[j]
      if beg_j >= end:
        self.constraints[j] = (beg_j - (end - start), end_j - (end - start))

    return constraints

  cdef void update_ordering(self, list new_ordering):
    self.current_ordering = new_ordering

  cdef void update_lp_after_i(self, lp, int i):
    cdef int start = self.lp.number_of_constraints()
    self.lp = lp
    cdef int end = self.lp.number_of_constraints()
    self.constraints[i] = (start, end)

cpdef list choose_local_ordering (list G, LocalSearchState state, int m):
  '''
  Local search dynamic function.
  Two orderings are neighbors iff they pick the same LM for all but one
  polynomial of G.
  So, in addition to finding a new ordering as in Caboara's algorithm,
  we also have a chance of changing a single LM from a previous
  polynomial.

  We cannot use the optimizations of CP, because changing in an unrestricted
  way breaks disjoint cones / boundary vectors.
  '''

  #STEP 1: Choose the new ordering the same as Caboara
  for k in range(m, len(G)): #Iterate this to be compatible with F4 reducer
    state.add_polynomial(G[:k+1])

  #STEP 2: Walk through previous polynomials, decide whether to try to change
  #their leading monomials or not.
  #Check if the changes are possible using linear programming.
  #Maybe we can keep a list of the ones that were feasible, but were not picked.
  #They have a higher chance of being feasible again.

  cdef clothed_polynomial g
  cdef int i, j
  cdef list candidates
  cdef bool found = False
  cdef list LTs = [ g.value().lm() for g in G ]
  cdef tuple can_work
  cdef MPolynomial_libsingular LTi

  for i in range(len(G) - 1):
    #Anything returned by candidates has better heuristic value than current
    candidates = state.candidates(i, LTs)
    j = 0
    LTi = LTs[i]

    if len(candidates) > 0: #If there are candidates, update lp
      removed_constraints = state.remove_constraints(i)

    candidate_exps = [ tuple(c[2][i].exponents()[0]) for c in candidates ]

    #Need to update the lp somewhere
    while not found and j < len(candidates):

      LTs[i] = candidates[j][2][i]#poly_from_exponents(candidates[j], state.ring)

      can_work = feasible(j, candidate_exps, state.lp, set(), G, LTs, False)
      if len(can_work) > 0: #Found a solution
        state.update_ordering(can_work[1])
        state.update_lp_after_i(can_work[2], i)
        found = True
      else:
        j += 1

    #Go back to original LT for i if we didn't find a better one
    if not found:
      LTs[i] = LTi
      if len(candidates) > 0: #I have to reinsert constraints
        state.readd_constraints(removed_constraints, i)
    else: #Work with first improvement
      break

  return state.current_ordering
