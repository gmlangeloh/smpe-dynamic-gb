from copy import copy

import cython
import sage.numerical.backends.glpk_backend as glpk_backend

#Globals for this module
cdef double tolerance_cone = 0.01
cdef double upper_bound = 100
cdef double upper_bound_delta = 100

cdef int restricted_iterations = 0

@cython.profile(True)
cpdef MixedIntegerLinearProgram new_linear_program \
    (MixedIntegerLinearProgram lp = None, int n = 0, \
     bool minimize_homogeneous = False):
  r"""
    This tracks the number of linear programs created, and initializes them
    with a common template.
  """
  global tolerance_cone, upper_bound

  #number_of_programs_created += 1
  statistics.inc_programs_created()
  if lp is None:
    mip = MixedIntegerLinearProgram(check_redundant=True, solver="GLPK", \
                                    maximization=False)
  # when solving integer programs, perform simplex first, then integer optimization
  # (avoids a GLPK bug IIRC)
    mip.solver_parameter(glpk_backend.glp_simplex_or_intopt, \
                         glpk_backend.glp_simplex_then_intopt)

    # need positive weights
    for k in xrange(n):
      mip.add_constraint(mip[k],min=tolerance_cone)
      #mip.set_min(mip[k],tolerance_cone)
      mip.set_integer(mip[k])
      mip.set_max(mip[k],upper_bound)

    # do we hate the homogenizing variable?
    if minimize_homogeneous:
      for k in xrange(n-1):
        mip.add_constraint(mip[k] - mip[n-1],min=tolerance_cone)

    mip.set_objective(mip.sum([mip[k] for k in xrange(n)]))

    return mip

  else: return copy(lp)

@cython.profile(True)
cpdef tuple monitor_lts(list G, list LTs, list new_ordering):
  r"""
    Checks whether new_ordering changes the leading terms of G.

    INPUT:

    - ``G`` -- current basis
    - ``LTs`` -- leading terms of elements of G according to previous ordering
    - ``new_ordering`` -- new ordering

    OUTPUT:

    - 1 if polynomials whose leading monomials change; 0 otherwise
    - list of lists; if first result is nonzero, then each list in this list
      contains the exponent vectors of monomials that weigh the same or more
      than the correct leading monomial
  """
  cdef int i, j, k, n, changes
  cdef double tw, uw # monomial weights
  cdef tuple texp, uexp # monomial exponents
  cdef MPolynomial_libsingular g
  cdef list U, result, current
    # U is the set of monomials of the polynomial g currently under examination
    # current is the list of monomials of g that weigh more than the old
    #polynomial
    # result is collection of all current's

  # setup

  n = len(new_ordering)
  result = list()
  changes = 0

  # main loop

  # check each leading monomial
  for i in xrange(len(LTs)):

    texp = LTs[i].exponents(as_ETuples=False)[0]
    tw = sum([texp[k]*new_ordering[k] for k in xrange(n)])
    current = list()
    result.append(current)
    g = G[i].value(); U = g.monomials()

    # heaviest entry in U should be t
    for u in U:

      uexp = u.exponents(as_ETuples=False)[0]

      if uexp != texp:

        uw = sum([uexp[k]*new_ordering[k] for k in xrange(n)])

        if uw >= tw:

          changes = 1
          current.append(uexp)

  #print result
  return changes, result

@cython.profile(True)
cpdef set boundary_vectors(MixedIntegerLinearProgram lp, int n):
  r"""
    Finds a boundary vector for an admissible ordering defined by ``lp``,
    so that a vector satisfying ``lp`` passes between these vectors,
    WITH HIGH PROBABILITY. By this, I mean that we may lose some vectors,
    but as long as ``lp`` is defined well, this should be rare in general.
    It is ESSENTIAL that the program have at least one solution, because
    we WILL NOT check that here.

    INPUT:

    - ``lp`` -- a linear program corresponding to an admissible ordering
    - ``n`` -- number of variables in the program

    OUTPUT:

    A tuple containing a list of corner vectors that approximate the region

    ALGORITHM:

      #. find the minimum feasible degree d;
         that is, there exists a feasible solution (x1,...,xn) such that
         x1 + ... + xn = d
      #. create a cross-section of the solution cone
         by intersecting with the hyperplane x1 + ... + xn = d + 1
      #. for each variable x, compute the vectors that maximize and minimize x
         in this cross-section
  """
  cdef int i, j, k # counters
  cdef int start_constraints, end_constraints# used for deleting bad constraints
  cdef float sol_degree # degree of solution
  cdef tuple result, solution_vector
  cdef set boundaries
  cdef list lp_sol

  #print "in boundary vectors"
  np = lp
  start_constraints = np.number_of_constraints()

  # first we compute the minimum feasible degree
  # this first step should add zero overhead
  np.solve()
  lp_sol = np.get_values([np[k] for k in xrange(n)])
  sol_degree = sum(lp_sol)

  # now create the cross-section
  np.add_constraint(np.sum([np[k] for k in xrange(n)]), min=sol_degree + 1)
  np.add_constraint(np.sum([np[k] for k in xrange(n)]), max=sol_degree + 1)
  end_constraints = start_constraints + 2

  # find corners where variables are maximized and minimized
  boundaries = set()

  for k in xrange(n):

    # lp is a minimization problem, so this minimizes xk
    np.set_objective(np[k])
    #print "variable", k, "min"
    np.solve()
    boundaries.add(tuple(np.get_values([np[k] for k in xrange(n)])))
    # lp is a minimization problem, so this maximizes xk
    np.set_objective(-np[k])
    #print "variable", k, "max"
    np.solve()
    boundaries.add(tuple(np.get_values([np[k] for k in xrange(n)])))

  # now remove the cross-section
  np.remove_constraints((start_constraints,start_constraints+1))
  np.set_objective(np.sum([np[k] for k in xrange(n)]))
  np.solve()

  #print "boundaries", boundaries
  #print "leaving boundary vectors"
  return boundaries

@cython.profile(True)
cpdef int solve_real(MixedIntegerLinearProgram lp, int n):
  r"""
  Set the linear program ``lp`` to treat each of the ``n`` variables
  as a real variable, not an integer variable. Then, solve the program.
  """
  cdef int k

  for k in xrange(n): lp.set_real(lp[k])

  try: lp.solve()
  except:
    statistics.inc_failed_systems()
    return 0

  return 1

@cython.profile(True)
cpdef int solve_integer(MixedIntegerLinearProgram lp, int n):
  r"""
  Set the linear program ``lp`` to treat each of the ``n`` variables
  as an integer variable, not a real variable. Then, solve the program.

  A difficulty in using glpk is that it requires an upper bound in order to find
  an integer solution in a decent amount of time.
  As a result, we have set up our linear program with such an upper bound.
  However, while the real solution can be found with even a fairly small upper
  bound, the integer solution will at times exceed the previously set upper
  bound.
  This poses a challenge.

  This function is not called if there is not a real solution,
  and the structure of our feasible regions implies that
  if there is a real solution, then there is also an integer solution.
  If no solution is found here, then, the problem must be that our upper bound
  is too low.
  """
  global upper_bound, upper_bound_delta
  cdef int k, passed
  cdef double old_upper_bound

  for k in xrange(n):
    lp.set_integer(lp[k])
    lp.set_max(lp[k], upper_bound)

  # try to get solution
  passed = False
  old_upper_bound = upper_bound

  while not passed:

    # first find ordering
    #print "obtaining ordering below", upper_bound

    try:

      lp.solve()
      passed = True

    except:

      # upper bound too low, so raise it
      #print "failed"
      upper_bound += upper_bound_delta

      if upper_bound > 128000: # even I give up after a certain point
          #need a better solver?

        #print "returning no solution", upper_bound
        for k in xrange(n): lp.set_real(lp[k])
        upper_bound = old_upper_bound
        #failed_systems += 1
        statistics.inc_failed_systems()
        return 0

      for k in xrange(n): lp.set_max(lp[k], upper_bound)

  return 1

@cython.profile(True)
cpdef tuple feasible(int i, list CMs, MixedIntegerLinearProgram olp, \
                     set rejects, list G, list LTs, int use_rejects):
  r"""
    Determines if the ``i``th monomial in ``CMs`` is a feasible leading monomial
    of the polynomial whose compatible monomials are listed in CMs,
    consistent with the linear program olp.

    INPUT:

    - ``i`` -- an integer (0 <= i < len(M))
    - ``CMs`` -- monomials judged to be compatible with previous choices
    - ``olp`` -- a linear program for solutions
    - ``rejects`` -- a set of linear constraints that are known to be incompatible
      with ``olp``
    - ``G`` -- the current basis of the ideal
    - ``LTs`` -- the leading terms of ``G``
    - `use_rejects` -- whether to use rejected programs (disjoint cones) to
      avoid useless systems

    OUTPUT:

    A tuple containing nothing, if there is no solution. Otherwise, it contains:

      * ``i``,
      * a vector that solves ``lp``, and
      * ``lp``, a linear program that extends ``olp`` and whose solution vector
        v satisfies v.CMs[i] > v.CMs[j] for all j=/=i.
        (Here, a.b indicates the dot product.)

    ALGORITHM:

    It constructs two linear systems, both of the form

      #. `x_k \geq \epsilon` for each `k`
      #. `\sum((alpha_k - beta_k)*x_k) \geq \epsilon` where

        #. `alpha_k` is the degree of `t` in `x_k` and
           `beta_k` is the degree of `u` from `CMs[i]` in `x_k` and
        #. epsilon is a constant used to minimize floating point error;
           its value is determined by the global `tolerance_cone`

    The first system merely checks against elements of CMs; that is,
    against the current polynomial.
    The second combines the first system with ``olp``; that is,
    against all the previous polynomials, as well.

  """
  # the following line is needed to control the solving process
  import sage.numerical.backends.glpk_backend as glpk_backend

  global tolerance_cone, upper_bound

  cdef int j, k, l # counters
  cdef int passed, lms_changed # signals
  cdef int n, start_constraints, end_constraints, number_of_constraints # measures
  cdef float sol_degree, a
  cdef tuple t, u, v, old_lm, new_lm # exponent vectors
  cdef tuple constraint
  cdef list changed_lms, changed_lm # used for when we change the ordering,
  #rather than refine it

  cdef set new_constraints = set()
  cdef set new_rejects = set()

  t = <tuple>CMs[i]
  #print "testing", t
  n = len(t)
  cdef tuple result = tuple()

  # STEP 1: check if solution exists merely for this polynomial

  # set up solver to solve LP relaxation first
  # (GLPK chokes on no integer solution)
  cdef MixedIntegerLinearProgram lp = new_linear_program(n = n)
  #cdef MixedIntegerLinearProgram lp = new_linear_program()
  #lp.solver_parameter(glpk_backend.glp_simplex_or_intopt, \
  #                    glpk_backend.glp_simplex_then_intopt)

  ## xi >= epsilon
  #for k in xrange(n): lp.add_constraint(lp[k],min=tolerance_cone)
  ## minimize x1 + ... + xn
  #lp.set_objective(lp.sum([lp[k] for k in xrange(n)]))
  ##print "initial constraints set"

  # add constraints
  cdef int m = len(CMs)
  #print m, "new constraints", CMs


  #loop through all exponent vectors of compatible monomials
  for j in xrange(m):

    u = <tuple>CMs[j] # get current exponent vector

    if t != u: # don't check against oneself!

      constraint = tuple([float(t[k]-u[k]) for k in xrange(n)])
      new_constraints.add(constraint)

      # check whether adding constraint to this system is known to be
      #incompatible
      if use_rejects:
        #print "checking rejects",
        for c in (rejects):

          if c.issubset(new_constraints):

            #print "rejected", new_constraints, c
            #print "rejected!!!"
            #rejections += 1
            statistics.inc_rejections()
            return result

      #print "checked rejects"
      lp.add_constraint(lp.sum([constraint[k]*lp[k] for k in xrange(n)]), \
                        min=tolerance_cone)

      try:

        #print "LP relaxation"
        lp.solve()

      except: # unable to solve; monomial cannot lead even this polynomial

        #failed_systems += 1
        statistics.inc_failed_systems()
        rejects.add(frozenset(new_constraints)) # remember this failed system
        return result

  # we have a solution to the LP relaxation
  #print "have solution"


  # STEP 2: check if system is consistent with previous systems

  # first, solve the LP relaxation
  #print "copying program"
  np = olp; start_constraints = end_constraints = np.number_of_constraints()
  #print "copied old program, which had", start_constraints, "constraints"

  # add new constraints to old constraints
  for constraint in new_constraints:

    #print t, constraint
    number_of_constraints = np.number_of_constraints()
    new_rejects.add(constraint)
    np.add_constraint(np.sum([constraint[k]*np[k] for k in xrange(n)]), \
                      min=tolerance_cone)
    # check of size is necessary because we prevent the addition of redundant constraints
    if np.number_of_constraints() > number_of_constraints: end_constraints += 1

  #print "now have", np.number_of_constraints(), "constraints"

  # moving this inside the loop might make rejection-checking more efficient,
  # but slows the process greatly for large systems

  try:

    # a lot of debugging information for when I mess things up...
    #print "trying to solve with old constraints"
    #np.show()
    np.solve()
    #print "solved"

  except: # failed to find a solution; monomial is incompatible w/previous

    #print "not solved"
    rejects.add(frozenset(new_rejects)) # remember this failed system
    #print "killing new constraints in program, which has", \
        #np.number_of_constraints(), "constraints"
    np.remove_constraints(range(start_constraints,end_constraints))
    #failed_systems += 1
    statistics.inc_failed_systems()
    return result

  # now, try integer optimization
  #print "have real solution", np.get_values([np[k] for k in xrange(n)])
  #print "copied constraints"
  cdef list sol

  # STEP 3: obtain weight vector, which must have integer solutions

  upper_bound = round(sum(np.get_values([np[k] for k in xrange(n)]))) + 1

  if not solve_integer(np, n): return False


  # make sure older LTs have not changed
  sol = np.get_values([np[k] for k in xrange(n)])
  lms_changed = 1

  while lms_changed != 0:

    lms_changed, changed_lms = monitor_lts(G, LTs, sol)
    resolve = False

    for j in xrange(len(changed_lms)):

      if len(changed_lms[j]) != 0: # some monomial changed :-(

        #print "adding constraints for", j
        resolve = True
        u = changed_lms[j][0]
        #print u
        v = LTs[j].exponents(as_ETuples=False)[0]
        constraint = tuple([float(v[k] - u[k]) for k in xrange(n)])
        new_constraints.add(constraint)
        number_of_constraints = np.number_of_constraints()
        np.add_constraint(np.sum([constraint[k]*np[k] for k in xrange(n)]), \
                          min=tolerance_cone)
        if np.number_of_constraints() > number_of_constraints:

          end_constraints += 1

    # if a monomial changed, we have to solve anew
    #print "resolve?", resolve
    if resolve:

      #print "resolving"
      if solve_real(np, n) and solve_integer(np, n):

        #print "found solutions"
        sol = np.get_values([np[k] for k in xrange(n)])

      else: # a-ha! this polynomial was actually incompatible!

        #print "failed to solve"
        np.remove_constraints(range(start_constraints, end_constraints))
        rejects.add(frozenset(new_constraints))
        return result

  #print "got it", sol
  # return to LP relaxation for future work
  for k in xrange(n):

    np.set_real(np[k])
    np.set_max(np[k],64000)


  # set up new solution
  result = (t,sol,np)
  #print np.number_of_constraints(), "constraints"
  #print "returning solution"
  return result

cpdef list possible_lts_all(MPolynomial_libsingular f):
  cdef list U = f.monomials()
  cdef list M = []
  cdef MPolynomial_libsingular ux
  cdef tuple u

  for ux in U:

    if ux != 1:
      u = ux.exponents(as_ETuples=False)[0]
      M.append((u, ux))

  return M

@cython.profile(True)
cpdef list possible_lts(MPolynomial_libsingular f, set boundary_vectors, \
                        int use_boundary_vectors):
  r"""
    Identifies terms of ``f`` that could serve as a leading term,
    and are compatible with the boundary approximated by ``boundary_vectors``.

    INPUT:

    - ``f`` -- a polynomial
    - ``boundary_vectors`` -- a tuple of maximum and minimum values of the
      variables on the current feasible region, allowing us to approximate the
      corners of the solution cone
    - `use_boundary_vectors` -- whether to use the boundary vectors; setting
      this to `False` gives behavior similar to Caboara's original
      implementation

    OUTPUT:

    A list of tuples, representing the exponent vectors of the potential
    leading monomials.

    ALGORITHM:

      #. Let ``t`` be current leading monomial of ``f``.
      #. Let ``U`` be set of exponent vectors of monomials of ``f``, except
        ``t``.
      #. Let ``M`` be subset of ``U`` s.t. `u\in M` iff `c.u > c.t`
        for some `c\in boundary_vectors`. (here, c.u denote multiplication)
      #. Return ``M``.

  """
  #print "in possible_lts"
  cdef list M
  cdef MPolynomial_libsingular ux, vx # terms
  cdef int i, j # counters
  cdef int passes # signals
  cdef tuple ordering, u, v # ordering and exponent vectors

  # identify current leading monomial, other monomials;
  #obtain their exponent vectors

  cdef MPolynomialRing_libsingular R = f.parent()
  cdef int n = len(R.gens())
  cdef tuple t = f.lm().exponents(as_ETuples=False)[0]
  cdef list U = f.monomials()
  U.remove(f.lm())

  # determine which elements of U might outweigh t somewhere within the current
  #solution cone
  # if there is no solution cone yet (first polynomial), pass them all

  # no point in checking anything if we have no boundaries yet...
  # todo: initialize boundary_vectors to obvious boundaries (e_i for i=1,...,n)
  #on first polynomial
  # so that we can remove this check
  if use_boundary_vectors and boundary_vectors != None:

    M = list()

    for j in xrange(len(U)):

      ux = U[j]; u = ux.exponents(as_ETuples=False)[0]; passes = False

      # check whether u weighs more than t according to some boundary vector
      for ordering in boundary_vectors:

        # compare dot products
        if sum([u[k]*ordering[k] for k in xrange(n)]) \
           > sum([t[k]*ordering[k] for k in xrange(n)]):

          M.append((u,ux))
          passes = True
          break

      if not passes: statistics.inc_monomials_eliminated()
      #monomials_eliminated += 1

  else: M = [(ux.exponents(as_ETuples=False)[0], ux) for ux in U] # if no boundary_vectors, allow all monomials

  # don't forget to append t
  M.append((t,f.lm()))

  # remove monomials u divisible by some other monomial v
  #-- this will happen rarely when using boundary vectors,
  #but could happen even then
  cdef list V = list()

  for (u,ux) in M:

    passes = True; i = 0

    while passes and i < len(M):

      v, vx = M[i]
      if vx != ux and monomial_divides(ux, vx): passes = False
      i += 1

    if passes: V.append((u,ux))

  #print V
  #print len(M), "possible leading monomials"
  return V

restricted_iterations = 1
@cython.profile(True)
cpdef tuple choose_ordering_restricted \
    (list G, list current_Ts, int mold, list current_ordering, \
     MixedIntegerLinearProgram lp, set rejects, set bvs, int use_bvs, \
     int use_dcs, bool print_candidates, str heuristic, \
     bool all_possible_lts=False):
  r"""
    Chooses a weight vector for a term ordering for the basis ``G`` that refines
    the weight vector that solves ``lp``.

    INPUTS:

    - ``G`` --  a basis of a polynomial ideal
    - ``current_Ts`` -- leading monomials of *some* elements of ``G``,
      according to the current ordering
    - ``mold`` -- the number of elements of ``current_Ts``
    - ``lp`` -- a linear program whose solutions dictate the choice
      ``current_Ts``
    - ``rejects`` -- record of linear programs rejected previously
    - ``bvs`` -- approximation to feasible region, in the form of
      maximum and minimum values of a cross-section of the feasible region
    - `use_bvs` -- whether to use boundary vectors; setting this and `use_dcs`
      to False gives us behavior similar to Caboara's original implementation
    - `use_dcs` -- whether to use disjoint cones; setting this and `use_bvs` to
      False gives us behavior similar to Caboara's original implementation
    - `print_candidates` -- whether to print number of LT candidates this
      iteration

    OUTPUTS:

    - a weighted ordering that solves...
    - a linear program that extends ``lp``, approximated by...
    - a list of vectors

    ALGORITHM:

    #. Of the possible leading monomials of each new `g\in G` that are
      compatible with ``lp``,
    #. determine which combinations are consistent with each other, then
    #. identify one which we think is a good choice.
  """
  global restricted_iterations
  #print "in optimal ordering"
  cdef int i, j, k # counters

  cdef MPolynomial_libsingular t, g # term, polynomial
  cdef MPolynomialRing_libsingular R = G[0].value().parent() # current ring

  # measures
  cdef int m = len(G)
  cdef int n = len(R.gens())

  #signals
  cdef int found, passes

  # contain leading terms, exponent vectors, weight ordering
  cdef list CLTs, LTs, LTups, w
  # elements of previous
  cdef tuple can_work, tup, changed_lms

  #print mold, m
  w = current_ordering

  for i in xrange(mold,m): # treat each new polynomial

    #print "finding lts for", i
    g = G[i].value()
    #print len(g.monomials()), "monomials to start with"
    if all_possible_lts:
      CLTs = possible_lts_all(g)
    else:
      CLTs = possible_lts(g, bvs, use_bvs)
    if print_candidates:
      print(restricted_iterations, len(CLTs))
    restricted_iterations += 1
    #print len(CLTs), "compatible leading monomials"
    #print CLTs

    #print("before sort")
    # use Caboara's Hilbert heuristic
    #CLTs = sort_CLTs_by_Hilbert_heuristic(R, current_Ts, CLTs)
    CLTs = sort_CLTs_by_heuristic_restricted(R, current_Ts, CLTs, heuristic)
    #print CLTs
    # discard unnecessary information
    #-- need to see why I have this information in the first place
    CLTs = [tup[len(tup)-1] for tup in CLTs]
    # extract the leading tuples
    LTups = [tup[0] for tup in CLTs]

    # find a compatible leading monomial that actually works
    # use fact that preferred monomials appear earlier in list
    found = False
    j = 0

    while not found and (CLTs[j][1] != g.lm() or all_possible_lts):

      #print "testing", CLTs[j], "against", g.lm()
      #print j, LTups, CLTs
      can_work = feasible(j, LTups, lp, rejects, G, current_Ts, use_dcs)
      #print can_work

      if len(can_work) > 0: # this means we found a solution

        w = can_work[1]
        #print CLTs[j][1], "worked! with", can_work
        found = True

      else: # monomial turned out to be incompatible after all

        #print CLTs[j][1], "did not work!"
        j += 1
    # now that we've found one, use it
    #print "success with monomial", j, CLTs[j], LTups[j]
    # no point in finding boundary vectors if g has the same leading term as
    #before
    if CLTs[j][1] == g.lm() and not all_possible_lts: return current_ordering, lp, bvs
    t = <MPolynomial_libsingular>CLTs[j][1]
    current_Ts.append(t) # hmm
    G[i].set_value(G[i].value() * G[i].value().coefficient(t)**(-1)) # make polynomial monic
    lp = can_work[2] # get new linear program
    #lp.show()
    bvs = boundary_vectors(lp, n) # compute new boundary vectors
    w = can_work[1] # get new weight vector

  #print "We have", len(rejects), "rejects"
  return w, lp, bvs
