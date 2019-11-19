import time
from functools import cmp_to_key

cpdef int hs_heuristic(tuple f, tuple g):
  r"""
    Implements the Hilbert heuristic recommended by Caboara in his 1993 paper.
    It first compares the degrees of the Hilbert polynomials;
    if one polynomial's degree is smaller than the other,
    then it returns the difference b/w the degree for f's hp and g's.
    Otherwise, it compares the Hilbert series of the polynomials;
    if the two are equal, it returns 0; otherwise,
    it returns the trailing coefficient of the numerator of the difference b/w
    f's hs and g's. See Caboara's paper for details
  """

  if f[0].degree() == g[0].degree():

    if f[1] == g[1]: return 0
    else:
      C = (f[1]-g[1]).numerator().coefficients()
      return C[len(C)-1]

  else: return f[0].degree() - g[0].degree()

@cython.profile(True)
cpdef list sort_CLTs_by_Hilbert_heuristic(MPolynomialRing_libsingular R, \
                                          list current_Ts, list CLTs):
  r"""
    Sorts the compatible leading monomials using the Hilbert heuristic
    recommended by Caboara in his 1993 paper.
    Preferable leading monomials appear earlier in the list.

    INPUTS:
    - `R` -- current ring (updated according to latest ordering!)
    - `current_Ts` -- current list of leading terms
    - `CLTs` -- compatible leading terms from which we want to select leading
      term of new polynomial
  """
  cdef tuple tup

  # create a list of tuples
  # for each leading term tup, we consider the monomial ideal created if tup
  # is chosen as the leading term.
  # the first entry is the tentative Hilbert polynomial
  # the second is the tentative Hilbert series
  # the third is tup itself (the compatible leading term)
  CLTs = [(R.ideal(current_Ts + [tup[1]]).hilbert_polynomial(algorithm='singular'),
           R.ideal(current_Ts + [tup[1]]).hilbert_series(),
           tup) for tup in CLTs]
  CLTs.sort(key=cmp_to_key(hs_heuristic)) # sort according to hilbert heuristic
  #print CLTs

  return CLTs

cpdef list min_CLT_by_Hilbert_heuristic(MPolynomialRing_libsingular R, \
                                        list CLTs):
  r"""
    Sorts the compatible leading monomials using the Hilbert heuristic
    recommended by Caboara in his 1993 paper.
    Preferable leading monomials appear earlier in the list.

    INPUTS:
    - `R` -- current ring (updated according to latest ordering!)
    - `CLTs` -- compatible leading terms from which we want to select
      leading term of new polynomial

    OUTPUTS:
    - the preferred list of leading monomials from CLTs
  """

  cdef list leads #lists of leading monomials in CLTs

  CLTs = [(R.ideal(leads).hilbert_polynomial(algorithm='singular'),
           R.ideal(leads).hilbert_series(),
           leads) for leads in CLTs]
  CLTs.sort(key=cmp_to_key(hs_heuristic))
  #TODO this could be optimized, we are sorting just to find the minimum...
  return CLTs[0][2]

cpdef list min_weights_by_Hilbert_heuristic(MPolynomialRing_libsingular R, \
                                            list CLTs):
  r"""
    Sorts the compatible leading monomials using the Hilbert heuristic
    recommended by Caboara in his 1993 paper.
    Preferable leading monomials appear earlier in the list.

    INPUTS:
    - `R` -- current ring (updated according to latest ordering!)
    - `current_Ts` -- current list of leading terms
    - `CLTs` -- compatible leading terms from which we want to select leading
      term of new polynomial

    OUTPUTS:
    - the preferred weight vector from CLTs
  """

  cdef tuple leads #lists of leading monomials in CLTs

  CLTs = [(R.ideal(leads[1]).hilbert_polynomial(algorithm='singular'),
           R.ideal(leads[1]).hilbert_series(),
           leads[0]) for leads in CLTs]
  CLTs.sort(key=cmp_to_key(hs_heuristic))

  #TODO this could be optimized, we are sorting just to find the minimum...
  return CLTs[0][2]

##Implementation of Betti-based heuristics

cpdef bool is_edge(int i, int j, list LMs):
  r"""
  Returns true iff (i, j) is an edge in the Buchberger graph of LMs
  """
  cdef int k, m
  cdef MPolynomialRing_libsingular R = LMs[0].parent()
  for k in xrange(len(LMs)):
    #if k == i or k == j:
    #  continue
    for v in R.gens():
      m = max([LMs[i].degree(v), LMs[j].degree(v)])
      if LMs[k].degree(v) >= m:
        if m > 0 or LMs[k].degree(v) > m:
          break #LMs[k] does not strictly divide in every variable lcm(LMs[i], LMs[j])
    else: #LMs[k] strictly divides lcm(LMs[i], LMs[j]) in every variable
      return False
  return True

cpdef int graph_edges(list LMs):
  cdef int num_edges = 0
  cdef int j, i
  #cdef MPolynomialRing_libsingular R = LMs[0].parent()
  #cdef list reduced = list(R.ideal(LMs).interreduced_basis())
  for j in xrange(len(LMs)):
    for i in xrange(j):
      if is_edge(i, j, LMs):
        num_edges += 1
  return num_edges

cpdef int betti_number(list LMs):
  #cdef MPolynomialRing_libsingular R = LMs[0].parent()
  #cdef list reduced = list(R.ideal(LMs).interreduced_basis())
  #I = singular.ideal(reduced)
  I = singular.ideal(LMs)
  #Compute the number of generators of Syz(I) in a minimal free resolution
  cdef int betti = I.mres(0)[2].sage().nrows()
  return betti

cpdef int betti_heuristic(tuple f, tuple g):
  return f[0] - g[0]

cpdef int hilbert_betti_heuristic(tuple f, tuple g):

  if f[0].degree() == g[0].degree():
    # Break Hilbert degree ties by Betti number approximation
    return f[1] - g[1]

  return f[0].degree() - g[0].degree()

cpdef float avgdeg(list LMs):

  '''
  Computes the average degree of the minimum generators of (the ideal generated
  by) LMs.
  '''

  cdef MPolynomialRing_libsingular R = LMs[0].parent()
  cdef list reduced = list(R.ideal(LMs).interreduced_basis())
  cdef int s = 0

  for monomial in reduced:
      #TODO also, try for non std grading...
      s += monomial.degree(std_grading=True)

  return float(s) / len(reduced)

cpdef float avgdeg_heuristic(tuple f):

  return f[0] #The avg degree is the first coordinate of the tuple

cpdef list sort_CLTs_by_heuristic_restricted \
    (MPolynomialRing_libsingular R, list current_Ts, list CLTs, str heuristic):

  init_time = time.time()
  if heuristic == 'hilbert':
    return_val = sort_CLTs_by_Hilbert_heuristic(R, current_Ts, CLTs)
    statistics.inc_heuristic_overhead(time.time() - init_time)
    return return_val

  elif heuristic == 'betti':

    L = [(graph_edges(current_Ts + [tup[1]]),
          (),
          tup) for tup in CLTs]
    L.sort(key=cmp_to_key(betti_heuristic))
    statistics.inc_heuristic_overhead(time.time() - init_time)
    return L

  elif heuristic == 'mixed':

    L = [(R.ideal(current_Ts + [tup[1]]).hilbert_polynomial(algorithm='singular'),
          graph_edges(current_Ts + [tup[1]]),
          tup) for tup in CLTs ]
    L.sort(key=cmp_to_key(hilbert_betti_heuristic))
    statistics.inc_heuristic_overhead(time.time() - init_time)
    return L

  elif heuristic == 'avgdeg':

    L = [ (avgdeg(current_Ts + [tup[1]]), (), tup) for tup in CLTs]
    L.sort(key=avgdeg_heuristic)
    statistics.inc_heuristic_overhead(time.time() - init_time)
    return L

  raise ValueError("Invalid heuristic function: " + heuristic)

cpdef list sort_CLTs_by_heuristic(list CLTs, str heuristic, bool use_weights, \
                                  int prev_betti=-1, int prev_hilb=-1):

  # if use_weights, CLTs is a list of tuples (candidate lts, weight vector)
  # otherwise, it is a list of candidate lts
  cdef list L, old_order
  cdef MPolynomialRing_libsingular R = CLTs[0][0][0].parent() if use_weights else CLTs[0][0].parent()
  init_time = time.time()
  if heuristic == 'hilbert':

    old_order = [ ((), (), CLTs[0]) ]
    if use_weights:
      L = [ (R.ideal(LTs[0]).hilbert_polynomial(algorithm='singular'),
             R.ideal(LTs[0]).hilbert_series(),
             LTs) for LTs in CLTs ]
    else:
      L = [ (R.ideal(LTs).hilbert_polynomial(algorithm='singular'),
             R.ideal(LTs).hilbert_series(),
             LTs) for LTs in CLTs ]
    L.sort(key=cmp_to_key(hs_heuristic))
    #if prev_hilb <= L[0][0].degree():
    #  statistics.inc_heuristic_overhead(time.time() - init_time)
    #  return old_order
    statistics.inc_heuristic_overhead(time.time() - init_time)
    return L

  elif heuristic == 'betti':

    old_order = [ ((), (), CLTs[0]) ]
    if use_weights:
      L = [ (graph_edges(LTs[0]), (), LTs) for LTs in CLTs ]
      #L = [ (betti_number(LTs[0]), (), LTs) for LTs in CLTs ]
    else:
      L = [ (graph_edges(LTs), (), LTs) for LTs in CLTs ]
      #L = [ (betti_number(LTs), (), LTs) for LTs in CLTs ]
    L.sort(key=cmp_to_key(betti_heuristic))
    if prev_betti >= 0 and prev_betti <= L[0][0]:
      statistics.inc_heuristic_overhead(time.time() - init_time)
      return old_order
    statistics.inc_heuristic_overhead(time.time() - init_time)
    return L

  elif heuristic == 'mixed':

    old_order = [ ((), (), CLTs[0]) ]
    if use_weights:
      L = [ (R.ideal(LTs[0]).hilbert_polynomial(algorithm='singular'),
             graph_edges(LTs[0]),
             #betti_number(LTs[0]),
             LTs) for LTs in CLTs ]
    else:
      L = [ (R.ideal(LTs).hilbert_polynomial(algorithm='singular'),
             graph_edges(LTs),
             #betti_number(LTs),
             LTs) for LTs in CLTs ]
    L.sort(key=cmp_to_key(hilbert_betti_heuristic))
    if prev_hilb >= L[0][0] and prev_betti >= 0 and prev_betti < L[0][1]:
      statistics.inc_heuristic_overhead(time.time() - init_time)
      return old_order
    statistics.inc_heuristic_overhead(time.time() - init_time)
    return L

  elif heuristic == 'avgdeg':

    if use_weights:
      L = [ (avgdeg(LTs[0]), (), LTs) for LTs in CLTs ]
    else:
      L = [ (avgdeg(LTs), (), LTs) for LTs in CLTs ]
    L.sort(key=avgdeg_heuristic)
    statistics.inc_heuristic_overhead(time.time() - init_time)
    return L

  raise ValueError("Invalid heuristic function: " + heuristic)

cdef tuple hilbert_key (tuple t):
  '''
  Comparison by the Hilbert heuristic is done using this key:
  (Hilbert degree, HS coefficients)
  '''
  I = t[0]
  return (I.hilbert_polynomial(algorithm='singular').degree(),
          I.hilbert_series().numerator().coefficients())

cdef list best_orderings_hilbert (list orderings, list G):

  cdef list w, LMs, candidates = []
  cdef clothed_polynomial g
  cdef MPolynomialRing_libsingular R = G[0].value().parent()
  for w in orderings:
    R = PolynomialRing(R.base_ring(), R.gens(), order=create_order(w))
    LMs = [ R(g.value()).lm() for g in G ]
    I = R.ideal(LMs)
    candidates.append((I, w))

  candidates.sort(key = hilbert_key)

  cdef tuple t
  return [ t[1] for t in candidates]

cdef list hilbert_smaller (list w1, list w2, list G):
  cdef MPolynomialRing_libsingular R = G[0].value().parent()
  cdef clothed_polynomial g

  cdef MPolynomialRing_libsingular R1, R2
  R1 = PolynomialRing(R.base_ring(), R.gens(), order=create_order(w1))
  R2 = PolynomialRing(R.base_ring(), R.gens(), order=create_order(w2))

  cdef list LMs1, LMs2
  LMs1 = [ R1(g.value()).lm() for g in G ]
  LMs2 = [ R2(g.value()).lm() for g in G ]

  I1 = R1.ideal(LMs1)
  I2 = R2.ideal(LMs2)

  candidates = [ (I1, w1), (I2, w2) ]
  candidates.sort(key = hilbert_key)

  return candidates[0][1]
