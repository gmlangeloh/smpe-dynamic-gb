'''
    Sage script to read MathicGB format ideals.
'''

from string import ascii_lowercase, ascii_uppercase
from time import time

def _makeRing(char, n, s, M):
    baseRing = QQ if char == 0 else GF(char)
    varList = list(ascii_lowercase[0:n]) + list(ascii_uppercase[0:max(n-26, 0)])
    if s == 1: #weighted grevlex order
        o = TermOrder('wdegrevlex', M)
    else:
        o = TermOrder(matrix(M))
    R = PolynomialRing(baseRing, order=o, names=varList)
    return R

def _readWeights(s, w, f):
    if s == 1:
        return list(map(int, w))
    total = len(w)
    M = []
    M.append(list(map(int, w)))
    for i in range(1, s):
        w = f.readline().split()
        M.append(list(map(int, w)))
    return M

def _readHeader(f):
    header = f.readline()
    elems = header.split()
    characteristic = int(elems[0])
    n = int(elems[1])
    s = int(elems[2])
    M = _readWeights(s, elems[3:], f)
    R = _makeRing(characteristic, n, s, M)
    m = int(f.readline().split()[0])
    return R, m

def _readPolynomial(R, f):
    pstring = f.readline()
    nstring = ""
    for c in pstring:
        if len(nstring) == 0:
            nstring += c
        elif nstring[-1].isdigit():
            if c.isalpha():
                nstring += "*"
            nstring += c
        elif nstring[-1].isalpha():
            if c.isalpha():
                nstring += "*"
            elif c.isdigit():
                nstring += "^"
            nstring += c
        else:
            nstring += c
    return R(nstring)

def _readIdeal(filename):
    L = []
    with open(filename, "r") as f:
        R, m = _readHeader(f)
        for i in range(m):
            p = _readPolynomial(R, f)
            L.append(p)
    return ideal(L)

def _change_ring_order(w, L):
    '''
    Changes the polynomials of L to a ring order given by w, grevlex.
    '''

    def _makegrevlex(w):
        '''
        Makes a w,grevlex order represented by a matrix M.
        '''
        grevM = [w]
        n = len(w)
        for i in range(n-1):
            row = [ 1 ] * (n-i) + [0] * i
            grevM.append(row)
        return matrix(grevM)

    oldR = L[0].parent()
    M = matrix(w)
    if M.nrows() == 1 and all([ wi > 0 for wi in w ]):
        newR = oldR.change_ring(order=TermOrder("wdegrevlex", w))
        return map(lambda p: newR(p), L)
    elif M.nrows() == 1:
        M = _makegrevlex(M[0])
    newR = oldR.change_ring(order=TermOrder(M))
    return map(lambda p: newR(p), L)

def _term_data(G):

    def average(L):
        return sum(L) / len(L)

    L = map(lambda p: len(p.monomials()), G)
    return min(L), average(L), max(L), sum(L)

def _degree_data(G):

    def average(L):
        return sum(L) / len(L)

    L = map(lambda p: p.total_degree(), G)
    return min(L), average(L), max(L)

def _metadata(G):
    return _term_data(G) + _degree_data(G)

def _writeRing(R, f):
    characteristic = R.base_ring().characteristic()
    n = R.ngens()
    T = R.term_order()
    if T.is_weighted_degree_order():
        if T.matrix():
            M = T.matrix().rows()
            s = len(M)
        else:
            M = [ T.weights() ]
            s = 1
    else: #Using grevlex order, weight vector 1
        M = [ [ 1 ] * n ]
        s = 1
    f.write(str(characteristic) + " ")
    f.write(str(n) + " ")
    f.write(str(s) + " ")
    for row in M:
        for elem in row:
            f.write(str(elem) + " ")
    f.write("\n")

def _writePolynomial(g, f):
    pol_string = filter(lambda c: c not in "*^ ", str(g))
    f.write(pol_string + "\n")

def _writeIdeal(I, filename):
    R = I.ring()
    with open(filename, "w") as f:
        _writeRing(R, f)
        m = len(I.gens())
        f.write(str(m) + "\n")
        for g in I.gens():
            _writePolynomial(g, f)

class Benchmark:

    '''
    An Ideal of a polynomial ring that has a representation as a MathicGB
    .ideal file.
    '''

    def __init__(self, I):
        '''
        I may be either an Ideal or the path to a .ideal file.
        '''
        if isinstance(I, str):
            self.ideal = _readIdeal(I)
        else:
            self.ideal = I

    def write(self, filename):
        '''
        Writes this benchmark to <filename> in the MathicGB format.
        '''
        _writeIdeal(self.ideal, filename)

    def change_order(self, w):
        '''
        Changes the order of this ideal to <_{w, grevlex}.
        '''
        self.ideal = ideal(_change_ring_order(w, self.ideal.gens()))

    def ideal_lt(self):
        '''
        Returns the ideal of leading terms of a list of polynomials.
        '''
        return ideal(map(lambda p: p.lt(), self.ideal.gens()))

    def hilbert_polynomial(self):
        '''
        Returns the Hilbert Polynomial of <LT(G)>, G the generating set of
        this ideal.
        '''
        return self.ideal_lt().hilbert_polynomial()

    def hilbert_series(self):
        '''
        Returns the Hilbert Series of <LT(G)>, G the generating set of this
        ideal.
        '''
        return self.ideal_lt().hilbert_series()

    def hilbert_function(self):
        '''
        Returns the first few values of the Hilbert Function of <LT(G)>, G the
        generating set of this ideal.
        '''
        t = PowerSeriesRing(QQ, 't')
        return t(self.hilbert_series()).coefficients()

    def gb_data(self):
        '''
        Returns the following data about GB computation of this ideal:
        - time;
        - basis size;
        - minimum, average, maximum and total terms;
        - minimum, average and maximum degrees;
        '''
        G = self.ideal.gens()
        init_time = time()
        G = ideal(G).groebner_basis()
        final_time = time()
        ellapsed_time = final_time - init_time
        basis_size = len(G)
        P = ideal(map(lambda p: p.lt(), G)).hilbert_polynomial()
        return (ellapsed_time, basis_size) + _metadata(G) + (P.degree(), float(P.lc()))

    def hilbert_metadata(self):
        P = self.hilbert_polynomial()
        return P.degree(), float(P.lc())

    def metadata(self):
        n = self.ideal.ring().ngens()
        m = len(self.ideal.gens())
        return (n, m) + _metadata(self.ideal.gens()) + self.hilbert_metadata()

    def data(self):
        n = self.ideal.ring().ngens()
        m = len(self.ideal.gens())
        k = self.ideal.ring().base_ring().characteristic()
        deg = max([ f.total_degree(True) for f in self.ideal.gens() ])
        mon = sum([ len(f.monomials()) for f in self.ideal.gens() ])
        hom = self.ideal.is_homogeneous()
        return n, m, k, mon, deg, hom
