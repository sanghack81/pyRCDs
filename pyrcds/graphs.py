import collections

import networkx as nx


class PDAG:
    """Partially-directed Acyclic Graph"""

    def __init__(self, edges=None):
        self.E = set()
        self._Pa = collections.defaultdict(set)
        self._Ch = collections.defaultdict(set)
        if edges is not None:
            self.add_edges(edges)
        self.aux_vert = set()

    def add_nodes(self, nbunch):
        for n in nbunch:
            self.aux_vert.add(n)

    def __bool__(self):
        return bool(self.vertices())

    def vertices(self):
        return set(self._Pa.keys()) | set(self._Ch.keys()) | self.aux_vert

    def __iter__(self):
        return iter(self.vertices())

    def __contains__(self, item):
        return item in self.E or item in self.vertices()

    def an(self, x, at=None):
        """Ancestors of x"""
        if at is None:
            at = set()

        for p in self.pa(x):
            if p not in at:
                at.add(p)
                self.an(p, at)

        return at

    def de(self, x, at=None):
        """Descendants of x"""
        if at is None:
            at = set()

        for p in self.ch(x):
            if p not in at:
                at.add(p)
                self.de(p, at)

        return at

    def oriented(self):
        """all oriented edges"""
        ors = set()
        for x, y in self.E:
            if (y, x) not in self.E:
                ors.add((x, y))
        return ors

    def unoriented(self):
        """all unoriented edges"""
        uors = set()
        for x, y in self.E:
            if (y, x) in self.E:
                uors.add(frozenset({x, y}))
        return uors

    def remove_vertex(self, v):
        for x, y in list(self.E):
            if x == v or y == v:
                self.E.remove((x, y))

        self._Pa.pop(v, None)
        self._Ch.pop(v, None)

        for k, values in self._Pa.items():
            if v in values:
                values.remove(v)
        for k, values in self._Ch.items():
            if v in values:
                values.remove(v)

        if v in self.aux_vert:
            self.aux_vert.remove(v)

    def copy(self):
        new_copy = PDAG()
        new_copy.E = set(self.E)
        new_copy._Pa = collections.defaultdict(set)
        new_copy._Ch = collections.defaultdict(set)
        for k, vs in self._Pa.items():
            new_copy._Pa[k] = set(vs)
        for k, vs in self._Ch.items():
            new_copy._Ch[k] = set(vs)
        new_copy.aux_vert = set(self.aux_vert)

        return new_copy

    def add_path(self, iterable):
        for x, y in zip(iterable, iterable[1:]):
            self.add_edge(x, y)

    def add_undirected_path(self, iterable):
        for x, y in zip(iterable, iterable[1:]):
            self.add_undirected_edge(x, y)

    def is_adj(self, x, y):
        return (x, y) in self.E or (y, x) in self.E

    def add_edges(self, xys):
        for x, y in xys:
            self.add_edge(x, y)

    def add_undirected_edges(self, xys):
        for x, y in xys:
            self.add_undirected_edge(x, y)

    def add_edge(self, x, y):
        """Add an edge

        Notes
        -----
        if y-->x exists, adding x-->y makes x -- y (i.e., undirected edge).
        """
        assert x != y
        self.E.add((x, y))
        self._Pa[y].add(x)
        self._Ch[x].add(y)

    def add_undirected_edge(self, x, y):
        """Add an edge

        Notes
        -----
        This will override any existing directed edge.
        """
        assert x != y
        self.add_edge(x, y)
        self.add_edge(y, x)

    def orients(self, xys):
        return any([self.orient(x, y) for x, y in xys])

    def orient(self, x, y):
        """Orient an undirected edge to directed edge x-->y

        Returns
        -------
        True if an undirected edge x--y is changed to x-->y
        """
        if (x, y) in self.E:  # already oriented as x -> y?
            if (y, x) in self.E:  # bi-directed?
                self.E.remove((y, x))
                self._Pa[x].remove(y)
                self._Ch[y].remove(x)
                return True
        return False

    def is_oriented_as(self, x, y):
        return (x, y) in self.E and (y, x) not in self.E

    def is_unoriented(self, x, y):
        return (x, y) in self.E and (y, x) in self.E

    def is_oriented(self, x, y):
        """Returns true if there exists a directed edge between x and y without orientation (i.e., x-->y or x<--y)."""
        return ((x, y) in self.E) ^ ((y, x) in self.E)

    def ne(self, x):
        """Neighbors -- connected through an undirected edge"""
        return self._Pa[x] & self._Ch[x]

    def adj(self, x):
        """Adjacencies -- connected through an (un)directed edge"""
        return self._Pa[x] | self._Ch[x]

    # get parents
    def pa(self, x):
        """Parents -- connected through a directed edge towards x"""
        return self._Pa[x] - self._Ch[x]

    # get children
    def ch(self, x):
        """Children -- connected through a directed edge from x"""
        return self._Ch[x] - self._Pa[x]

    def as_networkx_dag(self):
        assert len(self.unoriented()) == 0
        dg = nx.DiGraph()
        dg.add_nodes_from(self.aux_vert)
        dg.add_edges_from(self.oriented())
        return dg

    def as_networkx_ug(self):
        assert len(self.oriented()) == 0
        ug = nx.Graph()
        ug.add_nodes_from(self.aux_vert)
        ug.add_edges_from(self.unoriented())
        return ug
