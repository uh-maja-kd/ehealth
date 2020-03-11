from pydot import Dot, Node, Edge

class Tree(object):
    def __init__(self, idx: int, info: str):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.idx = idx
        self.state = None
        self.info = info

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size', None):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth', None):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    @property
    def nodes(self):
        if getattr(self, '_nodes', None):
            return self._nodes

        self._nodes = [self]
        for child in self.children:
            self._nodes.extend(child.nodes)

        return self._nodes

    @property
    def edges(self):
        if getattr(self, '_edges', None):
            return self._edges

        self._edges = [(self, child) for child in self.children]
        for child in self.children:
            self._edges.extend(child.edges)

        return self._edges

    def pydot_img(self, path: str):
        graph = Dot(graph_type='digraph')
        graph.set_node_defaults(
            color='lightgray',
            style='filled',
            shape='box',
            fontname='Courier',
            fontsize='10'
        )

        nodes = {node.idx:Node(f"{node.info}") for node in self.nodes}
        for u,v in self.edges:
            graph.add_edge(Edge(nodes[u.idx], nodes[v.idx]))

        p = r"C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\pkgs\graphviz-2.38-hfd603c8_2\Library\bin\graphviz\dot.exe"

        graph.write_png(path, prog = p)
