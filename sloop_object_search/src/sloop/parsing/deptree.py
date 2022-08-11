from . import spacy_utils as sputils
import spacy

class DependencyTreeNode:
    def __init__(self, node_type, content, children, parent=None):
        """
        A DependencyTreeNode stores a piece of string (`content`)
        and has a set of children where each is connected through
        a dependency type.

        Args:
            node_type (str): The label of the edge that connects to
                this node. E.g. nsubj, root, etc.
            content (str): The string (e.g. a word or phrase) that
                this DependencyTreeNode represents.
            parent (DependencyTreeNode): The parent node; This
                makes the tree doubly linked. Useful for treating
                the tree as a graph.
            children (dict): The children of this node, mapping
                a dependency type (a string) to another node. (Assuming
                in a DependencyTree all edges have unique labels.
        """
        self.node_type = node_type
        self.content = content
        self.children = children
        self.parent = parent

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "DTNode(\"%s\"; type: %s; children: %s)"\
            % (self.content, self.node_type.upper(),
               str(list(map(str.upper, self.children.keys()))))

    def __hash__(self):
        return hash((self.node_type, self.content))


class DependencyTree:
    """Models a dependency tree as a result of parsing the
    natural spatial language. This is a standard dependency
    tree. No special processing related to the spatial language
    or object search problem.

    For more on dependency trees:
    https://www.cs.virginia.edu/~kc2wc/teaching/NLP16/slides/15-DP.pdf"""

    def __init__(self, root):
        self.root = root
        self._info = self._process_node(self.root)

    def lookup(self, cue_type, cue, err_if_unfound=True):
        """
        Returns a single node or a set of nodes
        by looking up the given cue in the dependency tree.
        A `cue_type` (str) could be either "content" or "type",
        corresponding to a phrase/word or a dependency type.
        """
        if cue_type.lower() != "content" and cue_type.lower() != "type":
            raise ValueError("Wrong cue \"%s\" for lookup." % cue_type)

        nodes = set({})
        if cue_type.lower() == "content":
            if cue in self._info["wd"]:
                nodes = self._info["wd"][cue]
        else: #cue.lower() == "type":
            if cue in self._info["td"]:
                nodes = self._info["td"][cue]
        if err_if_unfound and len(nodes) == 0:
            raise ValueError("Failed to lookup \"%s\" by %s" % (cue, cue_type))
        return nodes

    def _process_node(self, node):
        """Goes through the subtree rooted at `node` and
        return a dictionary of information."""
        info = {"wd": {},  # wd means 'word dictionary': content->node
                "td": {},  # td means 'type dictionary': dep_type->node
                "num_nodes": 0}  # the number of nodes in this dep.tree
        self._process_node_helper(node, info)
        return info

    def _process_node_helper(self, node, info):
        """Goes through the subtree rooted at `node` and
        return a dictionary of information."""
        info["num_nodes"] += 1
        if node.content not in info["wd"]:  # wd means 'word dictionary'
            info["wd"][node.content] = set({})
        info["wd"][node.content].add(node)
        if node.node_type not in info["td"]: # td means 'type dictionary'
            info["td"][node.node_type] = set({})
        info["td"][node.node_type].add(node)
        for ch_type in node.children:
            self._process_node_helper(node.children[ch_type], info)

    def path_between(self, node1, node2, directed=False):
        """Returns a list that contains the path between
        node1 and node2. If directed, then the path goes
        from node1 to node2. Treat the dependency tree as a graph,
        that is, there is no direction.

        Note that the returned path includes node1 as the first node,
        and node2 as the last node."""
        def _get_path(n1, n2, parent_map):
            v = n2
            path = [n2]
            while v != n1:
                v = parent_map[v]
                path.append(v)
            return list(reversed(path))
        # DFS
        stack = [node1]
        visited = set({node1})
        parent_map = {node1:None}
        while len(stack) > 0:
            n = stack.pop()
            if n == node2:
                return _get_path(node1, node2, parent_map)
            neighbors = {n.children[ch_type]
                         for ch_type in n.children}
            if not directed and n.parent is not None:
                neighbors |= set({n.parent})
            for v in neighbors:
                if v not in visited:
                    stack.append(v)
                    visited.add(v)
                    parent_map[v] = n
        return None  # no path found

    def pprint(self, max_depth=None):
        """Print the dependency tree in a readable manner"""
        if hasattr(self, "_nltktree"):
            # Just print the nltk tree
            self._nltktree.pretty_print()
        else:
            self._pprint_helper(self.root, "", 0, max_depth=max_depth)

    def _pprint_helper(self, parent, parent_edge, depth, max_depth=None):
        """
        Helper for printing the dependency tree.

        Args:
            parent (DependencyTreeNode)
            parent_edge (str): The label of the dependency
            depth (int): the depth of parent
            max_depth (int): maximum depth to print
        """
        if max_depth is not None and depth >= max_depth:
            return
        if len(parent_edge) > 0:
            print("%sL" % ("    "*depth))
            print("%s |%s" % ("    "*depth, str(parent_edge.upper())))
            print("%s |" % ("    "*depth))
        print("%s %s" % ("    "*depth, str(parent)))
        for dep_label in parent.children:
            self._pprint_helper(parent.children[dep_label],
                                dep_label,
                                depth+1,
                                max_depth=max_depth)

    @classmethod
    def build(cls, sentence, model, method=None):
        """
        Build the dependency tree of the given sentence.
        `method` can be either "allennlp" or "spacy".
        If former, then the model is a predictor (see their
        dependency parsing example). If latter, the model
        is a spacy loaded model (e.g. en_core_web_md).
        """
        if method == "allennlp":
            # The following triggers warning: Your label namespace was 'pos'. We
            # recommend you use a namespace ending with 'labels' or 'tags', so we
            # don't add UNK and PAD tokens by default to your vocabulary.  See
            # documentation for `non_padded_namespaces` parameter in Vocabulary.
            res = model.predict(sentence=sentence)
            root = cls._build_helper_allennlp(res["hierplane_tree"]["root"], None)
            return DependencyTree(root)
        elif method == "spacy":
            if type(sentence) != spacy.tokens.Doc:
                sents = list(model(sentence).sents)
            else:
                sents = list(sentence.sents)
            if len(sents) > 1:
                print("Warning: This sentence contains multiple sentences (recognized by spacy)."
                      "Only the first one will be processed.")
            root = cls._build_helper_spacy(sents[0].root, None)
            deptree = DependencyTree(root)
            deptree._nltktree = sputils.to_nltk_tree(sents[0].root)
            return deptree
        else:
            raise NotImplementedError("Build method %s is not implemented")

    @classmethod
    def _build_helper_spacy(cls, token, parent):
        node = DependencyTreeNode(token.dep_,
                                  token.text, {}, parent=parent)
        for child_token in token.children:
            ch_node = cls._build_helper_spacy(child_token, node)
            ch_key = child_token.dep_
            count = len([ck for ck in node.children
                         if ck.split("-")[0] == ch_key])
            if ch_key in node.children:
                ch_key += "-%d" % count
            node.children[ch_key] = ch_node
        return node

    @classmethod
    def _build_helper_allennlp(cls, node_dict, parent):
        """
        Args:
            node_dict (dict): A dictionary that is part of the
                parsed result of the allennlp predictor, that contains
                information about one word.
        """
        node = DependencyTreeNode(node_dict["nodeType"],
                                  node_dict["word"], {}, parent=parent)
        if "children" in node_dict:
            for ch_dict in node_dict["children"]:
                ch_node = cls._build_helper_allennlp(ch_dict, node)
                assert node_dict["link"] == node_dict["nodeType"],\
                    "Unexpected difference between 'link' and 'nodeType' of\n%s" % str(node_dict)
                ch_key = ch_dict["link"]
                count = len([ck for ck in node.children
                             if ck.split("-")[0] == ch_key])
                if ch_key in node.children:
                    ch_key += "-%d" % count
                node.children[ch_key] = ch_node
        return node
