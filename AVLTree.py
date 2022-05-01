"""
Project 3
CSE 331 F21 (Onsay)
Austin Copeland
AVLTree.py
"""

import queue
from typing import TypeVar, Generator, List, Tuple

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")  # represents generic type
Node = TypeVar("Node")  # represents a Node object (forward-declare to use in Node __init__)
AVLWrappedDictionary = TypeVar("AVLWrappedDictionary")  # represents a custom type used in application


####################################################################################################


class Node:
    """
    Implementation of an AVL tree node.
    Do not modify.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["value", "parent", "left", "right", "height"]

    def __init__(self, value: T, parent: Node = None,
                 left: Node = None, right: Node = None) -> None:
        """
        Construct an AVL tree node.

        :param value: value held by the node object
        :param parent: ref to parent node of which this node is a child
        :param left: ref to left child node of this node
        :param right: ref to right child node of this node
        """
        self.value = value
        self.parent, self.left, self.right = parent, left, right
        self.height = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"

    def __str__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"


####################################################################################################


class AVLTree:
    """
    Implementation of an AVL tree.
    Modify only below indicated line.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty AVL tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree as a string. Inspired by Anna De Biasi (Fall'20 Lead TA).

        :return: string representation of the AVL tree
        """
        if self.origin is None:
            return "Empty AVL Tree"

        # initialize helpers for tree traversal
        root = self.origin
        result = ""
        q = queue.SimpleQueue()
        levels = {}
        q.put((root, 0, root.parent))
        for i in range(self.origin.height + 1):
            levels[i] = []

        # traverse tree to get node representations
        while not q.empty():
            node, level, parent = q.get()
            if level > self.origin.height:
                break
            levels[level].append((node, level, parent))

            if node is None:
                q.put((None, level + 1, None))
                q.put((None, level + 1, None))
                continue

            if node.left:
                q.put((node.left, level + 1, node))
            else:
                q.put((None, level + 1, None))

            if node.right:
                q.put((node.right, level + 1, node))
            else:
                q.put((None, level + 1, None))

        # construct tree using traversal
        spaces = pow(2, self.origin.height) * 12
        result += "\n"
        result += f"AVL Tree: size = {self.size}, height = {self.origin.height}".center(spaces)
        result += "\n\n"
        for i in range(self.origin.height + 1):
            result += f"Level {i}: "
            for node, level, parent in levels[i]:
                level = pow(2, i)
                space = int(round(spaces / level))
                if node is None:
                    result += " " * space
                    continue
                result += f"{node}".center(space, " ")
            result += "\n"
        return result

    def __str__(self) -> str:
        """
        Represent the AVL tree as a string. Inspired by Anna De Biasi (Fall'20 Lead TA).

        :return: string representation of the AVL tree
        """
        return repr(self)

    def height(self, root: Node) -> int:
        """
        Return height of a subtree in the AVL tree, properly handling the case of root = None.
        Recall that the height of an empty subtree is -1.

        :param root: root node of subtree to be measured
        :return: height of subtree rooted at `root` parameter
        """
        return root.height if root is not None else -1

    def left_rotate(self, root: Node) -> Node:
        """
        We are giving you the implementation of left rotate, use this to write right rotate ;)
        Perform a left rotation on the subtree rooted at `root`. Return new subtree root.

        :param root: root node of unbalanced subtree to be rotated.
        :return: new root node of subtree following rotation.
        """

        if root is None:
            return None

        # pull right child up and shift right-left child across tree, update parent
        new_root, rl_child = root.right, root.right.left
        root.right = rl_child
        if rl_child is not None:
            rl_child.parent = root

        # right child has been pulled up to new root -> push old root down left, update parent
        new_root.left = root
        new_root.parent = root.parent
        if root.parent is not None:
            if root is root.parent.left:
                root.parent.left = new_root
            else:
                root.parent.right = new_root
        root.parent = new_root

        # handle tree origin case
        if root is self.origin:
            self.origin = new_root

        # update heights and return new root of subtree
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + max(self.height(new_root.left), self.height(new_root.right))
        return new_root

    ########################################
    # Implement functions below this line. #
    ########################################

    def right_rotate(self, root: Node) -> Node:
        """
        This function will be implemented during live class activity!
        Come to class to learn about it :) Oct 28th live class,
        in case you can't and you do  not want to
        you can easily look at rotate left function and use symmetry as your friend to complete
        rotate right this..
        You don't really need that much help to complete it.
        """
        if root is None:
            return None

        new_root, lr_child = root.left, root.left.right
        root.left = lr_child
        if lr_child is not None:
            lr_child.parent = root

        new_root.right = root
        new_root.parent = root.parent
        if root.parent is not None:
            if root is root.parent.left:
                root.parent.left = new_root
            else:
                root.parent.right = new_root
        root.parent = new_root

        if root is self.origin:
            self.origin = new_root

        root.height = 1 + max(self.height(root.right), self.height(root.left))
        new_root.height = 1 + max(self.height(new_root.right), self.height(new_root.left))
        return new_root

    def balance_factor(self, root: Node) -> int:
        """
        This function calculates the balance factor of
        an avl tree
        Input:
            root: the root node of an avl tree
        Returns:
            the balance factor of the tree
        """
        if root is None:
            return 0
        h_l = self.height(root.left)
        h_r = self.height(root.right)
        b_f = h_l - h_r
        return b_f

    def rebalance(self, root: Node) -> Node:
        """
         This function rebalances an avl tree based on its bf
        Input:
            root: the root node of an avl tree
        Returns:
            a generator of type node
        """
        if self.balance_factor(root) <= -2:
            if self.balance_factor(root.right) > 0:
                root.right = self.right_rotate(root.right)
                return self.left_rotate(root)

            else:
                return self.left_rotate(root)

        elif self.balance_factor(root) >= 2:
            if self.balance_factor(root.left) < 0:
                root.left = self.left_rotate(root.left)
                return self.right_rotate(root)

            else:
                return self.right_rotate(root)

        else:
            return root

    def insert(self, root: Node, val: T) -> Node:
        """
         This function inserts into the correct place
         and then is rebalanced based on its bf
        Input:
            root: the root node of an avl tree
        Returns:
            the new root of the balanced avl tree
        """
        if root is None:
            new_node = Node(val)
            if self.size == 0:
                self.origin = new_node
            self.size += 1
            return new_node


        if val < root.value:
            left_sub_root = self.insert(root.left, val)
            root.left = left_sub_root
            left_sub_root.parent = root


        elif val > root.value:
            right_sub_root = self.insert(root.right, val)
            root.right = right_sub_root
            right_sub_root.parent = root


        else:
            return root

        root.height = 1 + max(self.height(root.left), self.height(root.right))
        self.origin = self.rebalance(root)

        return self.origin

    def min(self, root: Node) -> Node:
        """
         This function finds the min val in an avl tree
        Input:
            root: the root node of an avl tree
        Returns:
            the min node
        """
        if root == self.origin and root is None:
            return None

        if root.left is None:
            min_node = root
        else:
            min_node = self.min(root.left)

        return min_node

    def max(self, root: Node) -> Node:
        """
         This function returns the max value in an avl tree
        Input:
            root: the root node of an avl tree
        Returns:
            the max node
        """
        if root == self.origin and root is None:
            return None

        if root.right is None:
            max_node = root
        else:
            max_node = self.max(root.right)
        return max_node

    def search(self, root: Node, val: T) -> Node:
        """
        This function searches for a given value in an
        avl tree
        Input:
            root: the root of an avl tree
        Returns: the node if found
        """
        if root is None:
            return
        if val < root.value:
            if root.left is None:
                return root
            return self.search(root.left, val)
        elif val > root.value:
            if root.right is None:
                return root
            return self.search(root.right, val)
        else:
            return root

    def inorder(self, root: Node) -> Generator[Node, None, None]:
        """
         This function yields each node in in order format
        Input:
            root: the root node of an avl tree
        Returns:
            a generator of type node
        """
        if root is None:
            return
        yield from self.inorder(root.left)
        yield root
        yield from self.inorder(root.right)

    def preorder(self, root: Node) -> Generator[Node, None, None]:
        """
         This function yields each node in pre order format
        Input:
            root: the root node of an avl tree
        Returns:
            a generator of type node
        """
        if root is None:
            return
        yield root
        yield from self.preorder(root.left)
        yield from self.preorder(root.right)

    def postorder(self, root: Node) -> Generator[Node, None, None]:
        """
         This function yields each node in post order format
        Input:
            root: the root node of an avl tree
        Returns:
            a generator of type node
        """
        if root is None:
            return
        yield from self.postorder(root.left)
        yield from self.postorder(root.right)
        yield root

    def levelorder(self, root: Node) -> Generator[Node, None, None]:
        """
        This function yields each node in level order format
        Input:
            root: the root node of an avl tree
        Returns:
            a generator of type node
        """
        if root is None:
            return
        q = queue.SimpleQueue()
        q.put(root)
        while q.qsize() > 0:
            p = q.get()
            yield p
            if p.left is not None:
                q.put(p.left)
            if p.right is not None:
                q.put(p.right)

    def remove(self, root: Node, val: T) -> Node:
        """
        We give you this function but you should understand its functionality for the exam!!!!
        Remove the node with `value` from the subtree rooted at `root` if it exists.
        Return the root node of the balanced subtree following removal.

        :param root: root node of subtree from which to remove.
        :param val: value to be removed from subtree.
        :return: root node of balanced subtree.
        """
        # handle empty and recursive left/right cases
        if root is None:
            return None
        elif val < root.value:
            root.left = self.remove(root.left, val)
        elif val > root.value:
            root.right = self.remove(root.right, val)
        else:
            # handle actual deletion step on this root
            if root.left is None:
                # pull up right child, set parent, decrease size, properly handle origin-reset
                if root is self.origin:
                    self.origin = root.right
                if root.right is not None:
                    root.right.parent = root.parent
                self.size -= 1
                return root.right
            elif root.right is None:
                # pull up left child, set parent, decrease size, properly handle origin-reset
                if root is self.origin:
                    self.origin = root.left
                if root.left is not None:
                    root.left.parent = root.parent
                self.size -= 1
                return root.left
            else:
                # two children: swap with predecessor and delete predecessor
                predecessor = self.max(root.left)
                root.value = predecessor.value
                root.left = self.remove(root.left, predecessor.value)

        # update height and rebalance every node that was traversed in recursive deletion
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        return self.rebalance(root)


################################## APPLICATION PROBLEM #####################


class Employee(Node):
    "Represents an employee as a modified node"
    __slots__ = ["nominations", "total", "parent", "left", "right", "height"]

    def __init__(self, value: T, parent: Node = None,
                 left: Node = None, right: Node = None, nominations: int = 0, ):
        super().__init__(value, parent, left, right)
        self.nominations = nominations

    def __repr__(self) -> str:
        """
        Represent the Employee node as a string.

        :return: string representation of the node.
        """
        return f"({str(self.value)},{str(self.nominations)})"

    def __str__(self) -> str:
        """
        Represent the Employee node as a string.

        :return: string representation of the node.
        """
        return f"({str(self.value)}, {str(self.nominations)})"

    def __eq__(self, other) -> bool:
        return self.value == other.value and self.nominations == other.nominations


class Company:
    """Represents a company, implemented as a BST"""

    def __init__(self) -> None:
        self.ceo = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree as a string. Inspired by Anna De Biasi (Fall'20 Lead TA).
        :return: string representation of the AVL tree
        """
        if self.ceo is None:
            return "Empty AVL Tree"

        # initialize helpers for tree traversal
        root = self.ceo
        result = ""
        q = queue.SimpleQueue()
        levels = {}
        q.put((root, 0, root.parent))
        for i in range(self.ceo.height + 1):
            levels[i] = []

        # traverse tree to get node representations
        while not q.empty():
            node, level, parent = q.get()
            if level > self.ceo.height:
                break
            levels[level].append((node, level, parent))

            if node is None:
                q.put((None, level + 1, None))
                q.put((None, level + 1, None))
                continue

            if node.left:
                q.put((node.left, level + 1, node))
            else:
                q.put((None, level + 1, None))

            if node.right:
                q.put((node.right, level + 1, node))
            else:
                q.put((None, level + 1, None))

        # construct tree using traversal
        spaces = pow(2, self.ceo.height) * 12
        result += "\n"
        result += f"Sum Tree: size = {self.size}, height = {self.ceo.height}".center(spaces)
        result += "\n\n"
        for i in range(self.ceo.height + 1):
            result += f"Level {i}: "
            for node, level, parent in levels[i]:
                level = pow(2, i)
                space = int(round(spaces / level))
                if node is None:
                    result += " " * space
                    continue
                result += f"{node}".center(space, " ")
            result += "\n"
        return result

    def __str__(self) -> str:
        """
        Represent the AVL tree as a string. Inspired by Anna De Biasi (Fall'20 Lead TA).
        :return: string representation of the AVL tree
        """
        return repr(self)

    def height(self, root: Employee) -> int:
        """
        Return height of a subtree in the AVL tree, properly handling the case of root = None.
        Recall that the height of an empty subtree is -1.
        :param root: root node of subtree to be measured
        :return: height of subtree rooted at `root` parameter
        """
        return root.height if root is not None else -1

    def insert(self, root: Employee, employee: Employee) -> Employee:
        if root is None:
            self.ceo = employee
            self.size = 1
            return self.ceo

        val = employee.value
        if val == root.value:
            return root

        if val > root.value:
            if root.right:
                self.insert(root.right, employee)
            else:
                root.right = employee
                self.size += 1
        else:
            if root.left:
                self.insert(root.left, employee)
            else:
                root.left = employee
                self.size += 1

        root.height = 1 + max(self.height(root.left), self.height(root.right))
        return root


def findEmployeesOfTheMonth(ceo: Employee) -> List[Employee]:
    """
    This function finds the best employees in each hierarchy
    and returns them
    Input:
     ceo: the root employee of a company
    Returns: a list of the most nominated employees for each level
    """
    level = 0
    gone_thru = 0
    temp = ceo
    res = []
    if ceo is None:
        return []
    if ceo.left is None and ceo.right is None:
        return [ceo]
    q = queue.SimpleQueue()
    q.put(ceo)
    while q.qsize() > 0:
        p = q.get()
        if p.nominations > temp.nominations or gone_thru == 0:
            temp = p
        gone_thru += 1
        if gone_thru + 1 > 2 ** level:
            level += 1
            gone_thru = 0
            res.append(temp)
        if p.left is not None:
            q.put(p.left)
        if p.right is not None:
            q.put(p.right)
    if res[-1] != temp:
        res.append(temp)
    return res


############################ Extra Credit Only ############################

class NodeWithSum(Node):
    "Represents a node with sum of subtree that node is the root"
    __slots__ = ["sum", "total", "parent", "left", "right", "height"]

    def __init__(self, value: T, parent: Node = None,
                 left: Node = None, right: Node = None):
        super().__init__(value, parent, left, right)
        self.sum = value

    def __repr__(self) -> str:
        """
        Represent the Employee node as a string.

        :return: string representation of the node.
        """
        return f"({str(self.value)},{str(self.sum)})"

    def __str__(self) -> str:
        """
        Represent the Employee node as a string.

        :return: string representation of the node.
        """
        return f"({str(self.value)}, {str(self.sum)})"

    def __eq__(self, other) -> bool:
        return self.value == other.value and self.sum == other.sum


class TreeWithSum():
    """Represents a Tree, implemented as a BST"""

    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty AVL tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the BST tree as a string. Inspired by Anna De Biasi (Fall'20 Lead TA).

        :return: string representation of the BST tree
        """
        if self.origin is None:
            return "Empty BST Tree"

        # initialize helpers for tree traversal
        root = self.origin
        result = ""
        q = queue.SimpleQueue()
        levels = {}
        q.put((root, 0, root.parent))
        for i in range(self.origin.height + 1):
            levels[i] = []

        # traverse tree to get node representations
        while not q.empty():
            node, level, parent = q.get()
            if level > self.origin.height:
                break
            levels[level].append((node, level, parent))

            if node is None:
                q.put((None, level + 1, None))
                q.put((None, level + 1, None))
                continue

            if node.left:
                q.put((node.left, level + 1, node))
            else:
                q.put((None, level + 1, None))

            if node.right:
                q.put((node.right, level + 1, node))
            else:
                q.put((None, level + 1, None))

        # construct tree using traversal
        spaces = pow(2, self.origin.height) * 12
        result += "\n"
        result += f"BST Tree: size = {self.size}, height = {self.origin.height}".center(spaces)
        result += "\n\n"
        for i in range(self.origin.height + 1):
            result += f"Level {i}: "
            for node, level, parent in levels[i]:
                level = pow(2, i)
                space = int(round(spaces / level))
                if node is None:
                    result += " " * space
                    continue
                result += f"{node}".center(space, " ")
            result += "\n"
        return result

    def __str__(self) -> str:
        """
        Represent the BST tree as a string. Inspired by Anna De Biasi (Fall'20 Lead TA).

        :return: string representation of the AVL tree
        """
        return repr(self)

    def height(self, root: Node) -> int:
        """
        Return height of a subtree in the BST tree, properly handling the case of root = None.
        Recall that the height of an empty subtree is -1.

        :param root: root node of subtree to be measured
        :return: height of subtree rooted at `root` parameter
        """
        return root.height if root is not None else -1

    def insert(self, root: NodeWithSum, val: T) -> NodeWithSum:
        """
        Insert a node with value into the BST
        :param root: root node of subtree in which to insert.
        :param val: value to be inserted in subtree.
        :return: root node of balanced subtree.
        """
        if root is None:
            self.origin = NodeWithSum(val)
            self.size = 1
            return self.origin

        if val == root.value:
            return root

        if val > root.value:
            if root.right:
                self.insert(root.right, val)
            else:
                root.right = NodeWithSum(val)
                self.size += 1
        else:
            if root.left:
                self.insert(root.left, val)
            else:
                root.left = NodeWithSum(val)
                self.size += 1

        root.height = 1 + max(self.height(root.left), self.height(root.right))
        root.sum = root.value + self.subtree_sum(root.left) + self.subtree_sum(root.right)
        return root

    def subtree_sum(self, root: NodeWithSum) -> int:
        """
        Return the sum of all values in the subtree of root
        :param root: root that need to return sum
        :return: sum of subtree of root
        """
        return root.sum if root is not None else 0


def findSum(tree: TreeWithSum, rangeValues: Tuple[int, int]) -> int:
    """
    This function returns the given node sum of nodes within
    a given rang
    Input:
    tree: a given avl tree
    rangeValues: the values the nodes should satisfy
    Returns: the int sum of nodes in a given range
    """
    fin = 0
    root = tree.origin
    # Base Case
    if root is None:
        return 0
    q = queue.SimpleQueue()
    q.put(root)

    while q.qsize() > 0:
        p = q.get()
        if (p.value >= rangeValues[0]) and (p.value <= rangeValues[1]):
            fin += p.value
        if (p.left is not None) and (p.value > rangeValues[0]):
            q.put(p.left)
        if (p.right is not None) and (p.value < rangeValues[1]):
            q.put(p.right)

    return fin

if __name__ == "__main__":
    pass
