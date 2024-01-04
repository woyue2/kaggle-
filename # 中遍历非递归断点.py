# 前中后遍历非递归
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    def __str__(self):
        return str(self.val) + '(' + str(self.left) + ',' + str(self.right) + ')'
root = TreeNode(1)
root.left = TreeNode(5)
root.right = TreeNode(32)
root.left.left = TreeNode(44)
root.left.right = TreeNode(51)
root.right.right = TreeNode(62)
def inorderTraversal(root):
    """
    中 栈
    """
    res = [] 
    stack = []
    node = root
    while stack or node:
        if node:
            print(node)
            stack.append(node)
            node = node.left
        else:
            print('deal start')
            print(f'stack  {[str(node) for node in stack]} -->' )
            node = stack.pop()
            print(f'{[str(node) for node in stack]}' )
            res.append(node.val)
            print(f'res: {res}')
            node = node.right
            print(f"node {node}")
            print('deal end')
    return res
inorderTraversal(root)